"""Convert daily SIM2 (SAFRAN-ISBA-MODCOU) CSV files into the monthly table SwiGAN trains on.

SwiGAN's eleven covariates were built from the daily DRIAS (ADAMONT-SAFRAN) netCDF files. This
script builds the same eleven columns, in the same units and on the same 8 km grid, from the
Meteo-France "QUOT_SIM2" daily reanalysis, so a region can be trained on observed weather. Usage:

    python utils/sim2_to_monthly.py QUOT_SIM2_2026.csv --output sim2_2026_monthly.parquet

    python utils/sim2_to_monthly.py "sim2/QUOT_SIM2_*.csv.gz" \
        --points Outputs/rcp_8.5/Limousin/data_train.parquet --swi ../../swi \
        --output Outputs/sim2/Limousin/data.parquet

DAILY CONVERSION, applied before the monthly aggregation as on the DRIAS files:

    prtotAdjust        kg m-2 s-1   (PRELIQ + PRENEI) / 86400    rain plus snow, mm/day
    evspsblpotAdjust   kg m-2 s-1   ETP / 86400                  mm/day
    hussAdjust         kg kg-1      Q / 1000                     g/kg
    rldsAdjust         W m-2        DLI * 1e4 / 86400            J/cm2 over the day
    rsdsAdjust         W m-2        SSI * 1e4 / 86400            J/cm2 over the day
    sfcWindAdjust      m s-1        FF
    tasAdjust          K            T + 273.15
    tasmaxAdjust       K            TSUP_H + 273.15
    tasminAdjust       K            TINF_H + 273.15

MONTHLY AGGREGATION, per grid cell. Precipitation gives three columns: ``prtotAdjust_sum`` (sum
of the daily fluxes, so kg m-2 s-1 summed over the days, not mm), ``prtotAdjust_avg`` and
``prtotAdjust_max``. The nine other variables give one column each, reduced by ``--stat``.

``--stat max`` is the default because it is what the DRIAS parquets hold: the pipeline that
built them reduced every non-precipitation variable with ``.max()``, so ``tasminAdjust`` there is
the warmest daily minimum of the month and ``evspsblpotAdjust`` the largest daily ETP. Checked
against the daily netCDF on Limousin, 1987-07 and 2024-02: all nine match the monthly maximum
exactly. ``--stat mean`` gives the conventional monthly mean instead, but a model trained on it
cannot be fed the existing DRIAS parquets. ``--etp`` overrides the statistic for ETP alone.

INCOMPLETE MONTHS. A cell-month is kept only when every calendar day is present with all ten
source values; the others are dropped (``--incomplete nan`` keeps the rows with NaN features,
which the trainer cannot digest). The SIM2 series starts on 1958-08-01, so 1959 is its first
complete year, and the most recent file usually ends mid-month.

GRID. LAMBX and LAMBY are hectometres in Lambert II etendu (EPSG:27572); times 100 they are the
x and y of the DRIAS parquets, node for node. ``--points`` keeps exactly the cells of an
existing table (and carries its lat, lon and region columns over), ``--polygone`` the cells
whose centre falls inside a contour.

TARGET. ``--swi`` merges the monthly uniform SWI (``SWI_UNIF_MENS``) as ``swi_u``, which the
trainer needs; months without SWI are dropped, as the DRIAS pipeline did.

Only numpy, pandas and pyarrow are required. ``--swi`` also needs pyproj and ``--polygone``
geopandas, both in the ``pricing`` extra.
"""

from __future__ import annotations

import argparse
import calendar
import glob
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

SECONDS_PER_DAY = 86_400.0
KELVIN_OFFSET = 273.15
# J/cm2 accumulated over one day -> mean W/m2 over that day.
JOULES_PER_CM2_TO_WATTS_PER_M2 = 1e4 / SECONDS_PER_DAY

KEY_COLUMNS = ("LAMBX", "LAMBY", "DATE")
SOURCE_COLUMNS = ("PRENEI", "PRELIQ", "T", "FF", "Q", "DLI", "SSI", "ETP", "TINF_H", "TSUP_H")

PRECIPITATION = "prtotAdjust"
ETP = "evspsblpotAdjust"
DAILY_CONVERSIONS: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    PRECIPITATION: lambda day: (day["PRELIQ"] + day["PRENEI"]) / SECONDS_PER_DAY,
    ETP: lambda day: day["ETP"] / SECONDS_PER_DAY,
    "hussAdjust": lambda day: day["Q"] / 1000.0,
    "rldsAdjust": lambda day: day["DLI"] * JOULES_PER_CM2_TO_WATTS_PER_M2,
    "rsdsAdjust": lambda day: day["SSI"] * JOULES_PER_CM2_TO_WATTS_PER_M2,
    "sfcWindAdjust": lambda day: day["FF"],
    "tasAdjust": lambda day: day["T"] + KELVIN_OFFSET,
    "tasmaxAdjust": lambda day: day["TSUP_H"] + KELVIN_OFFSET,
    "tasminAdjust": lambda day: day["TINF_H"] + KELVIN_OFFSET,
}
# The eleven covariates, in the column order of the DRIAS parquets.
FEATURE_COLUMNS = [
    "evspsblpotAdjust",
    "hussAdjust",
    "prtotAdjust_avg",
    "prtotAdjust_max",
    "prtotAdjust_sum",
    "rldsAdjust",
    "rsdsAdjust",
    "sfcWindAdjust",
    "tasAdjust",
    "tasmaxAdjust",
    "tasminAdjust",
]
STATIC_COLUMNS = ("lat", "lon", "region")
TARGET_COLUMN = "swi_u"
# The trainer derives its region mask from ``scenario != 0``, so any label marks a real cell.
SCENARIO_LABEL = "sim2"

# The SAFRAN lattice in hectometres of Lambert II etendu: 8 km steps, nodes at x = 40 (mod 80)
# and y = 10 (mod 80). The DRIAS netCDF grid is the same lattice in metres.
HECTOMETRE = 100.0
GRID_STEP_HM = 80
GRID_X_ORIGIN_HM = 40
GRID_Y_ORIGIN_HM = 10
LAMBERT_93_EPSG = 2154
LAMBERT_2_ETENDU_EPSG = 27572
# The SWI maille centres, reprojected from Lambert-93, land within 5 m of their node.
SNAP_TOLERANCE_M = 100.0

# One int64 per cell: LAMBX * CELL_KEY_BASE + LAMBY. LAMBY stays below 30 000 hm.
CELL_KEY_BASE = 100_000
BLOCK_SIZE = 32 << 20
# Partial aggregates are merged whenever this many rows have piled up.
COMPACTION_ROWS = 2_000_000
PARSE_OPTIONS = pacsv.ParseOptions(delimiter=";")


def expand_paths(patterns: Sequence[str], what: str) -> list[Path]:
    """Resolve files, directories and glob patterns into a sorted list of CSV files.

    Globs are expanded here because PowerShell hands them over verbatim.
    """
    paths: list[Path] = []
    for pattern in patterns:
        if Path(pattern).is_dir():
            found = [p for p in Path(pattern).iterdir() if ".csv" in p.suffixes]
        elif glob.has_magic(pattern):
            found = [Path(p) for p in glob.glob(pattern)]
        else:
            found = [Path(pattern)]
        if not found:
            raise FileNotFoundError(f"No {what} file matches '{pattern}'.")
        paths.extend(sorted(found))
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"{what} file(s) not found: {missing}")
    return paths


def cell_key(x_hm: np.ndarray, y_hm: np.ndarray) -> np.ndarray:
    """Pack hectometre grid coordinates into one int64 per cell."""
    return np.asarray(x_hm, dtype=np.int64) * CELL_KEY_BASE + np.asarray(y_hm, dtype=np.int64)


def snap_to_grid(x_m: np.ndarray, y_m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nearest SAFRAN node of metric Lambert II coordinates, and the distance to it in metres."""
    x_m, y_m = np.asarray(x_m, dtype=np.float64), np.asarray(y_m, dtype=np.float64)
    x_hm = GRID_X_ORIGIN_HM + GRID_STEP_HM * np.rint(
        (x_m / HECTOMETRE - GRID_X_ORIGIN_HM) / GRID_STEP_HM
    )
    y_hm = GRID_Y_ORIGIN_HM + GRID_STEP_HM * np.rint(
        (y_m / HECTOMETRE - GRID_Y_ORIGIN_HM) / GRID_STEP_HM
    )
    offset = np.hypot(x_m - HECTOMETRE * x_hm, y_m - HECTOMETRE * y_hm)
    return x_hm.astype(np.int64), y_hm.astype(np.int64), offset


def read_points(path: Path) -> pd.DataFrame:
    """Cells to keep, indexed by cell key, with the static columns the file carries.

    ``path`` is a parquet or CSV table with ``x`` and ``y`` in Lambert II etendu metres, e.g.
    one of the DRIAS ``data_train.parquet`` files.
    """
    if path.suffix == ".parquet":
        available = pq.read_schema(path).names
        columns = ["x", "y", *(c for c in STATIC_COLUMNS if c in available)]
        points = pd.read_parquet(path, columns=columns)
    else:
        points = pd.read_csv(path, sep=None, engine="python")
        points = points[["x", "y", *(c for c in STATIC_COLUMNS if c in points.columns)]]
    points = points.drop_duplicates(["x", "y"])

    x_hm, y_hm, offset = snap_to_grid(points["x"].to_numpy(), points["y"].to_numpy())
    if offset.max() > 1.0:
        raise ValueError(
            f"'{path}' holds points up to {offset.max():.0f} m away from a SAFRAN node. "
            "--points expects x and y in Lambert II etendu metres, as in the DRIAS parquets."
        )
    return points.drop(columns=["x", "y"]).set_index(pd.Index(cell_key(x_hm, y_hm), name="cell"))


def cells_in_polygon(path: Path) -> np.ndarray:
    """Keys of the SAFRAN nodes whose centre falls inside the contour(s) stored in ``path``."""
    try:
        import geopandas as gpd
        import shapely
    except ImportError as exc:
        raise ImportError("--polygone needs geopandas (the 'pricing' extra).") from exc

    contour = gpd.read_file(path)
    if contour.crs is None:
        raise ValueError(f"'{path}' declares no coordinate reference system.")
    shape = contour.geometry.to_crs(LAMBERT_2_ETENDU_EPSG).union_all()

    def nodes(low_m: float, high_m: float, origin_hm: int) -> np.ndarray:
        first = origin_hm + GRID_STEP_HM * np.ceil((low_m / HECTOMETRE - origin_hm) / GRID_STEP_HM)
        return np.arange(first, high_m / HECTOMETRE + 1, GRID_STEP_HM)

    x_min, y_min, x_max, y_max = shape.bounds
    x_hm, y_hm = np.meshgrid(
        nodes(x_min, x_max, GRID_X_ORIGIN_HM), nodes(y_min, y_max, GRID_Y_ORIGIN_HM)
    )
    x_hm, y_hm = x_hm.ravel(), y_hm.ravel()
    inside = shapely.contains_xy(shape, x_hm * HECTOMETRE, y_hm * HECTOMETRE)
    if not inside.any():
        raise ValueError(f"No SAFRAN cell centre falls inside the contour of '{path}'.")
    return cell_key(x_hm[inside], y_hm[inside])


def iter_daily(path: Path) -> Iterator[pd.DataFrame]:
    """Stream one SIM2 daily file as dataframes holding the key and source columns.

    Meteo-France publishes the variables either bare (``PRELIQ``) or with a ``_Q`` suffix
    (``PRELIQ_Q``) depending on the vintage of the file; both are read.
    """
    with pacsv.open_csv(path, parse_options=PARSE_OPTIONS) as reader:
        header = reader.schema.names
    spelling = {name.strip().upper().removesuffix("_Q"): name for name in header}
    missing = [c for c in (*KEY_COLUMNS, *SOURCE_COLUMNS) if c not in spelling]
    if missing:
        raise ValueError(
            f"'{path}' is not a SIM2 daily file: column(s) {missing} not found in {header}."
        )

    convert_options = pacsv.ConvertOptions(
        include_columns=[spelling[c] for c in (*KEY_COLUMNS, *SOURCE_COLUMNS)],
        column_types={
            **{spelling[c]: pa.int32() for c in KEY_COLUMNS},
            **{spelling[c]: pa.float64() for c in SOURCE_COLUMNS},
        },
    )
    with pacsv.open_csv(
        path,
        read_options=pacsv.ReadOptions(block_size=BLOCK_SIZE),
        parse_options=PARSE_OPTIONS,
        convert_options=convert_options,
    ) as reader:
        for batch in reader:
            yield batch.to_pandas().rename(columns={v: k for k, v in spelling.items()})


def required_reductions(stat: str, etp_stat: str) -> list[tuple[str, str]]:
    """The (daily variable, reduction) pairs to accumulate; a mean is a sum over a day count."""
    reductions = [(PRECIPITATION, "sum"), (PRECIPITATION, "max")]
    for variable in DAILY_CONVERSIONS:
        if variable != PRECIPITATION:
            chosen = etp_stat if variable == ETP else stat
            reductions.append((variable, "max" if chosen == "max" else "sum"))
    return reductions


def monthly_partials(
    daily: pd.DataFrame, reductions: Sequence[tuple[str, str]], wanted_cells: np.ndarray | None
) -> tuple[pd.DataFrame, int]:
    """Reduce a chunk of daily rows to one row per (cell, month) present in the chunk.

    A month can straddle chunks and files, so these are partial aggregates -- sums, maxima and
    day counts -- that :func:`merge_partials` combines. Returns them with the number of rows
    skipped for holding a missing source value.
    """
    if daily[list(KEY_COLUMNS)].isna().any(axis=None):
        raise ValueError("The SIM2 file holds rows without LAMBX, LAMBY or DATE.")
    cell = cell_key(daily["LAMBX"].to_numpy(), daily["LAMBY"].to_numpy())
    keep = np.ones(len(daily), dtype=bool) if wanted_cells is None else np.isin(cell, wanted_cells)
    valid = daily[list(SOURCE_COLUMNS)].notna().all(axis=1).to_numpy()
    skipped = int((keep & ~valid).sum())
    keep &= valid
    daily, cell = daily[keep], cell[keep]

    date = daily["DATE"].to_numpy(dtype=np.int64)
    day, month = date % 100, date // 100 % 100
    if ((day < 1) | (day > 31) | (month < 1) | (month > 12)).any():
        raise ValueError("DATE is expected as YYYYMMDD, e.g. 20260131.")

    values = pd.DataFrame({name: convert(daily) for name, convert in DAILY_CONVERSIONS.items()})
    values["cell"] = cell
    values["ym"] = date // 100
    # One bit per day of the month. Summed over a month they equal 2**days - 1 if and only if
    # every day is there exactly once, which is the completeness test of `finalize_months`.
    values["day_bits"] = np.left_shift(np.int64(1), day - 1)

    partial = values.groupby(["cell", "ym"], sort=False).agg(
        n_days=("day_bits", "size"),
        day_bits=("day_bits", "sum"),
        **{f"{variable}:{how}": (variable, how) for variable, how in reductions},
    )
    return partial, skipped


def merge_partials(partials: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Combine partial aggregates: maxima by maximum, sums and day counts by sum."""
    stacked = pd.concat(partials)
    how = {column: "max" if column.endswith(":max") else "sum" for column in stacked.columns}
    return stacked.groupby(level=["cell", "ym"], sort=False).agg(how)


def aggregate_files(
    paths: Sequence[Path],
    reductions: Sequence[tuple[str, str]],
    wanted_cells: np.ndarray | None = None,
) -> pd.DataFrame:
    """Accumulate the monthly sums, maxima and day counts of every cell over all the files."""
    partials: list[pd.DataFrame] = []
    pending_rows = 0
    for path in paths:
        # The files are sorted by cell, not by date, so the span is tracked over every chunk.
        rows, skipped, first, last = 0, 0, 99_999_999, 0
        for daily in iter_daily(path):
            if daily.empty:
                continue
            rows += len(daily)
            first, last = min(first, int(daily["DATE"].min())), max(last, int(daily["DATE"].max()))
            partial, chunk_skipped = monthly_partials(daily, reductions, wanted_cells)
            skipped += chunk_skipped
            partials.append(partial)
            pending_rows += len(partial)
            if pending_rows > COMPACTION_ROWS:
                partials = [merge_partials(partials)]
                pending_rows = len(partials[0])
        print(f"  {path.name}: {rows:,} daily rows, {first} -> {last}")
        if skipped:
            print(f"    {skipped:,} rows skipped for a missing source value")
    if not partials:
        raise ValueError("The SIM2 files hold no data.")
    return merge_partials(partials)


def finalize_months(
    accumulated: pd.DataFrame, stat: str, etp_stat: str, incomplete: str = "drop"
) -> pd.DataFrame:
    """Turn the accumulated aggregates into the eleven covariates, one row per (cell, month)."""
    months = accumulated.reset_index()
    year, month = months["ym"] // 100, months["ym"] % 100
    month_lengths = {
        ym: calendar.monthrange(ym // 100, ym % 100)[1] for ym in months["ym"].unique()
    }
    month_length = months["ym"].map(month_lengths).to_numpy(dtype=np.int64)
    n_days = months["n_days"].to_numpy(dtype=np.int64)

    if (n_days > month_length).any():
        ym = int(months.loc[n_days > month_length, "ym"].iloc[0])
        raise ValueError(
            f"Some cells hold more daily rows than {ym // 100}-{ym % 100:02d} has days: the "
            "input files overlap in time. Pass each period once."
        )
    complete = (n_days == month_length) & (
        months["day_bits"].to_numpy(dtype=np.int64) == np.left_shift(np.int64(1), month_length) - 1
    )

    table = pd.DataFrame(
        {
            "cell": months["cell"],
            "ym": months["ym"],
            "year": year.astype(np.int32),
            "month": month.astype(np.int32),
            "x": (months["cell"] // CELL_KEY_BASE * HECTOMETRE).astype(np.float32),
            "y": (months["cell"] % CELL_KEY_BASE * HECTOMETRE).astype(np.float32),
        }
    )
    table[f"{PRECIPITATION}_sum"] = months[f"{PRECIPITATION}:sum"]
    table[f"{PRECIPITATION}_avg"] = months[f"{PRECIPITATION}:sum"] / n_days
    table[f"{PRECIPITATION}_max"] = months[f"{PRECIPITATION}:max"]
    for variable in DAILY_CONVERSIONS:
        if variable == PRECIPITATION:
            continue
        chosen = etp_stat if variable == ETP else stat
        if chosen == "max":
            table[variable] = months[f"{variable}:max"]
        elif chosen == "mean":
            table[variable] = months[f"{variable}:sum"] / n_days
        else:
            table[variable] = months[f"{variable}:sum"]
    table[FEATURE_COLUMNS] = table[FEATURE_COLUMNS].astype(np.float32)

    if not complete.all():
        short = months.loc[~complete].groupby("ym")["n_days"].agg(["size", "min", "max"])
        print(f"  {int((~complete).sum()):,} incomplete cell-months ({incomplete}):")
        for ym, row in short.iterrows():
            days = row["min"] if row["min"] == row["max"] else f"{row['min']}-{row['max']}"
            print(
                f"    {ym // 100}-{ym % 100:02d}: {row['size']:,} cells with {days} of "
                f"{month_lengths[ym]} days"
            )
        if incomplete == "drop":
            table = table[complete]
        else:
            table.loc[~complete, FEATURE_COLUMNS] = np.nan
    return table.reset_index(drop=True)


def load_swi(paths: Sequence[Path], wanted_cells: np.ndarray | None = None) -> pd.DataFrame:
    """Monthly uniform SWI on the SIM2 grid: columns ``cell``, ``ym`` and ``swi_u``.

    The SWI files locate each maille by its centre in Lambert-93 metres. Reprojected to
    Lambert II etendu those centres land within 5 m of a SAFRAN node, one maille per node, so
    the nearest node is the cell -- no averaging of colliding mailles is needed, unlike on the
    cropped DRIAS grids.
    """
    try:
        from pyproj import Transformer
    except ImportError as exc:
        raise ImportError("--swi needs pyproj to reproject the SWI mailles.") from exc
    to_lambert_2 = Transformer.from_crs(LAMBERT_93_EPSG, LAMBERT_2_ETENDU_EPSG, always_xy=True)

    frames = []
    for path in paths:
        swi = pd.read_csv(path, sep=";", usecols=["LAMBX", "LAMBY", "DATE", "SWI_UNIF_MENS"])
        mailles = swi[["LAMBX", "LAMBY"]].drop_duplicates()
        x_m, y_m = to_lambert_2.transform(
            mailles["LAMBX"].to_numpy(dtype=np.float64), mailles["LAMBY"].to_numpy(dtype=np.float64)
        )
        x_hm, y_hm, offset = snap_to_grid(x_m, y_m)
        if offset.max() > SNAP_TOLERANCE_M:
            raise ValueError(
                f"'{path}': maille centres land up to {offset.max():.0f} m from a SAFRAN node. "
                "LAMBX and LAMBY are expected in Lambert-93 metres."
            )
        mailles = mailles.assign(cell=cell_key(x_hm, y_hm))
        if mailles["cell"].duplicated().any():
            raise ValueError(f"'{path}': several mailles fall on the same SAFRAN node.")
        swi = swi.merge(mailles, on=["LAMBX", "LAMBY"])
        if wanted_cells is not None:
            swi = swi[np.isin(swi["cell"].to_numpy(), wanted_cells)]
        frames.append(
            swi.rename(columns={"DATE": "ym", "SWI_UNIF_MENS": TARGET_COLUMN})[
                ["cell", "ym", TARGET_COLUMN]
            ]
        )
    swi = pd.concat(frames, ignore_index=True).dropna(subset=[TARGET_COLUMN])
    if swi.duplicated(["cell", "ym"]).any():
        raise ValueError("The SWI files overlap in time: a maille has two values for one month.")
    return swi


def month_labels(yms: Sequence[int]) -> str:
    """Compact label of a set of YYYYMM months, e.g. '2026-01 to 2026-10 (10 months)'."""
    yms = sorted(int(ym) for ym in set(yms))
    if not yms:
        return "none"

    def label(ym: int) -> str:
        return f"{ym // 100}-{ym % 100:02d}"

    return f"{label(yms[0])} to {label(yms[-1])} ({len(yms)} months)"


def sim2_to_monthly(
    inputs: Sequence[Path],
    stat: str = "max",
    etp_stat: str | None = None,
    points: pd.DataFrame | None = None,
    polygon_cells: np.ndarray | None = None,
    swi_paths: Sequence[Path] | None = None,
    incomplete: str = "drop",
) -> pd.DataFrame:
    """Build the monthly SwiGAN table from daily SIM2 files.

    Args:
    ----
        inputs: The SIM2 daily CSV files (optionally gzipped), each period once.
        stat: Monthly statistic of the nine non-precipitation variables, "max" or "mean".
            "max" is what the DRIAS parquets hold.
        etp_stat: Statistic of the ETP alone, "max", "mean" or "sum". None follows ``stat``.
        points: Cells to keep, as returned by :func:`read_points`. Its columns are carried
            over to the output.
        polygon_cells: Cell keys to keep, as returned by :func:`cells_in_polygon`.
        swi_paths: Monthly uniform SWI files to merge as ``swi_u``. Months without SWI are
            dropped.
        incomplete: "drop" removes the cell-months that miss a day, "nan" keeps them with NaN
            covariates.

    Returns:
    -------
        One row per (year, month, x, y) with the eleven covariates, sorted like the DRIAS
        parquets.

    """
    etp_stat = etp_stat or stat
    wanted_cells = polygon_cells
    if points is not None:
        wanted_cells = points.index.to_numpy()

    accumulated = aggregate_files(inputs, required_reductions(stat, etp_stat), wanted_cells)
    table = finalize_months(accumulated, stat, etp_stat, incomplete)
    if table.empty:
        raise ValueError("No complete month is left for the requested cells.")

    if wanted_cells is not None:
        found = np.isin(wanted_cells, table["cell"].unique())
        print(f"  {int(found.sum())} of {len(wanted_cells)} requested cells found in SIM2")
        if points is not None and not found.all():
            absent = [
                (int(key // CELL_KEY_BASE * 100), int(key % CELL_KEY_BASE * 100))
                for key in wanted_cells[~found][:5]
            ]
            print(f"  WARNING: {int((~found).sum())} --points cells have no SIM2 data, "
                  f"e.g. (x, y) = {absent}. The region mask will differ from the reference.")
    if points is not None and len(points.columns):
        table = table.join(points, on="cell")
    table["scenario"] = SCENARIO_LABEL

    if swi_paths:
        swi = load_swi(swi_paths, wanted_cells)
        merged = table.merge(swi, on=["cell", "ym"], how="inner")
        lost = set(table["ym"]) - set(merged["ym"])
        print(f"  SWI covers {month_labels(swi['ym'])}; {len(table) - len(merged):,} rows "
              f"without SWI dropped" + (f", months {month_labels(lost)}" if lost else ""))
        if merged.empty:
            raise ValueError(
                f"The SWI files cover {month_labels(swi['ym'])} and the SIM2 files "
                f"{month_labels(table['ym'])}: no month in common for the requested cells."
            )
        table = merged

    yms = sorted(table["ym"].unique())
    expected = pd.period_range(str(yms[0] * 100 + 1), str(yms[-1] * 100 + 1), freq="M")
    gaps = sorted(set(expected.year * 100 + expected.month) - set(yms))
    if gaps:
        print(f"  WARNING: {len(gaps)} month(s) missing inside the series, first "
              f"{gaps[0] // 100}-{gaps[0] % 100:02d}. The trainer reads consecutive rows as "
              "consecutive months.")

    columns = ["year", "month", "x", "y", *(c for c in STATIC_COLUMNS if c in table), "scenario",
               *FEATURE_COLUMNS, *([TARGET_COLUMN] if TARGET_COLUMN in table else [])]
    return table.sort_values(["year", "month", "x", "y"])[columns].reset_index(drop=True)


def main() -> None:
    """Parse the command line, convert and write the parquet."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("inputs", nargs="+",
                        help="SIM2 daily CSV files (.csv or .csv.gz), directories or globs.")
    parser.add_argument("--output", type=Path, required=True, help="Parquet file to write.")
    parser.add_argument("--stat", choices=["max", "mean"], default="max",
                        help="Monthly statistic of the nine non-precipitation variables. "
                             "'max' (default) is what the DRIAS parquets hold.")
    parser.add_argument("--etp", choices=["max", "mean", "sum"], default=None,
                        help="Monthly statistic of the ETP alone. Defaults to --stat.")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--points", type=Path,
                           help="Parquet or CSV with x, y in Lambert II etendu metres (e.g. a "
                                "DRIAS data_train.parquet): keep exactly these cells and carry "
                                "its lat, lon and region columns over.")
    selection.add_argument("--polygone", type=Path,
                           help="Vector file (GeoJSON, shapefile...): keep the cells whose "
                                "centre falls inside its contour(s).")
    parser.add_argument("--swi", nargs="+", default=None,
                        help="Monthly uniform SWI CSV files (SWI_UNIF_MENS), directories or "
                             "globs, merged as swi_u. Months without SWI are dropped.")
    parser.add_argument("--incomplete", choices=["drop", "nan"], default="drop",
                        help="Cell-months missing a day: drop the rows (default) or keep them "
                             "with NaN covariates.")
    args = parser.parse_args()

    inputs = expand_paths(args.inputs, "SIM2")
    swi_paths = expand_paths(args.swi, "SWI") if args.swi else None
    print(f"Converting {len(inputs)} SIM2 file(s), --stat {args.stat}, "
          f"--etp {args.etp or args.stat}")
    table = sim2_to_monthly(
        inputs=inputs,
        stat=args.stat,
        etp_stat=args.etp,
        points=read_points(args.points) if args.points else None,
        polygon_cells=cells_in_polygon(args.polygone) if args.polygone else None,
        swi_paths=swi_paths,
        incomplete=args.incomplete,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.output, index=False)
    cells = table[["x", "y"]].drop_duplicates()
    print(f"Wrote {args.output}: {len(table):,} rows, "
          f"{month_labels(table['year'] * 100 + table['month'])}, {len(cells):,} cells")
    print(f"  dataset.map_dimensions: ({cells['y'].nunique()}, {cells['x'].nunique()})")
    if TARGET_COLUMN not in table:
        print(f"  No '{TARGET_COLUMN}' column: pass --swi to make the file trainable.")


if __name__ == "__main__":
    main()
