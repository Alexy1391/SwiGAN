"""Tests of the SIM2 daily -> monthly conversion that feeds SwiGAN.

The DRIAS parquets the model was trained on hold, per cell and month, the sum / mean / maximum
of the daily precipitation flux and the MAXIMUM of the daily values of the nine other
variables. ``utils/sim2_to_monthly.py`` has to reproduce that from another file format, in
other units, so these tests pin the unit conversions, the statistics, the handling of months
that miss days, and that the result goes through the trainer's own rasterization.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from utils import sim2_to_monthly as sim2
from utils.preprocessing import dataframe_to_rasters, fill_all_missing_pixels

# Three nodes of the SAFRAN lattice, in hectometres: an L-shape on a 2 x 2 grid.
CELLS = [(5400, 20970), (5480, 20970), (5400, 21050)]
SOURCES = ["PRENEI", "PRELIQ", "T", "FF", "Q", "DLI", "SSI", "ETP", "TINF_H", "TSUP_H"]
OTHER_COLUMNS = ["HU", "EVAP", "SWI"]  # present in the real files, never read


def _daily(start: str, end: str) -> pd.DataFrame:
    """Synthetic SIM2 rows, sorted by cell then date like the real files."""
    rng = np.random.default_rng(0)
    dates = pd.date_range(start, end, freq="D")
    rows = pd.DataFrame(
        [(x, y, int(d.strftime("%Y%m%d"))) for x, y in CELLS for d in dates],
        columns=["LAMBX", "LAMBY", "DATE"],
    )
    n = len(rows)
    rows["PRENEI"] = rng.choice([0.0, 0.0, 1.5], n)
    rows["PRELIQ"] = rng.gamma(0.4, 6.0, n).round(1)
    rows["T"] = rng.normal(12.0, 6.0, n).round(1)
    rows["FF"] = rng.gamma(4.0, 0.7, n).round(1)
    rows["Q"] = rng.uniform(2.0, 14.0, n).round(3)
    rows["DLI"] = rng.uniform(2000.0, 3600.0, n).round(1)
    rows["SSI"] = rng.uniform(100.0, 3000.0, n).round(1)
    rows["HU"] = 80.0
    rows["EVAP"] = 1.0
    rows["ETP"] = rng.uniform(0.0, 7.0, n).round(1)
    rows["SWI"] = 0.5
    rows["TINF_H"] = rows["T"] - 4.0
    rows["TSUP_H"] = rows["T"] + 5.0
    return rows


def _write(rows: pd.DataFrame, path: Path, suffix: str = "") -> Path:
    renamed = rows.rename(columns={c: c + suffix for c in [*SOURCES, *OTHER_COLUMNS]})
    renamed.to_csv(path, sep=";", index=False)
    return path


def _cell_month(table: pd.DataFrame, cell: tuple[int, int], year: int, month: int) -> pd.Series:
    row = table[
        (table["x"] == cell[0] * 100)
        & (table["y"] == cell[1] * 100)
        & (table["year"] == year)
        & (table["month"] == month)
    ]
    assert len(row) == 1
    return row.iloc[0]


def test_units_and_statistics_reproduce_the_drias_convention(tmp_path: Path) -> None:
    """Default run: precipitation sum/mean/max of the flux, monthly maximum of the rest."""
    rows = _daily("2023-02-01", "2023-02-28")
    table = sim2.sim2_to_monthly([_write(rows, tmp_path / "sim2.csv")])

    assert len(table) == len(CELLS)
    assert list(table.columns) == ["year", "month", "x", "y", "scenario", *sim2.FEATURE_COLUMNS]
    assert table["scenario"].eq("sim2").all()

    cell = CELLS[1]
    days = rows[(rows["LAMBX"] == cell[0]) & (rows["LAMBY"] == cell[1])]
    got = _cell_month(table, cell, 2023, 2)
    flux = (days["PRELIQ"] + days["PRENEI"]) / 86400.0  # mm/day -> kg m-2 s-1
    expected = {
        "prtotAdjust_sum": flux.sum(),
        "prtotAdjust_avg": flux.mean(),
        "prtotAdjust_max": flux.max(),
        "evspsblpotAdjust": days["ETP"].max() / 86400.0,
        "hussAdjust": days["Q"].max() / 1000.0,  # g/kg -> kg/kg
        "rldsAdjust": days["DLI"].max() * 1e4 / 86400.0,  # J/cm2 per day -> W/m2
        "rsdsAdjust": days["SSI"].max() * 1e4 / 86400.0,
        "sfcWindAdjust": days["FF"].max(),
        "tasAdjust": days["T"].max() + 273.15,
        "tasmaxAdjust": days["TSUP_H"].max() + 273.15,
        "tasminAdjust": days["TINF_H"].max() + 273.15,  # the warmest daily minimum
    }
    for column, value in expected.items():
        assert got[column] == pytest.approx(value, rel=1e-6), column
    assert set(expected) == set(sim2.FEATURE_COLUMNS)


def test_mean_statistic_and_etp_override(tmp_path: Path) -> None:
    """--stat mean averages the nine other variables; --etp picks the ETP statistic alone."""
    rows = _daily("2023-02-01", "2023-02-28")
    path = _write(rows, tmp_path / "sim2.csv")
    cell = CELLS[0]
    days = rows[(rows["LAMBX"] == cell[0]) & (rows["LAMBY"] == cell[1])]

    got = _cell_month(sim2.sim2_to_monthly([path], stat="mean", etp_stat="sum"), cell, 2023, 2)
    assert got["tasAdjust"] == pytest.approx(days["T"].mean() + 273.15, rel=1e-6)
    assert got["sfcWindAdjust"] == pytest.approx(days["FF"].mean(), rel=1e-6)
    assert got["evspsblpotAdjust"] == pytest.approx(days["ETP"].sum() / 86400.0, rel=1e-6)
    # Precipitation does not depend on --stat.
    assert got["prtotAdjust_max"] == pytest.approx(
        (days["PRELIQ"] + days["PRENEI"]).max() / 86400.0, rel=1e-6
    )

    got = _cell_month(sim2.sim2_to_monthly([path], stat="mean"), cell, 2023, 2)
    assert got["evspsblpotAdjust"] == pytest.approx(days["ETP"].mean() / 86400.0, rel=1e-6)


def test_incomplete_months_are_dropped_or_blanked(tmp_path: Path) -> None:
    """A month missing a day, at the end of the file or inside it, is not aggregated."""
    rows = _daily("2023-01-01", "2023-03-10")  # March stops on the 10th
    hole = (rows["LAMBX"] == CELLS[0][0]) & (rows["LAMBY"] == CELLS[0][1]) & (
        rows["DATE"] == 20230214
    )
    path = _write(rows[~hole], tmp_path / "sim2.csv")

    dropped = sim2.sim2_to_monthly([path])
    assert set(dropped["month"]) == {1, 2}
    assert len(dropped) == 2 * len(CELLS) - 1  # February of the first cell is gone

    blanked = sim2.sim2_to_monthly([path], incomplete="nan")
    assert len(blanked) == 3 * len(CELLS)
    nan_rows = blanked[sim2.FEATURE_COLUMNS].isna().all(axis=1)
    assert int(nan_rows.sum()) == len(CELLS) + 1
    assert blanked.loc[~nan_rows, sim2.FEATURE_COLUMNS].notna().all(axis=None)


def test_missing_source_value_makes_the_month_incomplete(tmp_path: Path) -> None:
    """A day without one of the ten source values does not count as a day."""
    rows = _daily("2023-02-01", "2023-02-28")
    rows.loc[5, "SSI"] = np.nan  # first cell, February 6th
    table = sim2.sim2_to_monthly([_write(rows, tmp_path / "sim2.csv")])
    assert len(table) == len(CELLS) - 1


def test_month_straddling_chunks_and_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Partial aggregates of one month merge across read chunks and across files."""
    rows = _daily("2023-01-01", "2023-02-28")
    whole = sim2.sim2_to_monthly([_write(rows, tmp_path / "whole.csv")])

    first = _write(rows[rows["DATE"] <= 20230117], tmp_path / "a.csv")
    second = _write(rows[rows["DATE"] > 20230117], tmp_path / "b.csv.gz")
    monkeypatch.setattr(sim2, "BLOCK_SIZE", 2048)  # a few dozen rows per chunk
    pieces = sim2.sim2_to_monthly([first, second])

    pd.testing.assert_frame_equal(pieces, whole)


def test_overlapping_files_are_refused(tmp_path: Path) -> None:
    """The same days passed twice would double the sums, so it is an error."""
    rows = _daily("2023-02-01", "2023-02-28")
    paths = [_write(rows, tmp_path / "a.csv"), _write(rows, tmp_path / "b.csv")]
    with pytest.raises(ValueError, match="overlap"):
        sim2.sim2_to_monthly(paths)


def test_q_suffixed_header_is_read(tmp_path: Path) -> None:
    """Meteo-France also ships the columns as PRELIQ_Q, T_Q, Q_Q, TINF_H_Q..."""
    rows = _daily("2023-02-01", "2023-02-28")
    bare = sim2.sim2_to_monthly([_write(rows, tmp_path / "bare.csv")])
    suffixed = sim2.sim2_to_monthly([_write(rows, tmp_path / "suffixed.csv", suffix="_Q")])
    pd.testing.assert_frame_equal(suffixed, bare)


def test_points_keep_the_reference_cells_and_their_columns(tmp_path: Path) -> None:
    """--points restricts to the cells of an existing table and carries lat, lon, region."""
    path = _write(_daily("2023-02-01", "2023-02-28"), tmp_path / "sim2.csv")
    reference = pd.DataFrame(
        {
            "year": [1960, 1960, 1961],
            "x": np.array([540000.0, 540000.0, 540000.0], dtype=np.float32),
            "y": np.array([2097000.0, 2105000.0, 2097000.0], dtype=np.float32),
            "lat": np.array([45.9, 46.0, 45.9], dtype=np.float32),
            "lon": np.array([1.5, 1.5, 1.5], dtype=np.float32),
            "region": "Limousin",
        }
    )
    reference.to_parquet(tmp_path / "reference.parquet", index=False)

    table = sim2.sim2_to_monthly([path], points=sim2.read_points(tmp_path / "reference.parquet"))
    leading = ["year", "month", "x", "y", "lat", "lon", "region", "scenario"]
    assert list(table.columns[:8]) == leading
    assert sorted(zip(table["x"], table["y"])) == [(540000.0, 2097000.0), (540000.0, 2105000.0)]
    assert table["region"].eq("Limousin").all()
    assert table.loc[table["y"] == 2105000.0, "lat"].iloc[0] == pytest.approx(46.0)


def test_points_off_the_grid_are_refused(tmp_path: Path) -> None:
    """Coordinates that are not SAFRAN nodes in Lambert II metres are a unit mistake."""
    pd.DataFrame({"x": [5400.0], "y": [20970.0]}).to_parquet(tmp_path / "hm.parquet")
    with pytest.raises(ValueError, match="Lambert II"):
        sim2.read_points(tmp_path / "hm.parquet")


def test_output_goes_through_the_trainer_rasterization(tmp_path: Path) -> None:
    """The table rasterizes as swigan/train.py does it: every month kept, mask on the cells."""
    path = _write(_daily("2023-01-01", "2023-04-30"), tmp_path / "sim2.csv")
    table = sim2.sim2_to_monthly([path])
    table["swi_u"] = 0.5

    filled = fill_all_missing_pixels(table, x_dim_col="x", y_dim_col="y")
    filled["mask"] = 1.0 * (filled["scenario"] != 0)
    maps, targets, timestamps, mask = dataframe_to_rasters(
        filled, "swi_u", sim2.FEATURE_COLUMNS, height=2, width=2
    )
    assert maps.shape == (4, 11, 2, 2)
    assert targets.shape == (4, 1, 2, 2)
    assert timestamps.ravel().tolist() == [0.0, 1.0, 2.0, 3.0]
    # Rows are sorted by y then x: the missing corner is (x = 548000, y = 2105000).
    assert mask.reshape(2, 2).tolist() == [[1.0, 1.0], [1.0, 0.0]]
    assert np.isfinite(maps).all()


def test_swi_is_merged_on_the_sim2_grid(tmp_path: Path) -> None:
    """--swi reprojects the Lambert-93 maille centres onto the SIM2 nodes and merges swi_u."""
    pyproj = pytest.importorskip("pyproj")
    to_lambert_93 = pyproj.Transformer.from_crs(27572, 2154, always_xy=True)
    swi_rows = []
    for number, (x_hm, y_hm) in enumerate(CELLS[:2], start=1):  # no SWI for the third cell
        x93, y93 = to_lambert_93.transform(x_hm * 100.0, y_hm * 100.0)
        for ym, value in ((202301, 0.9), (202302, 0.8)):  # no SWI for March
            swi_rows.append((number, round(x93), round(y93), ym, value + number / 100))
    swi_path = tmp_path / "swi.202301-202312.csv"
    pd.DataFrame(
        swi_rows, columns=["NUMERO", "LAMBX", "LAMBY", "DATE", "SWI_UNIF_MENS"]
    ).to_csv(swi_path, sep=";", index=False)

    path = _write(_daily("2023-01-01", "2023-03-31"), tmp_path / "sim2.csv")
    table = sim2.sim2_to_monthly([path], swi_paths=[swi_path])

    assert table.columns[-1] == "swi_u"
    assert len(table) == 4  # two cells x two months
    assert set(table["month"]) == {1, 2}
    assert _cell_month(table, CELLS[1], 2023, 2)["swi_u"] == pytest.approx(0.82)


def test_polygon_selects_the_cells_whose_centre_is_inside(tmp_path: Path) -> None:
    """--polygone keeps the nodes inside a contour given in any reference system."""
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box

    # Around the two southern cells, 1 km of margin, handed over in longitude / latitude.
    contour = gpd.GeoSeries([box(539000, 2096000, 549000, 2098000)], crs=27572).to_crs(4326)
    contour.to_file(tmp_path / "contour.geojson", driver="GeoJSON")

    cells = sim2.cells_in_polygon(tmp_path / "contour.geojson")
    assert sorted(cells.tolist()) == sorted(sim2.cell_key(*np.array(CELLS[:2]).T).tolist())

    path = _write(_daily("2023-02-01", "2023-02-28"), tmp_path / "sim2.csv")
    table = sim2.sim2_to_monthly([path], polygon_cells=cells)
    assert sorted(zip(table["x"], table["y"])) == [(540000.0, 2097000.0), (548000.0, 2097000.0)]
