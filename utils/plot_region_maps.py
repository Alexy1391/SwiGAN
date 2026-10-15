"""Plot the monthly maps of the SWI, the temperature and the precipitation of a region.

One figure per variable, one panel per month of the requested year, one colour scale shared
by the twelve months so they can be compared. The source is either a region parquet of the
project or a monthly SIM2 file of Météo-France (``MENS_SIM2_*.csv``, whole France), which is
cut to the pixels of the region parquet given by ``--grid``.

    python -m utils.plot_region_maps
    python -m utils.plot_region_maps --year 2010 \
        --filepath "Outputs/rcp_8.5/Grand Est/data_train.parquet"
    python -m utils.plot_region_maps --filepath ../MENS_SIM2_2022.csv \
        --directory "Outputs/rcp_8.5/Grand Est"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.callbacks import MONTHS

SECONDS_PER_DAY = 86400.0
DEFAULT_PARQUET = Path("Outputs/rcp_8.5/Grand Est/Grand_Est_2022_2025.parquet")

# name -> (colourbar label, colormap)
VARIABLES = {
    "swi": ("SWI", "RdYlGn"),
    "temperature": ("Mean temperature (°C)", "RdYlBu_r"),
    "precipitation": ("Precipitation (mm/month)", "Blues"),
}


def load_parquet(filepath: Path, year: int) -> tuple[pd.DataFrame, str]:
    """The year of a region parquet in the plotted units, and the name of its source."""
    df = pd.read_parquet(filepath)
    df = df.loc[df["year"] == year].reset_index(drop=True)
    out = df[["x", "y", "month", "region"]].copy()
    out["swi"] = df["swi_u"]
    out["temperature"] = df["tasAdjust"] - 273.15
    # prtotAdjust_sum is the monthly sum of the daily rates in kg/m2/s.
    out["precipitation"] = df["prtotAdjust_sum"] * SECONDS_PER_DAY
    return out, str(df["scenario"].iloc[0]) if len(df) else ""


def load_sim2(filepath: Path, year: int, grid: Path) -> tuple[pd.DataFrame, str]:
    """The year of a monthly SIM2 file on the pixels of the region parquet ``grid``."""
    df = pd.read_csv(filepath, sep=";")
    df["year"], df["month"] = df["DATE"] // 100, df["DATE"] % 100
    # LAMBX and LAMBY are in hectometres, on the grid of the region files.
    df["x"], df["y"] = df["LAMBX"] * 100.0, df["LAMBY"] * 100.0
    pixels = pd.read_parquet(grid, columns=["x", "y", "region"]).drop_duplicates()
    df = df.loc[df["year"] == year].merge(pixels, on=["x", "y"])
    df = df.rename(columns={"SWI": "swi", "T": "temperature", "PRETOTM": "precipitation"})
    return df[["x", "y", "month", "region", *VARIABLES]], "SIM2"


def plot_variable(df: pd.DataFrame, name: str, year: int, title: str, path: Path) -> None:
    """Twelve monthly maps of one variable on a shared colour scale."""
    label, cmap = VARIABLES[name]
    vmin, vmax = df[name].min(), df[name].max()
    # Lambert coordinates in km, so the pixels keep their true aspect.
    extent = [df["x"].min() / 1e3 - 4, df["x"].max() / 1e3 + 4]
    extent += [df["y"].min() / 1e3 - 4, df["y"].max() / 1e3 + 4]

    fig, axes = plt.subplots(3, 4, figsize=(16, 11), constrained_layout=True)
    for month, ax in zip(range(1, 13), axes.flat):
        sel = df.loc[df["month"] == month]
        grid = sel.pivot(index="y", columns="x", values=name).reindex(
            index=np.sort(df["y"].unique()), columns=np.sort(df["x"].unique())
        )
        p = ax.imshow(
            grid.to_numpy(), origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax
        )
        ax.set_title(f"{MONTHS[month - 1]} {year} (mean {sel[name].mean():.2f})")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(p, ax=axes, shrink=0.6, label=label)
    fig.suptitle(title, fontsize=16)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--filepath", default=DEFAULT_PARQUET, type=Path)
    parser.add_argument("--year", default=2022, type=int)
    parser.add_argument("--directory", default=None, type=Path, help="default: next to the file")
    parser.add_argument(
        "--grid", default=DEFAULT_PARQUET, type=Path, help="region parquet a SIM2 file is cut to"
    )
    args = parser.parse_args()

    if args.filepath.suffix == ".csv":
        df, source = load_sim2(args.filepath, args.year, args.grid)
    else:
        df, source = load_parquet(args.filepath, args.year)
    if df.empty:
        raise ValueError(f"{args.filepath} has no row for {args.year}")
    region = df["region"].iloc[0]

    out_dir = args.directory or args.filepath.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_SIM2" if source == "SIM2" else ""
    for name in VARIABLES:
        path = out_dir / f"{region.replace(' ', '_')}_{args.year}_{name}{suffix}.png"
        plot_variable(df, name, args.year, f"{region} {args.year}, {name} ({source})", path)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
