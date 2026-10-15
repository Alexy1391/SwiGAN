"""Score saved inference trajectories of several regions on one protocol, side by side.

Reads the (M, T, 1, H, W) trajectories written by ``swigan/inference.py`` and scores them against
the observed maps over the same window: the paper's per-pixel metrics, computed exactly as
``swigan/evaluate.py`` computes them (per pixel over the dates, per trajectory, then averaged over
trajectories), and the proper ensemble scores of ``utils/ensemble_scores.py``. Usage:

    python utils/compare_regions.py TAG:PARQUET:STATS_NPZ:TRAJ_NPY [...] [--start 2024-1]
"""

import argparse

import numpy as np
import pandas as pd

from utils.ensemble_scores import score
from utils.metrics import (
    compute_correlation_per_pixel,
    compute_r2_per_pixel,
    compute_rmse_per_pixel,
    compute_smape_per_pixel,
)
from utils.preprocessing import (
    coerce_comma_decimal_columns,
    dataframe_to_rasters,
    fill_all_missing_pixels,
)

INPUT_COLUMNS = [
    "evspsblpotAdjust", "hussAdjust", "prtotAdjust_max", "prtotAdjust_sum", "prtotAdjust_avg",
    "rldsAdjust", "rsdsAdjust", "sfcWindAdjust", "tasAdjust", "tasmaxAdjust", "tasminAdjust",
]
TARGET = "swi_u"


def observed_maps(parquet: str, year: int, month: int, height: int, width: int):
    """Observed SWI maps from (year, month) on, and the region mask -- as evaluate.py builds them."""
    df = fill_all_missing_pixels(pd.read_parquet(parquet), x_dim_col="x", y_dim_col="y")
    df["mask"] = 1.0 * (df["scenario"] != 0)
    idx = sorted(df.index[(df["year"] == float(year)) & (df["month"] == float(month))])[0]
    df = df.iloc[idx:]
    useful = [*INPUT_COLUMNS, "year", "month", "x", "y", TARGET, "mask"]
    df = coerce_comma_decimal_columns(df.drop(columns=[c for c in df if c not in useful]),
                                      [*INPUT_COLUMNS, TARGET])
    _, targets, _, mask = dataframe_to_rasters(df, TARGET, INPUT_COLUMNS, height, width)
    return targets.squeeze(1).astype(np.float32), mask.squeeze(0).astype(bool)


def per_pixel_mean(fn, y_true, traj, mask):
    """Per-pixel metric map averaged over trajectories, as swigan/evaluate.py does."""
    return np.nanmean(np.stack([fn(y_true, t, mask) for t in traj]), axis=0)


def score_region(parquet, stats, traj_path, year, month):
    traj = np.load(traj_path).squeeze(2).astype(np.float32)  # (M, T, H, W)
    y_true, mask = observed_maps(parquet, year, month, *traj.shape[2:])
    if traj.shape[1] != y_true.shape[0]:
        raise ValueError(f"{traj_path}: {traj.shape[1]} generated months, {y_true.shape[0]} observed")
    target_std = float(np.load(stats)["targets_std"])

    rmse = per_pixel_mean(compute_rmse_per_pixel, y_true, traj, mask)[mask]
    smape = per_pixel_mean(compute_smape_per_pixel, y_true, traj, mask)[mask]
    r2 = per_pixel_mean(compute_r2_per_pixel, y_true, traj, mask)[mask]
    rho = per_pixel_mean(compute_correlation_per_pixel, y_true, traj, mask)[mask]

    members, truth = traj[:, :, mask], y_true[:, mask]  # (M, T, P), (T, P)
    ens_mean = members.mean(axis=0)
    step_rmse = np.sqrt(((ens_mean - truth) ** 2).mean(axis=1))
    s = score(members, truth, target_std)
    return {
        "members": members.shape[0],
        "months": members.shape[1],
        "cells": int(mask.sum()),
        "max RMSE": np.nanmax(rmse),
        "p80 RMSE": np.nanpercentile(rmse, 80),
        "mean RMSE (per member)": np.nanmean(rmse),
        "RMSE ens. mean": float(np.sqrt(((ens_mean - truth) ** 2).mean())),
        "RMSE month 1": step_rmse[0],
        "RMSE month 24": step_rmse[-1],
        "bias (pred-obs)": float((ens_mean - truth).mean()),
        "max SMAPE": np.nanmax(smape),
        "p80 SMAPE": np.nanpercentile(smape, 80),
        "p20 R2": np.nanpercentile(r2, 20),
        "p20 rho": np.nanpercentile(rho, 20),
        "CRPS (std units)": s["crps"],
        "CRPS skill": s["crps_skill"],
        "spread-skill": s["spread_skill"],
        "outside envelope %": 100 * s["extreme_frac"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="TAG:PARQUET:STATS_NPZ:TRAJ_NPY")
    parser.add_argument("--start", default="2024-1")
    args = parser.parse_args()
    year, month = (int(v) for v in args.start.split("-"))

    results = {}
    for spec in args.runs:
        tag, parquet, stats, traj = spec.split(":")
        results[tag] = score_region(parquet, stats, traj, year, month)

    table = pd.DataFrame(results)
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 160):
        print(table.to_string())


if __name__ == "__main__":
    main()
