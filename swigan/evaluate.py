"""Script to evaluate generated SWI trajectories against observed data."""

import ast
import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from utils.metrics import (
    compute_correlation_per_pixel,
    compute_mae_per_pixel,
    compute_mse_per_pixel,
    compute_r2_per_pixel,
    compute_rmse_per_pixel,
    compute_smape_per_pixel,
)
from utils.preprocessing import (
    coerce_comma_decimal_columns,
    dataframe_to_rasters,
    fill_all_missing_pixels,
)

logger = logging.getLogger(__file__)

METRIC_FUNCTIONS = {
    "mse": compute_mse_per_pixel,
    "rmse": compute_rmse_per_pixel,
    "mae": compute_mae_per_pixel,
    "smape": compute_smape_per_pixel,
    "r2": compute_r2_per_pixel,
    "correlation": compute_correlation_per_pixel,
}


@hydra.main(config_path="../config", config_name="evaluate_swigan", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Compute per-pixel evaluation metrics on generated SWI trajectories.

    Compares generated trajectories (produced by `swigan.inference`) against the
    observed SWI maps over the same period, following the evaluation protocol described
    in the SwiGAN paper: for each pixel, each metric is computed over the time axis for
    every generated trajectory, then averaged across trajectories.
    """
    dataset_cfg = cfg["dataset"]
    eval_cfg = cfg["evaluation"]

    input_df: pd.DataFrame
    filepath = dataset_cfg["filepath"]
    if Path(filepath).suffix == ".csv":
        input_df = pd.read_csv(filepath)
    elif Path(filepath).suffix == ".tsv":
        input_df = pd.read_csv(filepath, sep="/t")
    elif Path(filepath).suffix == ".parquet":
        input_df = pd.read_parquet(filepath)
    else:
        raise NotImplementedError(
            "Unsupported file format for the input dataset. Please provide one of"
            " the following formats: .csv, .tsv, .parquet."
        )

    x_dim_col = dataset_cfg["x_dim_column"] if dataset_cfg["x_dim_column"] else "x"
    y_dim_col = dataset_cfg["y_dim_column"] if dataset_cfg["y_dim_column"] else "y"

    if dataset_cfg["fill_missing_pixels"]:
        logger.info("Filling missing pixels in the maps...")
        input_df = fill_all_missing_pixels(input_df, x_dim_col=x_dim_col, y_dim_col=y_dim_col)

    # Create a mask column of the pixels of interest.
    input_df["mask"] = 1.0 * (input_df.loc[:, "scenario"] != 0)

    # Keep only the observed months corresponding to the generated trajectories.
    starting_year = dataset_cfg["starting_year"]
    starting_month = dataset_cfg["starting_month"]
    all_indices = input_df.loc[
        (input_df["year"] == float(starting_year)) & (input_df["month"] == float(starting_month))
    ].index.tolist()
    idx = sorted(all_indices)[0]
    input_dataset = input_df.iloc[idx:]

    # Drop unnecessary columns
    feature_columns = dataset_cfg["input_columns"]
    target_column = dataset_cfg["target_column"]
    useful_columns = feature_columns + [
        "year",
        "month",
        x_dim_col,
        y_dim_col,
        target_column,
        "mask",
    ]
    columns_to_drop = [col for col in input_dataset.columns if col not in useful_columns]
    input_dataset = input_dataset.drop(columns=columns_to_drop)
    input_dataset = coerce_comma_decimal_columns(input_dataset, [*feature_columns, target_column])

    # Extract the observed target maps
    map_height, map_width = ast.literal_eval(dataset_cfg["map_dimensions"])
    _, target_maps, _, mask = dataframe_to_rasters(
        input_dataset, target_column, feature_columns, map_height, map_width
    )
    y_true = target_maps.squeeze(1)  # (n_dates, H, W)
    mask = mask.squeeze(0).astype(bool)  # (H, W)

    logger.info(f"Loading generated trajectories from '{eval_cfg['trajectories_path']}'...")
    trajectories = np.load(eval_cfg["trajectories_path"])  # (n_trajectories, n_dates, 1, H, W)
    trajectories = trajectories.squeeze(2)  # (n_trajectories, n_dates, H, W)

    if trajectories.shape[1] != y_true.shape[0]:
        raise ValueError(
            f"The number of generated months ({trajectories.shape[1]}) does not match "
            f"the number of observed months ({y_true.shape[0]}). Make sure 'starting_year' "
            "and 'starting_month' match the inference run that produced these trajectories."
        )

    logger.info(f"Computing metrics over {trajectories.shape[0]} trajectories...")
    metrics = {}
    for name, metric_fn in METRIC_FUNCTIONS.items():
        per_trajectory_maps = np.stack(
            [
                metric_fn(y_true, trajectories[traj_idx], mask)
                for traj_idx in range(trajectories.shape[0])
            ],
            axis=0,
        )
        metric_map = np.nanmean(per_trajectory_maps, axis=0).astype(np.float32)
        metrics[name] = metric_map
        logger.info(
            f"{name}: mean={np.nanmean(metric_map):.4f}, "
            f"median={np.nanmedian(metric_map):.4f}, "
            f"max={np.nanmax(metric_map):.4f}"
        )

    logger.info(f"Saving per-pixel metric maps to '{eval_cfg['saving_path']}'...")
    np.savez(eval_cfg["saving_path"], **metrics)


if __name__ == "__main__":
    main()