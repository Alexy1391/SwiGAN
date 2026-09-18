"""Script to launch inference of a trained model."""

import logging
import os
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from lightning import LightningModule
from omegaconf import DictConfig
from tqdm import tqdm

from swigan.engines.swigan_lit import TTTSWIGAN
from utils.dispersion import (
    apply_seasonal_offsets,
    load_dispersion_spec,
    member_perturbations,
    month_index,
)
from utils.inference_modes import dropout_active
from utils.preprocessing import (
    coerce_comma_decimal_columns,
    dataframe_to_rasters,
    fill_all_missing_pixels,
    standardize_swi_history,
)

logger = logging.getLogger(__file__)


def compute_trajectory_iterative(
    model: LightningModule,
    inputs_maps: torch.Tensor,
    starting_target_maps: torch.Tensor,
    input_timestamps: torch.Tensor,
    mask: torch.Tensor,
    z: torch.Tensor,
    dtype: np.dtype,
    history_scale: torch.Tensor | None = None,
    history_shift: torch.Tensor | None = None,
) -> np.ndarray:
    """Predict iteratively n timestamps k timesteps at a time.

    ``z`` carries one independent noise draw per rollout step, of shape
    (num_steps, num_trajectories, z_dim). Step k is generated from ``z[k]``, following the
    iterative procedure of Section 2.2 of the paper: the Z_k are i.i.d. across k, so a
    trajectory's deviations do not share a single fixed noise vector across all 24 months.

    ``history_scale`` and ``history_shift``, one value per trajectory, transform the SWI
    history the generator SEES at every step -- ``history * scale + shift`` on the region, in
    the standardised units the history channels are in -- and leave the buffer that is fed
    back untouched, so a trajectory's own predictions are never rescaled twice. Scale 0 is a
    climatologically neutral history; a shift is a persistent initial-condition perturbation.
    See :mod:`utils.dispersion`. With both ``None`` this is the unperturbed rollout, bit for
    bit.
    """
    num_trajectories = z.shape[1]
    target_maps = starting_target_maps.unsqueeze(0).repeat(num_trajectories, 1, 1, 1)
    perturbed = history_scale is not None or history_shift is not None
    if perturbed:
        device = inputs_maps.device
        scale = (
            torch.ones(num_trajectories, device=device)
            if history_scale is None
            else history_scale.to(device).float()
        )
        shift = (
            torch.zeros(num_trajectories, device=device)
            if history_shift is None
            else history_shift.to(device).float()
        )
        if scale.numel() != num_trajectories or shift.numel() != num_trajectories:
            raise ValueError("history_scale and history_shift need one value per trajectory")
        scale = scale.view(num_trajectories, 1, 1, 1)
        shift = shift.view(num_trajectories, 1, 1, 1)
        history_mask = mask.to(device).float()
    outputs = []
    with torch.no_grad():
        for idx in range(0, len(inputs_maps)):
            seen = (target_maps * scale + shift) * history_mask if perturbed else target_maps
            inputs = torch.cat(
                [inputs_maps[idx].unsqueeze(0).repeat(num_trajectories, 1, 1, 1), seen],
                dim=1,
            )
            timestamps = input_timestamps[idx].unsqueeze(0).repeat(num_trajectories)

            current_output = model(
                inputs.to(inputs_maps.device),
                timestamps.to(inputs_maps.device),
                mask.unsqueeze(0).to(inputs_maps.device),
                z[idx].to(inputs_maps.device),
            )
            target_maps = torch.cat([target_maps, current_output], dim=1)[:, 1:]
            outputs.append(current_output.unsqueeze(1))

    outputs = torch.cat(outputs, dim=1).detach().cpu().numpy()

    # Unnormalize the predictions
    mean_v, std_v = model.statistics["targets_mean"], model.statistics["targets_std"]
    outputs = outputs * std_v + mean_v

    # Apply the mask one final time
    outputs = outputs * mask.cpu().numpy()
    return outputs.astype(dtype)


@hydra.main(config_path="../config", config_name="inference_swigan", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Train the model."""
    dataset_cfg = cfg["dataset"]
    model_cfg = cfg["model"]

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

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
        input_df = fill_all_missing_pixels(
            input_df,
            x_dim_col=x_dim_col,
            y_dim_col=y_dim_col,
        )

    # Create a mask column of the pixels of interest.
    input_df["mask"] = 1.0 * (input_df.loc[:, "scenario"] != 0)

    # Get the index of the row corresponding to the starting date
    starting_year = dataset_cfg["starting_year"]
    starting_month = dataset_cfg["starting_month"]
    all_indices = input_df.loc[
        (input_df["year"] == float(starting_year)) & (input_df["month"] == float(starting_month))
    ].index.tolist()
    idx = sorted(all_indices)[0]

    # Split the dataset to retrieve all months after this starting date
    starting_dataset, input_dataset = input_df.iloc[:idx], input_df.iloc[idx:]

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
    starting_dataset = starting_dataset.drop(columns=columns_to_drop)

    input_dataset = coerce_comma_decimal_columns(input_dataset, [*feature_columns, target_column])
    starting_dataset = coerce_comma_decimal_columns(
        starting_dataset, [*feature_columns, target_column]
    )

    # Load the model from the checkpoint
    if not os.path.exists(model_cfg["checkpoint_path"]):
        raise ValueError(f"No checkpoints found at '{model_cfg['checkpoint_path']}'")

    logger.info(f"Loading the model from '{model_cfg['checkpoint_path']}'...")
    model = TTTSWIGAN.load_from_checkpoint(
        model_cfg["checkpoint_path"], loss_fn="l1", device=device
    )
    train_dataset_statistics = np.load(model_cfg["train_dataset_statistics_path"])
    model.statistics = train_dataset_statistics

    # ``load_from_checkpoint`` returns the module in train mode. Free-running inference must
    # run in eval mode, otherwise stochastic depth randomly drops residual blocks and
    # BatchNorm uses the statistics of the batch of fed-back predictions at every one of the
    # rollout steps ("During inference, all residual blocks are used", paper Section 8.2.1).
    model.eval()

    # Extract features
    map_height, map_width = model.hparams.input_map_dims
    feature_maps, _, timestamps, mask = dataframe_to_rasters(
        input_dataset, target_column, feature_columns, map_height, map_width
    )
    _, starting_targets, _, _ = dataframe_to_rasters(
        starting_dataset, target_column, feature_columns, map_height, map_width
    )

    # Extract initial target maps
    num_input_steps = model.hparams.input_channels - len(feature_columns)
    starting_target_maps = starting_targets[-num_input_steps:].squeeze(
        1
    )  # the first targets are the initial ones

    # Prepare dataset
    mean_value, std_value = model.statistics["feats_mean"], model.statistics["feats_std"]
    feature_maps = (feature_maps - mean_value) / std_value
    feature_maps = np.where(mask.squeeze(), feature_maps, 0.0)

    # The SWI history channels get the same treatment as the covariates above: the trainer
    # standardizes the target maps before they are ever fed back as history, and
    # `dataframe_to_rasters` hands us raw SWI. Without this the first `num_input_steps` months
    # of every rollout are in the wrong unit system while every step after them -- the
    # generator's own output -- is already standardized.
    starting_target_maps = standardize_swi_history(starting_target_maps, model.statistics, mask)

    # Compute the trajectories. The Z_k of Section 2.2 are i.i.d. across rollout steps, so
    # draw one noise vector per step and per trajectory: (num_steps, num_trajectories, z_dim).
    num_trajectories = model_cfg["num_trajectories"]
    num_steps = len(feature_maps)
    z = torch.randn(num_steps, num_trajectories, model.hparams.z_dim) * model_cfg["noise_std"]

    # Disperse on the state, not the noise: k neutral-history members and a per-member history
    # shift, selected out of sample by utils/dispersion_select.py, plus the seasonal offset it
    # fitted. Null reproduces the unperturbed rollout bit for bit.
    dispersion = model_cfg.get("dispersion")
    spec = load_dispersion_spec(dispersion) if dispersion else None
    scale, shift = (
        member_perturbations(
            num_trajectories, spec["neutral_members"], spec["state_sd"], spec["member_seed"]
        )
        if spec is not None
        else (None, None)
    )
    if spec is not None:
        logger.info(
            f"State dispersion: {spec['neutral_members']} neutral members, state sd "
            f"{spec['state_sd']}, seasonal offsets "
            f"{'on' if spec['seasonal_offsets'] is not None else 'off'}."
        )

    trajectories = []
    batch_size = model_cfg["num_trajectories_per_batch"]
    # Inference dropout: the generator's dropout layers left active in eval mode, which every
    # ensemble score in the reports was measured with (utils/dropout_probe.py). Off reproduces
    # the original inference path.
    inference_dropout = bool(model_cfg.get("inference_dropout", False))
    logger.info(
        f"Running inference for {num_trajectories} trajectories"
        f"{' with inference dropout' if inference_dropout else ''}..."
    )
    for idx in tqdm(range(0, num_trajectories, batch_size)):
        z_vec = z[:, idx : idx + batch_size].to(device)
        with dropout_active(model, inference_dropout):
            traj = compute_trajectory_iterative(
                model=model,
                inputs_maps=torch.tensor(feature_maps * mask.squeeze(), device=device).float(),
                starting_target_maps=torch.tensor(
                    starting_target_maps * mask.squeeze(), device=device
                ).float(),
                input_timestamps=torch.tensor(timestamps.squeeze(), device=device).long(),
                mask=torch.tensor(mask, device=device).float(),
                z=z_vec,
                dtype=np.dtype(model_cfg["dtype"]),
                history_scale=None if scale is None else scale[idx : idx + batch_size],
                history_shift=None if shift is None else shift[idx : idx + batch_size],
            )
        trajectories.append(traj)

    trajectories = np.concat(trajectories, axis=0)
    if spec is not None and spec["seasonal_offsets"] is not None:
        # Physical SWI units, per calendar month, on the region only -- the offsets were
        # fitted on un-standardised ensemble means.
        trajectories = apply_seasonal_offsets(
            trajectories, month_index(timestamps), spec["seasonal_offsets"], mask
        ).astype(trajectories.dtype)
    np.save(model_cfg["saving_path"], trajectories)
    logger.info(f"Saved outputs to '{model_cfg['saving_path']}' .")


if __name__ == "__main__":
    main()
