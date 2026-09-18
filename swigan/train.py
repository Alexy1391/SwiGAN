"""Script to launch the training."""

import ast
import datetime
import logging
import os
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from swigan.engines.swigan_lit import TTTSWIGAN
from utils.callbacks import RandomTimeStepMapCallback, RolloutMetricsCallback
from utils.preprocessing import (
    coerce_comma_decimal_columns,
    dataframe_to_rasters,
    fill_all_missing_pixels,
)
from utils.swi_dataset import build_datasets_from_split_rasters, build_train_val_test_datasets

logger = logging.getLogger(__file__)


def read_tabular(filepath: str) -> pd.DataFrame:
    """Read the input dataset from any of the supported tabular formats."""
    suffix = Path(filepath).suffix
    if suffix == ".csv":
        return pd.read_csv(filepath)
    if suffix == ".tsv":
        return pd.read_csv(filepath, sep="/t")
    if suffix == ".parquet":
        return pd.read_parquet(filepath)
    raise NotImplementedError(
        "Unsupported file format for the input dataset. Please provide one of"
        " the following formats: .csv, .tsv, .parquet."
    )


def load_rasters(
    filepath: str,
    dataset_cfg: DictConfig,
    feature_columns: list[str],
    target_column: str,
    map_height: int,
    map_width: int,
    x_dim_col: str,
    y_dim_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Turn one tabular file into (feature maps, target maps, timestamps, mask)."""
    input_df = read_tabular(filepath)

    if dataset_cfg["fill_missing_pixels"]:
        logger.info(f"Filling missing pixels in the maps of '{filepath}'...")
        input_df = fill_all_missing_pixels(input_df, x_dim_col=x_dim_col, y_dim_col=y_dim_col)

    # Create a mask column of the pixels of interest. The rows added just above are the ones
    # filled with 0.0, so this keys off the fill value rather than off any scenario label.
    input_df["mask"] = 1.0 * (input_df.loc[:, "scenario"] != 0)

    # Drop unnecessary columns
    useful_columns = feature_columns + [
        "year",
        "month",
        x_dim_col,
        y_dim_col,
        target_column,
        "mask",
    ]
    input_df = input_df.drop(columns=[c for c in input_df.columns if c not in useful_columns])
    input_df = coerce_comma_decimal_columns(input_df, [*feature_columns, target_column])

    feature_maps, targets, timestamps, mask = dataframe_to_rasters(
        input_df, target_column, feature_columns, map_height, map_width, x_dim_col, y_dim_col
    )
    if mask is None:
        raise ValueError(
            f"No complete {map_height}x{map_width} month could be built from '{filepath}'. "
            f"Check 'dataset.map_dimensions' against the grid of that file."
        )
    return feature_maps, targets, timestamps, mask


def build_splits(
    dataset_cfg: DictConfig, train_cfg: DictConfig
) -> tuple[dict[str, Any], dict[str, np.ndarray], int, int]:
    """Build the train/val/test datasets and the training-split standardization statistics.

    Two layouts are supported. When ``dataset.val_filepath`` and ``dataset.test_filepath``
    are set, each split is read from its own file and used as-is; otherwise
    ``dataset.filepath`` is read alone and split chronologically by ``train_ratio`` and
    ``val_ratio``.

    Args:
    ----
        dataset_cfg: The ``dataset`` section of the configuration.
        train_cfg: The ``train`` section of the configuration.

    Returns:
    -------
        The datasets keyed by split name, the standardization statistics, and the height
        and width of the rasters.

    """
    x_dim_col = dataset_cfg["x_dim_column"] if dataset_cfg["x_dim_column"] else "x"
    y_dim_col = dataset_cfg["y_dim_column"] if dataset_cfg["y_dim_column"] else "y"
    feature_columns = dataset_cfg["input_columns"]
    target_column = dataset_cfg["target_column"]
    map_height, map_width = ast.literal_eval(dataset_cfg["map_dimensions"])
    num_input_steps = train_cfg["input_steps"]

    def read_split(filepath: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return load_rasters(
            filepath=filepath,
            dataset_cfg=dataset_cfg,
            feature_columns=feature_columns,
            target_column=target_column,
            map_height=map_height,
            map_width=map_width,
            x_dim_col=x_dim_col,
            y_dim_col=y_dim_col,
        )

    val_filepath = dataset_cfg.get("val_filepath")
    test_filepath = dataset_cfg.get("test_filepath")
    history_dropout = {
        "history_dropout_prob": float(train_cfg.get("history_dropout_prob", 0.0)),
        "history_dropout_range": tuple(
            float(v) for v in train_cfg.get("history_dropout_range", (0.0, 1.0))
        ),
    }

    if val_filepath and test_filepath:
        # Chronological splits supplied as separate files, e.g. the per-scenario layout under
        # Outputs/<rcp>/<region>/ where data_train / data_val / data_test are contiguous.
        logger.info("Loading train, validation and test splits from separate files...")
        maps, targets_by_split, timestamps_by_split, masks = {}, {}, {}, {}
        for name, path in [
            ("train", dataset_cfg["filepath"]),
            ("val", val_filepath),
            ("test", test_filepath),
        ]:
            maps[name], targets_by_split[name], timestamps_by_split[name], masks[name] = read_split(
                path
            )

        for name, split_mask in masks.items():
            if not np.array_equal(split_mask, masks["train"]):
                raise ValueError(
                    f"The '{name}' split does not share the region mask of the training split. "
                    "All splits must be rasterized on the same grid."
                )
        mask = masks["train"]

        if dataset_cfg.get("carry_context_across_splits", True):
            # SWIDataset consumes its first `num_input_steps` frames as SWI history and never
            # predicts them, which would cost the 24-month val and test splits a third of their
            # months. The splits are contiguous in time, so prepend the tail of the preceding
            # one: those frames are only ever fed as context, never scored, so no target from
            # an earlier split leaks into the metrics of a later one.
            for name, previous in [("val", "train"), ("test", "val")]:
                for store in (maps, targets_by_split, timestamps_by_split):
                    store[name] = np.concatenate(
                        [store[previous][-num_input_steps:], store[name]], axis=0
                    )

        for name in ("train", "val", "test"):
            logger.info(
                f"Split '{name}': {len(maps[name])} months "
                f"-> {max(len(maps[name]) - num_input_steps, 0)} samples."
            )

        splits, statistics = build_datasets_from_split_rasters(
            split_maps=maps,
            split_targets=targets_by_split,
            split_timesteps=timestamps_by_split,
            mask=mask,
            num_input_steps=num_input_steps,
            **history_dropout,
        )
    else:
        # Single file split by ratio, the layout of the legacy "<region>/data_train.parquet".
        logger.info(f"Splitting '{dataset_cfg['filepath']}' by ratio...")
        feature_maps, targets, timestamps, mask = read_split(dataset_cfg["filepath"])
        splits, statistics = build_train_val_test_datasets(
            input_maps=feature_maps,
            target_maps=targets,
            timesteps=timestamps,
            mask=mask,
            train_ratio=train_cfg["train_ratio"],
            val_ratio=train_cfg["val_ratio"],
            num_input_steps=num_input_steps,
            **history_dropout,
        )

    return splits, statistics, map_height, map_width


@hydra.main(config_path="../config", config_name="train_swigan", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Train the model."""
    dataset_cfg = cfg["dataset"]
    train_cfg = cfg["train"]
    logging_cfg = cfg["logging"]

    np.random.seed(train_cfg["seed"])
    torch.manual_seed(train_cfg["seed"])

    splits, statistics, map_height, map_width = build_splits(dataset_cfg, train_cfg)
    feature_columns = dataset_cfg["input_columns"]

    # Create the dataloaders
    train = DataLoader(
        splits["train"],
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
    )
    val = DataLoader(
        splits["val"],
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
    )
    test = DataLoader(
        splits["test"],
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
    )

    # Instantiate the model
    logger.info("Instantiating the model...")
    if train_cfg["checkpoint_path"]:
        # Fine-tune from a checkpoint: the weights are the checkpoint's, the objective and the
        # schedule are THIS config's. Left to the saved hyper-parameters, Lightning would
        # rebuild the cosine schedule with the old run's max_epochs and lr, so a 50-epoch
        # fine-tune of a 250-epoch checkpoint would restart a 250-epoch schedule.
        # `noise_weight_lr_scale` belongs in this list for the same reason and one of its own:
        # it is read in `configure_optimizers`, which a loaded module runs from the SAVED
        # hparams, so without the override `train.noise_weight_lr_scale=100` on the command
        # line changes nothing at all -- the gains keep the base run's scale of 1.0, at which
        # they travel ~1e-3 over the whole schedule (see `init_noise_weights`) and the
        # injection is a fixed random hyper-parameter rather than something the fine-tune
        # can move. That is the one knob a spread-oriented objective needs to reach.
        overrides = {
            "loss_fn": train_cfg["loss_fn"],
            "lr": train_cfg["start_lr"],
            "min_lr": train_cfg["end_lr"],
            "max_epochs": train_cfg["num_epochs"],
            "ensemble_size": int(train_cfg.get("ensemble_size", 1)),
            "image_distance_weight": train_cfg["image_distance_weight"],
            "feature_matching_weight": train_cfg["feature_matching_weight"],
            "noise_weight_lr_scale": train_cfg["noise_weight_lr_scale"],
            "coherent_noise_init": train_cfg["coherent_noise_init"],
        }
        logger.info(
            f"Loading weights from '{train_cfg['checkpoint_path']}' with this config's "
            f"objective and schedule: {overrides}"
        )
        # A checkpoint written before the coherent channel existed holds none of its gains,
        # so turning the channel on means loading with strict=False. That is only safe if the
        # ONLY thing missing is those gains -- anything else absent is a real mismatch and has
        # to raise rather than be silently replaced by a fresh random init.
        saved_keys = set(
            torch.load(train_cfg["checkpoint_path"], map_location="cpu", weights_only=False)[
                "state_dict"
            ]
        )
        wants_coherent = train_cfg["coherent_noise_init"] is not None
        strict = not (wants_coherent and not any("coherent_noise_weights" in k for k in saved_keys))
        model = TTTSWIGAN.load_from_checkpoint(
            train_cfg["checkpoint_path"], strict=strict, **overrides
        )
        if not strict:
            missing = [k for k in model.state_dict() if k not in saved_keys]
            unexplained = [k for k in missing if "coherent_noise_weights" not in k]
            if unexplained:
                raise ValueError(
                    f"'{train_cfg['checkpoint_path']}' is missing {len(unexplained)} parameters "
                    f"that are not coherent-noise gains, e.g. {unexplained[:3]}. Loading it "
                    "would silently re-initialise trained weights."
                )
            logger.info(
                f"The checkpoint predates the coherent noise channel: {len(missing)} gains "
                f"initialised fresh as '{train_cfg['coherent_noise_init']}'. Every other "
                "parameter came from the checkpoint."
            )
        # A plain attribute, not part of the checkpoint: a loaded module has none until set.
        if getattr(model, "input_statistics", None) is None:
            model.input_statistics = statistics
    else:
        model = TTTSWIGAN(
            input_channels=len(feature_columns) + train_cfg["input_steps"],
            output_channels=1,
            input_map_dims=[map_height, map_width],
            encoder_channels=train_cfg["encoder_channels"],
            decoder_channels=train_cfg["decoder_channels"],
            timestamps_dim=train_cfg["timestamps_dim"],
            spatial_dropout=train_cfg["spatial_dropout"],
            apply_center_block=train_cfg["apply_center_block"],
            z_dim=train_cfg["noise_dim"],
            lr=train_cfg["start_lr"],
            weight_decay=train_cfg["weight_decay"],
            loss_fn=train_cfg["loss_fn"],
            optim=torch.optim.AdamW,
            normalization=train_cfg["normalization"],
            patch_critic_loss=train_cfg["patch_critic_loss"],
            patch_aggregation=train_cfg["patch_aggregation"],
            gradient_penalty_reduction=train_cfg["gradient_penalty_reduction"],
            num_critic_iterations_per_epoch=train_cfg["num_critic_iterations_per_epoch"],
            lambda_penalty=train_cfg["lambda_penalty"],
            lambda_penalty_patch=train_cfg["lambda_penalty_patch"],
            lambda_penalty_frame=train_cfg["lambda_penalty_frame"],
            image_distance_weight=train_cfg["image_distance_weight"],
            feature_matching_weight=train_cfg["feature_matching_weight"],
            max_epochs=train_cfg["num_epochs"],
            min_lr=train_cfg["end_lr"],
            noise_weight_init=train_cfg["noise_weight_init"],
            coherent_noise_init=train_cfg["coherent_noise_init"],
            noise_weight_lr_scale=train_cfg["noise_weight_lr_scale"],
            encoder_late_dropout=train_cfg["encoder_late_dropout"],
            critic_mask_channel=train_cfg.get("critic_mask_channel", True),
            critic_normalization=train_cfg.get("critic_normalization", "instancenorm"),
            generator_mask_channel=train_cfg.get("generator_mask_channel", True),
            generator_masked_norm=train_cfg.get("generator_masked_norm", True),
            critic_on_padded_canvas=train_cfg.get("critic_on_padded_canvas", False),
            diff_augment_on_region=train_cfg.get("diff_augment_on_region", False),
            ensemble_size=int(train_cfg.get("ensemble_size", 1)),
        )
        # Save the input_statistics
        model.input_statistics = statistics

    # Define callbacks
    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = (
        f"{logging_cfg['save_directory']}/{time_str}"
        if logging_cfg["save_directory"]
        else f"{cfg['experiment_name']}/{time_str}"
    )

    rollout_cfg = cfg.get("rollout")
    rollout_enabled = bool(rollout_cfg["enabled"]) if rollout_cfg is not None else False
    checkpoint_monitor = logging_cfg.get("checkpoint_monitor") or "val/generator_rmse_epoch"
    if checkpoint_monitor.startswith("val/rollout"):
        if not rollout_enabled:
            raise ValueError(
                f"'logging.checkpoint_monitor' is '{checkpoint_monitor}' but 'rollout.enabled' "
                "is false, so nothing ever logs that metric and ModelCheckpoint would have "
                "nothing to monitor. Enable the rollout or monitor 'val/generator_rmse_epoch'."
            )
        if rollout_cfg["every_n_epochs"] != 1:
            raise ValueError(
                f"'logging.checkpoint_monitor' is '{checkpoint_monitor}' but "
                f"'rollout.every_n_epochs' is {rollout_cfg['every_n_epochs']}, so the metric is "
                "absent on most epochs and ModelCheckpoint would fail as soon as it looked for "
                "it. Set 'rollout.every_n_epochs' to 1, or monitor a metric logged every epoch."
            )

    logger.info(
        f"Initializing checkpoint saver on '{checkpoint_monitor}'. "
        f"Saving checkpoints at {os.path.join(save_dir, 'checkpoints')}"
    )
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(save_dir, "checkpoints"),
            save_last=True,
            monitor=checkpoint_monitor,
            every_n_epochs=logging_cfg["save_checkpoint_every_n_epoch"],
            save_top_k=logging_cfg["save_top_k_checkpoints"],
        ),
        RandomTimeStepMapCallback(
            save_dir=os.path.join(save_dir, "figures"),
            input_statistics=statistics,
            train_dataset=splits["train"],
            val_dataset=splits["val"],
            print_every_n_epochs=logging_cfg["log_figures_every_n_epochs"],
            num_months_per_plot=logging_cfg["num_months_per_plots"],
        ),
    ]

    if rollout_enabled:
        # The free-running counterpart of val/generator_rmse. On the chronological layout the
        # val and test splits are 24 months each, so this is the paper's 24-step rollout error
        # measured every epoch instead of only in an offline rescoring pass.
        logger.info(
            f"Scoring a free-running rollout of the val split every "
            f"{rollout_cfg['every_n_epochs']} epoch(s): "
            f"{rollout_cfg['num_steps'] or len(splits['val'])} steps x "
            f"{rollout_cfg['num_members']} members."
        )
        callbacks.append(
            RolloutMetricsCallback(
                datasets={"val": splits["val"], "test": splits["test"]},
                statistics=statistics,
                num_members=rollout_cfg["num_members"],
                noise_std=rollout_cfg["noise_std"],
                seed=rollout_cfg["seed"],
                every_n_epochs=rollout_cfg["every_n_epochs"],
                num_steps=rollout_cfg["num_steps"],
            )
        )

    # Saving train dataset statistics
    np.savez(os.path.join(save_dir, "train_dataset_statistics"), **statistics)

    logger.info(f"Initializing Tensorboard logger at '{save_dir}/lightning_logs/version_0'")
    tb_logger = TensorBoardLogger(
        save_dir=save_dir,
    )
    trainer = Trainer(
        accelerator="auto",
        max_epochs=train_cfg["num_epochs"],
        log_every_n_steps=logging_cfg["log_every_n_steps"],
        callbacks=callbacks,
        logger=tb_logger,
    )

    resume_ckpt_path = train_cfg.get("resume_ckpt_path")
    logger.info(f"Starting training for {train_cfg['num_epochs']} epochs...")
    start_time = datetime.datetime.now()
    trainer.fit(model, train, val, ckpt_path=resume_ckpt_path)
    logger.info(f"Training finished in {datetime.datetime.now() - start_time}s..")

    logger.info("Testing model...")
    trainer.test(model, test)


if __name__ == "__main__":
    main()
