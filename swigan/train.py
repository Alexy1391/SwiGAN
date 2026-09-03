"""Script to launch the training."""

import ast
import datetime
import logging
import os
from pathlib import Path

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
from utils.callbacks import RandomTimeStepMapCallback
from utils.preprocessing import (
    coerce_comma_decimal_columns,
    dataframe_to_rasters,
    fill_all_missing_pixels,
)
from utils.swi_dataset import build_train_val_test_datasets

logger = logging.getLogger(__file__)


@hydra.main(config_path="../config", config_name="train_swigan", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Train the model."""
    dataset_cfg = cfg["dataset"]
    train_cfg = cfg["train"]
    logging_cfg = cfg["logging"]

    np.random.seed(train_cfg["seed"])
    torch.manual_seed(train_cfg["seed"])

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
    columns_to_drop = [col for col in input_df.columns if col not in useful_columns]
    input_df = input_df.drop(columns=columns_to_drop)

    input_df = coerce_comma_decimal_columns(input_df, [*feature_columns, target_column])

    # Extract features and targets
    map_height, map_width = ast.literal_eval(dataset_cfg["map_dimensions"])
    feature_maps, targets, timestamps, mask = dataframe_to_rasters(
        input_df, dataset_cfg["target_column"], feature_columns, map_height, map_width
    )

    splits, statistics = build_train_val_test_datasets(
        input_maps=feature_maps,
        target_maps=targets,
        timesteps=timestamps,
        mask=mask,
        train_ratio=train_cfg["train_ratio"],
        val_ratio=train_cfg["val_ratio"],
        num_input_steps=train_cfg["input_steps"],
    )

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
        # Resume from checkpoint
        model = TTTSWIGAN.load_from_checkpoint(
            train_cfg["checkpoint_path"], loss_fn=train_cfg["loss_fn"]
        )
        if model.input_statistics is None:
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
            noise_weight_lr_scale=train_cfg["noise_weight_lr_scale"],
            encoder_late_dropout=train_cfg["encoder_late_dropout"],
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

    logger.info(
        f"Initializing checkpoint saver. "
        f"Saving checkpoints at {os.path.join(save_dir, 'checkpoints')}"
    )
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(save_dir, "checkpoints"),
            save_last=True,
            monitor="val/generator_rmse_epoch",
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
