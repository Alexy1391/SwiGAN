"""Module containing custom training callbacks."""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning import Callback, Trainer
from matplotlib import colors
from pytorch_lightning import LightningModule
from torch.utils.data import Dataset

from utils.rollout import score_rollout

logger = logging.getLogger(__file__)

MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


class RolloutMetricsCallback(Callback):
    """Log the FREE-RUNNING rollout error of the validation and test splits during training.

    ``val/generator_rmse`` is teacher-forced -- one month ahead of eight observed SWI frames --
    so it never sees error compound and cannot separate a model that has learned the dynamics
    from one that has learned to copy its last input. This callback rolls the generator over the
    whole split on its own predictions and logs the metrics ``swigan/evaluate.py`` computes, so
    the number that matters is on the training curve instead of only in an offline rescoring
    pass. With the chronological layout of ``config/train_swigan.yaml`` the val and test splits
    are 24 months each, so ``val/rollout_rmse`` is a 24-step rollout RMSE.

    ``val/rollout_rmse`` is what ``logging.checkpoint_monitor`` should point at; the rest are
    diagnostics. ``rollout_rmse_max`` is the paper's headline reduction, ``rollout_drift`` is
    the ratio of the last free-running month's spatial RMSE to the first month's.

    The rollout runs on a private RNG stream and restores the global one, so it does not shift
    the draws the training steps consume: turning the metric on does not perturb the run it is
    measuring.
    """

    def __init__(
        self,
        datasets: dict[str, Dataset],
        statistics: dict[str, np.ndarray],
        num_members: int = 8,
        noise_std: float = 1.0,
        seed: int = 0,
        every_n_epochs: int = 1,
        num_steps: int | None = None,
    ) -> None:
        """Initialize the callback.

        Args:
        ----
            datasets: The splits to roll over, keyed by name. Only "val" and "test" are used,
                scored at the end of the validation and test epochs respectively.
            statistics: The training-split standardization statistics, used to bring the
                trajectories back to physical SWI units before scoring.
            num_members: Ensemble members per rollout, run as a single batch. They share the
                covariates and the seed history and differ only in their noise draws.
            noise_std: Scale of the bottleneck noise vector. 1.0 is what inference ships with.
            seed: Seed of the rollout's private RNG stream. Held fixed across epochs on
                purpose: the same noise every epoch means a move in the curve is a move in the
                model, not a different draw.
            every_n_epochs: Score every n-th validation epoch. Leave at 1 when
                ``logging.checkpoint_monitor`` is one of these metrics, otherwise the checkpoint
                callback has nothing to read on the epochs that were skipped.
            num_steps: Rollout horizon. ``None`` uses the whole split.

        """
        self.datasets = datasets
        self.statistics = statistics
        self.num_members = num_members
        self.noise_std = noise_std
        self.seed = seed
        self.every_n_epochs = max(1, every_n_epochs)
        self.num_steps = num_steps

    def _score(self, pl_module: LightningModule, split: str) -> None:
        """Roll over one split and log its metrics under the ``<split>/`` prefix."""
        dataset = self.datasets.get(split)
        if dataset is None:
            return

        metrics = score_rollout(
            model=pl_module,
            dataset=dataset,
            statistics=self.statistics,
            num_members=self.num_members,
            noise_std=self.noise_std,
            num_steps=self.num_steps,
            seed=self.seed,
            device=pl_module.device,
        )
        scalars = {
            f"{split}/{name}": value for name, value in metrics.items() if isinstance(value, float)
        }
        pl_module.log_dict(scalars, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        ensemble = ""
        if "rollout_crps" in metrics:
            ensemble = (
                f", crps {metrics['rollout_crps']:.4f}, spread-skill "
                f"{metrics['rollout_spread_skill']:.3f}"
            )
            if "rollout_spread_shape_min" in metrics:
                # The per-month test of utils.spread_shape: every calendar month's spread /
                # required inside the band, not just the annual ratio near 1.
                ensemble += (
                    f", monthly spread ratio {metrics['rollout_spread_shape_min']:.2f}-"
                    f"{metrics['rollout_spread_shape_max']:.2f} "
                    f"({100 * metrics['rollout_spread_shape_in_band']:.0f}% of months in band)"
                )
        logger.info(
            f"epoch {pl_module.current_epoch}: {split} rollout over "
            f"{len(metrics['rmse_by_step'])} steps x {self.num_members} members -- "
            f"rmse {metrics['rollout_rmse']:.4f} (max {metrics['rollout_rmse_max']:.4f}, "
            f"p80 {metrics['rollout_rmse_p80']:.4f}), drift {metrics['rollout_drift']:.2f}x"
            f"{ensemble}"
        )

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Score the validation split, early enough for ModelCheckpoint to monitor it."""
        if trainer.sanity_checking:
            return
        if (pl_module.current_epoch + 1) % self.every_n_epochs != 0:
            return
        self._score(pl_module, "val")

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Score the test split once, at the end of ``trainer.test``."""
        self._score(pl_module, "test")


class RandomTimeStepMapCallback(Callback):
    """Callback class for plotting randomly generated maps during training.

    This class will randomly select k maps from the train dataset and k
    maps from the validation dataset and plot the model predictions.
    """

    def __init__(
        self,
        save_dir: str,
        input_statistics: dict[str, np.ndarray],
        train_dataset: Dataset,
        val_dataset: Dataset,
        print_every_n_epochs: int = 20,
        num_months_per_plot: int = 6,
    ) -> None:
        """Initialize the callback.

        Args:
        ----
            save_dir: Folder where the figures will be saved.
            input_statistics: Dictionary containing the train targets mean and std
                for inference.
            train_dataset: The training dataset.
            val_dataset: The validation dataset.
            print_every_n_epochs: Plotting frequency.
            num_months_per_plot: Number of months to show in the same plot.

        """
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.input_statistics = input_statistics
        self.print_frequency = print_every_n_epochs
        self.num_months_per_plot = num_months_per_plot

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Print the maps at the end of the validation epoch."""
        plt.close("all")
        if (pl_module.current_epoch + 1) % self.print_frequency == 0:
            targets_mean_value, targets_std_value = (
                self.input_statistics["targets_mean"],
                self.input_statistics["targets_std"],
            )
            with torch.no_grad():
                for [name, dataset_p] in [
                    ("Train", self.train_dataset),
                    ("Validation", self.val_dataset),
                ]:
                    fig, axes = plt.subplots(
                        2, self.num_months_per_plot, figsize=(5 * self.num_months_per_plot, 6)
                    )
                    plt.subplots_adjust(wspace=0.5)
                    for k, idx in enumerate(np.random.choice(len(dataset_p), 6, replace=False)):
                        inputs = dataset_p[idx]

                        targets = inputs["target_maps"].squeeze().detach().cpu().numpy()
                        res = pl_module(
                            inputs["input_maps"].unsqueeze(0).to(pl_module.device),
                            inputs["timestamps"].unsqueeze(0).to(pl_module.device),
                            inputs["mask"].unsqueeze(0).to(pl_module.device),
                        )
                        timestamps = inputs["timestamps"].detach().cpu().numpy()
                        mask = inputs["mask"].squeeze().detach().cpu().numpy()

                        month_name = MONTHS[int(timestamps)]
                        targets_k = (
                            targets * targets_std_value.squeeze() + targets_mean_value.squeeze()
                        ) * mask
                        res_k = (
                            res.squeeze().detach().cpu().numpy() * targets_std_value.squeeze()
                            + targets_mean_value.squeeze()
                        ) * mask
                        merged = np.stack([targets_k, res_k], axis=0)
                        norm = colors.Normalize(vmin=np.min(merged), vmax=np.max(merged))
                        axes[0, k].set_title(f"GT Month {month_name}")
                        p = axes[0, k].imshow(
                            np.where(mask, targets_k, np.nan), cmap="RdYlGn", norm=norm
                        )
                        axes[1, k].set_title(f"Prediction Month {month_name}")
                        p = axes[1, k].imshow(
                            np.where(mask, res_k, np.nan), cmap="RdYlGn", norm=norm
                        )
                        plt.colorbar(p, ax=axes[:, k], shrink=0.4)
                    fig.suptitle(f"{name} maps epoch {pl_module.current_epoch}")
                    plt.savefig(
                        os.path.join(
                            self.save_dir,
                            f"{name}_predictions_epoch{pl_module.current_epoch:04d}.png",
                        ),
                        dpi=200,
                    )
