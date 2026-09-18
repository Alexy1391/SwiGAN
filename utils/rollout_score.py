"""Score a finished checkpoint's free-running rollout on the val and test splits.

The same measurement ``utils.callbacks.RolloutMetricsCallback`` puts on the training curve, run
after the fact on a checkpoint that trained before the callback existed -- which is every
checkpoint on disk today, including the two 1500-epoch arms and the current 1000-epoch run. It
reads the training config, so the splits, the standardization statistics and the horizon are
exactly the ones training would have used.

This is not a duplicate of ``run_paper1500_metrics.sh``. That path scores
``Grand_Est_2022_2025.parquet`` through ``swigan/inference.py``, whose rollout seeds its SWI
history from RAW target maps while feeding back STANDARDIZED predictions -- the first
``input_steps`` months of every trajectory it produces are generated from mixed units. The
rollout here builds its seed from the dataset, which is already standardized, so the two will
not agree exactly on the same window; this one is the protocol training should be selecting on.

Inference only. Run with the training config's overrides plus the checkpoint:

    .venv/bin/python utils/rollout_score.py \
        train.checkpoint_path=<run>/checkpoints/last.ckpt

``rollout.num_members``, ``rollout.noise_std``, ``rollout.seed`` and ``rollout.num_steps`` are
read from the same ``rollout`` section training uses, so a number printed here is the number
the callback would have logged.
"""

import logging

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from swigan.engines.swigan_lit import TTTSWIGAN
from swigan.train import build_splits
from utils.rollout import free_running_rollout, summarise_rollout

logger = logging.getLogger(__file__)

SPLITS = ("val", "test")


@hydra.main(config_path="../config", config_name="train_swigan", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Roll one checkpoint over each split and print the metrics."""
    train_cfg = cfg["train"]
    rollout_cfg = cfg["rollout"]

    checkpoint_path = train_cfg["checkpoint_path"]
    if not checkpoint_path:
        raise SystemExit(
            "Set 'train.checkpoint_path' to the checkpoint to score, e.g. "
            "train.checkpoint_path=<run>/checkpoints/last.ckpt"
        )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    splits, statistics, _, _ = build_splits(cfg["dataset"], train_cfg)

    logger.info(f"Loading '{checkpoint_path}'...")
    model = TTTSWIGAN.load_from_checkpoint(checkpoint_path, loss_fn=train_cfg["loss_fn"])
    model.input_statistics = statistics
    model.to(device)

    results = {}
    for split in SPLITS:
        trajectories, y_true, mask = free_running_rollout(
            model=model,
            dataset=splits[split],
            statistics=statistics,
            num_members=rollout_cfg["num_members"],
            noise_std=rollout_cfg["noise_std"],
            num_steps=rollout_cfg["num_steps"],
            seed=rollout_cfg["seed"],
            device=device,
        )
        results[split] = summarise_rollout(trajectories, y_true, mask)
        logger.info(f"{split}: {len(results[split]['rmse_by_step'])} steps scored.")

    horizon = len(results["val"]["rmse_by_step"])
    print("\n" + "=" * 88)
    print("FREE-RUNNING ROLLOUT".center(88))
    print(f"{checkpoint_path}".center(88))
    print(
        f"{horizon} steps x {rollout_cfg['num_members']} members x noise_std "
        f"{rollout_cfg['noise_std']} - eval mode".center(88)
    )
    print("=" * 88)
    print(
        f"{'split':>6s} {'rmse':>9s} {'p80':>9s} {'max':>9s} {'spatial':>9s} {'smape':>9s} "
        f"{'step 1':>9s} {'step -1':>9s} {'drift':>8s}"
    )
    print("-" * 88)
    for split in SPLITS:
        r = results[split]
        print(
            f"{split:>6s} {r['rollout_rmse']:9.4f} {r['rollout_rmse_p80']:9.4f} "
            f"{r['rollout_rmse_max']:9.4f} {r['rollout_rmse_spatial']:9.4f} "
            f"{r['rollout_smape']:9.2f} {r['rollout_rmse_first_step']:9.4f} "
            f"{r['rollout_rmse_last_step']:9.4f} {r['rollout_drift']:7.2f}x"
        )
    print("-" * 88)
    for split in SPLITS:
        curve = " ".join(f"{v:.3f}" for v in results[split]["rmse_by_step"])
        print(f"{split} spatial RMSE by step: {curve}")
    print("\nrmse/p80/max = per-pixel RMSE over the horizon, averaged over members, then")
    print("               reduced over pixels. 'max' is the paper's headline reduction (0.21).")
    print("spatial      = mean over months of the spatial RMSE -- the free-running twin of")
    print("               val/generator_rmse, so the two are directly comparable.")
    print("drift        = last step's spatial RMSE / first step's. 1.0 = no compounding.")
    print("=" * 88 + "\n")

    output_path = f"{checkpoint_path}.rollout.npz"
    np.savez(
        output_path,
        **{
            f"{split}_{name}": np.asarray(value)
            for split, metrics in results.items()
            for name, value in metrics.items()
        },
    )
    logger.info(f"Saved to '{output_path}'.")


if __name__ == "__main__":
    main()
