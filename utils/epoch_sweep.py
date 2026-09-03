"""Score CRPS, spread, RMSE and the z-contribution along the EPOCH axis of a retained run.

WHY THIS EXISTS. Every score the project has on the 1500-epoch arms is an ENDPOINT: last.ckpt,
one number. The spread-vs-epoch and CRPS-vs-epoch curves have never been measured, so "the
dispersion decays over training" and "RMSE stops improving after ~800" are inferences from two
runs of different length, not from a curve. Both arms retained ~20 intermediate checkpoints, so
the curve costs nothing but inference time.

WHAT IS COMPUTED, per checkpoint, on the paper window (Jan 2023 - Dec 2024, 24 free-running
steps, 20 members):

  ships arm   noise_std=1, injection independent -- exactly what utils/stochasticity_probe.py
              calls "ships today" and what utils/ensemble_scores.py scores. Gives CRPS (fair),
              CRPS_det, crps_skill, spread, spread/signal, spread-skill ratio, rank-histogram
              reliability, envelope coverage, and the paper's per-pixel RMSE/SMAPE.
  z arm       noise_std=1, injection SHARED across members -- removes the per-block injection
              from the across-member variance, leaving only the bottleneck vector z. Its spread
              is the z contribution; z_share = spread_z / spread_ships.

  noise_weights_rms   rms of the learned injection gains at that checkpoint, so the lever's own
                      trajectory is on the same epoch axis as the spread it is supposed to drive.

Two rollouts per checkpoint rather than the probe's eight: the six arms this drops (z at std
5/20, injection-only, both at std 5/20, and the determinism control) are amplitude sweeps, and
the amplitude question was settled by round 12. The epoch axis is the open one.

Inference only. Same overrides as swigan/inference.py, plus:

    +sweep.checkpoint_dir=<run>/checkpoints   directory of epoch=*.ckpt files
    +sweep.tag=<name>                         output goes to <saving_path>_<tag>_epochs.npz
"""

import logging
import re
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, open_dict

from utils.ensemble_scores import score
from utils.stochasticity_probe import prepare, run_arm, summarise

logger = logging.getLogger(__file__)

EPOCH_RE = re.compile(r"epoch=(\d+)")


def checkpoints(directory: str) -> list[tuple[int, Path]]:
    """Every epoch=*.ckpt under ``directory``, ordered by epoch. ``last.ckpt`` is skipped: it is
    a duplicate of the highest retained epoch, and scoring it twice would put two points on the
    same abscissa."""
    found = []
    for path in sorted(Path(directory).glob("epoch=*.ckpt")):
        match = EPOCH_RE.search(path.name)
        if match:
            found.append((int(match.group(1)), path))
    if not found:
        raise SystemExit(f"no epoch=*.ckpt under {directory}")
    return sorted(found)


def noise_weights_rms(model: torch.nn.Module) -> float:
    """rms over every learned injection gain -- the lever the spread is supposed to ride on."""
    values = torch.cat(
        [p.detach().flatten() for name, p in model.named_parameters() if "noise_weights" in name]
    )
    return float(values.pow(2).mean().sqrt())


@hydra.main(config_path="../config", config_name="inference_swigan", version_base="1.3")
def main(cfg: DictConfig) -> None:
    model_cfg = cfg["model"]
    tag = cfg["sweep"]["tag"]
    ckpts = checkpoints(cfg["sweep"]["checkpoint_dir"])
    logger.info(f"{tag}: {len(ckpts)} checkpoints, epochs {ckpts[0][0]}..{ckpts[-1][0]}")

    records = []
    for epoch, path in ckpts:
        with open_dict(cfg):
            cfg["model"]["checkpoint_path"] = str(path)
        prepared = prepare(cfg)
        shared = prepared["rollout"]
        y_true, bool_mask = prepared["y_true"], prepared["bool_mask"]
        target_std = prepared["target_std"]
        truth = y_true[:, bool_mask]

        ships = run_arm(noise_std=1.0, injection="independent", seed=0, **shared)
        z_only = run_arm(noise_std=1.0, injection="shared", seed=0, **shared)

        stats = summarise(ships, y_true, bool_mask, target_std)
        z_stats = summarise(z_only, y_true, bool_mask, target_std)
        scores = score(ships[:, :, bool_mask], truth, target_std)

        row = {
            "epoch": epoch,
            "crps": scores["crps"],
            "crps_det": scores["crps_det"],
            "crps_skill": scores["crps_skill"],
            "spread_skill": scores["spread_skill"],
            "reliability": scores["reliability"],
            "extreme_frac": scores["extreme_frac"],
            "spread": stats["spread"],
            "spread_ratio": stats["spread_ratio"],
            "signal": stats["signal"],
            "eff_rank": stats["eff_rank"],
            "coverage": stats["coverage"],
            "rmse_max": stats["rmse_max"],
            "rmse_p80": stats["rmse_p80"],
            "rmse_mean": stats["rmse_mean"],
            "smape_max": stats["smape_max"],
            "spread_z": z_stats["spread"],
            "spread_ratio_z": z_stats["spread_ratio"],
            "noise_weights_rms": noise_weights_rms(prepared["model"]),
        }
        row["z_share"] = row["spread_z"] / row["spread"] if row["spread"] > 0 else float("nan")
        records.append(row)
        logger.info(
            f"epoch {epoch:5d}  CRPS {row['crps']:.5f}  spread {row['spread']:.5f}  "
            f"rmse_max {row['rmse_max']:.4f}  z {row['spread_z']:.5f} ({row['z_share']:.1%})"
        )
        del prepared, shared, ships, z_only
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    keys = [k for k in records[0] if k != "epoch"]
    print("\n" + "=" * 112)
    print(f"EPOCH AXIS · {tag}".center(112))
    print("20 members · paper window · CRPS and spread in standardised units".center(112))
    print("=" * 112)
    print(f"{'epoch':>6s} {'CRPS':>9s} {'skill':>8s} {'spread':>9s} {'sprd/sig':>9s} "
          f"{'sprd/skl':>9s} {'rmse max':>9s} {'rmse mean':>10s} {'spread z':>9s} "
          f"{'z share':>8s} {'rms(w)':>9s}")
    print("-" * 112)
    for r in records:
        print(f"{r['epoch']:6d} {r['crps']:9.5f} {r['crps_skill']:+8.3f} {r['spread']:9.5f} "
              f"{r['spread_ratio']:9.4f} {r['spread_skill']:9.3f} {r['rmse_max']:9.4f} "
              f"{r['rmse_mean']:10.4f} {r['spread_z']:9.5f} {r['z_share']:7.1%} "
              f"{r['noise_weights_rms']:9.6f}")
    print("-" * 112)
    best = min(records, key=lambda r: r["crps"])
    last = records[-1]
    print(f"CRPS minimum at epoch {best['epoch']} ({best['crps']:.5f}); "
          f"final epoch {last['epoch']} is {(last['crps'] / best['crps'] - 1) * 100:+.2f}%")
    best_rmse = min(records, key=lambda r: r["rmse_max"])
    print(f"rmse_max minimum at epoch {best_rmse['epoch']} ({best_rmse['rmse_max']:.4f}); "
          f"final epoch {last['epoch']} is {(last['rmse_max'] / best_rmse['rmse_max'] - 1) * 100:+.2f}%")
    print("=" * 112 + "\n")

    out = Path(model_cfg["saving_path"]).with_suffix("")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        f"{out}_{tag}_epochs.npz",
        epoch=np.array([r["epoch"] for r in records]),
        **{k: np.array([r[k] for r in records], dtype=float) for k in keys},
    )
    logger.info(f"Saved to '{out}_{tag}_epochs.npz'.")


if __name__ == "__main__":
    main()
