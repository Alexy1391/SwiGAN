"""Critic diagnostics: how well does each head separate observed maps from generated ones.

This is the measurement harness behind the round-by-round critic analysis. It scores a
trained checkpoint the same way for every run so the numbers are comparable across
rounds: same noise draw, same splits, same scoring, ``last.ckpt``.

Read AUC and Cohen's d rather than the raw separation of means -- the raw difference
scales with the critic's output magnitude, so it is not comparable between runs whose
scale differs (round 6's correction).

Every score is an average over ``--draws`` independent draws of the generator's noise,
reported with the standard deviation across those draws (round 7's correction). A single
draw is a high-variance estimate: the generator is stochastic at more than the z vector,
so the same checkpoint scored twice does not give the same AUC.

Usage:
    python -m utils.critic_diagnostics <run_dir> [<run_dir> ...] --out results.json
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from modules.masking import mask_pyramid
from swigan.engines.swigan_lit import TTTSWIGAN
from utils.preprocessing import (
    coerce_comma_decimal_columns,
    dataframe_to_rasters,
    fill_all_missing_pixels,
)
from utils.swi_dataset import build_train_val_test_datasets

logger = logging.getLogger(__name__)

# Fixed so that every run is scored against the same noise draws. Deliberately not the
# training seed: otherwise runs would be compared under different generator inputs.
#
# Draw k uses SCORING_NOISE_SEED + k for both the z vector and the global RNG. The global
# seed matters because the generator injects noise beyond z: every UNet decoder block
# calls torch.randn on the ambient RNG (unet_frame_encoder.py:85), and so does the
# temporal decoder when no z is supplied. Seeding only the z generator -- as this harness
# did through round 6 -- left those draws free, which is why re-scoring one checkpoint
# moved val/test AUC by up to 0.13.
SCORING_NOISE_SEED = 1234

# A single draw is not a usable estimate on the 23/24-sample splits. Five keeps the whole
# six-run sweep under ~20 minutes on one T4 while giving a standard error worth printing.
DEFAULT_DRAWS = 5

# Dataset and split settings live in the training config rather than the checkpoint, and
# have been identical for every run in the analysis.
TRAIN_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "train_swigan.yaml"


def auc(real: np.ndarray, fake: np.ndarray) -> float:
    """Probability a random real map outscores a random generated one.

    Computed from rank sums (Mann-Whitney U), so it is invariant to any monotone
    rescaling of the critic's output -- which is the property the raw difference of
    means lacks.
    """
    if len(real) == 0 or len(fake) == 0:
        return float("nan")
    scores = np.concatenate([real, fake])
    ranks = pd.Series(scores).rank().to_numpy()
    rank_sum_real = ranks[: len(real)].sum()
    u = rank_sum_real - len(real) * (len(real) + 1) / 2
    return float(u / (len(real) * len(fake)))


def cohens_d(real: np.ndarray, fake: np.ndarray) -> float:
    """Standardised mean difference, pooled standard deviation."""
    if len(real) < 2 or len(fake) < 2:
        return float("nan")
    pooled_var = ((len(real) - 1) * real.var(ddof=1) + (len(fake) - 1) * fake.var(ddof=1)) / (
        len(real) + len(fake) - 2
    )
    if pooled_var <= 0:
        return float("nan")
    return float((real.mean() - fake.mean()) / np.sqrt(pooled_var))


def _load_hparams(run_dir: Path) -> dict[str, Any]:
    hparams_path = run_dir / "lightning_logs" / "version_0" / "hparams.yaml"
    with hparams_path.open() as handle:
        return yaml.unsafe_load(handle)


@torch.no_grad()
def _score_split(
    model: TTTSWIGAN, loader: DataLoader, device: torch.device, seed: int
) -> dict[str, np.ndarray]:
    """Score every sample in a split, real and generated, through both heads."""
    out: dict[str, list[torch.Tensor]] = {
        "patch_real": [],
        "patch_fake": [],
        "frame_real": [],
        "frame_fake": [],
        "tiles_real": [],
        "tiles_fake": [],
    }
    # Both RNGs, because the generator draws from both -- see SCORING_NOISE_SEED.
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        batch_size = batch["target_maps"].shape[0]
        z = torch.randn(batch_size, model.hparams.z_dim, generator=generator).to(device)
        fake = model(
            input_maps=batch["input_maps"],
            input_timestamps=batch["timestamps"],
            mask=batch["mask"],
            noise_vector=z,
        )
        _, mask = model.to_critic_canvas(batch["target_maps"], batch["mask"])
        weight = mask_pyramid(mask, len(model.base_critic.blocks))[-1]
        # Every sample of a split shares the region mask, so the set of tiles with data
        # under them is one fixed subset of the grid; the padding tiles are dropped here
        # so the per-cell metrics are over the cells the loss actually consumes.
        valid_tiles = (weight[0, 0] > 0).flatten()
        for kind, maps in (("real", batch["target_maps"]), ("fake", fake)):
            maps, _ = model.to_critic_canvas(maps, batch["mask"])
            base, _ = model.base_critic(maps, mask)
            patch, _ = model.patch_critic(base, weight=weight)
            frame, _ = model.frame_critic(base, weight=weight)
            tiles = model.patch_critic.tile_scores(base, weight=weight)
            # Under "mean" aggregation the loss consumes the coverage-weighted grid
            # average; under "learned" the head already returns the scalar. Reduce to
            # one score per sample either way so the two are comparable.
            out[f"patch_{kind}"].append(model.patch_scalar(patch, weight).cpu())
            out[f"frame_{kind}"].append(frame.flatten(1).mean(dim=1).cpu())
            out[f"tiles_{kind}"].append(tiles.flatten(1)[:, valid_tiles].cpu())
    return {key: torch.cat(value).numpy() for key, value in out.items()}


def _cell_metrics(scores: dict[str, np.ndarray], train_signs: np.ndarray) -> dict[str, Any]:
    """Per-cell AUC, and what the grid holds once the cells' directions are aligned.

    The gap between the pooled read-out and the sign-corrected one is the discriminative
    information the unweighted average throws away. ``train_signs`` comes from the train
    split so val and test stay honest held-out numbers.
    """
    tiles_real, tiles_fake = scores["tiles_real"], scores["tiles_fake"]
    per_cell = [auc(tiles_real[:, i], tiles_fake[:, i]) for i in range(tiles_real.shape[1])]
    corrected_real = (tiles_real * train_signs).mean(axis=1)
    corrected_fake = (tiles_fake * train_signs).mean(axis=1)
    return {
        "per_cell_auc": [round(value, 4) for value in per_cell],
        "cells_above_chance": int(sum(value > 0.5 for value in per_cell)),
        "mean_abs_effect": round(float(np.mean([abs(v - 0.5) for v in per_cell])), 4),
        "sign_corrected_auc": round(auc(corrected_real, corrected_fake), 4),
    }


def diagnose(
    run_dir: Path, data_path: Path, device: torch.device, draws: int = DEFAULT_DRAWS
) -> dict[str, Any]:
    """Score one run's final checkpoint across all three splits, over ``draws`` draws."""
    hparams = _load_hparams(run_dir)
    logger.info("Scoring %s", run_dir)

    input_df = pd.read_parquet(data_path)
    input_df = fill_all_missing_pixels(input_df, x_dim_col="x", y_dim_col="y")
    input_df["mask"] = 1.0 * (input_df.loc[:, "scenario"] != 0)

    # The dataset/split settings are training-script config, not model hparams, so they
    # are not in the checkpoint. They have been identical across every round -- splits
    # are chronological and fixed at 709 / 23 / 24 -- so they are read from the config.
    with TRAIN_CONFIG_PATH.open() as handle:
        train_config = yaml.safe_load(handle)
    dataset_cfg, train_cfg = train_config["dataset"], train_config["train"]
    feature_columns = dataset_cfg["input_columns"]
    target_column = dataset_cfg["target_column"]
    useful = [*feature_columns, "year", "month", "x", "y", target_column, "mask"]
    input_df = input_df.drop(columns=[c for c in input_df.columns if c not in useful])
    input_df = coerce_comma_decimal_columns(input_df, [*feature_columns, target_column])

    map_height, map_width = ast.literal_eval(dataset_cfg["map_dimensions"])
    feature_maps, targets, timestamps, mask = dataframe_to_rasters(
        input_df, target_column, feature_columns, map_height, map_width
    )
    splits, _ = build_train_val_test_datasets(
        input_maps=feature_maps,
        target_maps=targets,
        timesteps=timestamps,
        mask=mask,
        train_ratio=train_cfg["train_ratio"],
        val_ratio=train_cfg["val_ratio"],
        num_input_steps=train_cfg["input_steps"],
    )

    model = TTTSWIGAN.load_from_checkpoint(
        run_dir / "checkpoints" / "last.ckpt", loss_fn="l1", map_location=device
    )
    model.eval().to(device)

    result: dict[str, Any] = {
        "run": run_dir.as_posix(),
        # The training seed is not saved into the checkpoint's hparams, so it is taken
        # from the experiment name, which is how the runs are distinguished on disk.
        "seed": 61 if "seed61" in run_dir.as_posix() else 60,
        "hparams": {
            key: hparams.get(key)
            for key in (
                "patch_critic_loss",
                "patch_aggregation",
                "gradient_penalty_reduction",
                "lambda_penalty",
                "lambda_penalty_patch",
                "lambda_penalty_frame",
                "weight_decay",
                "feature_matching_weight",
            )
        },
        "splits": {},
    }

    per_draw: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for draw in range(draws):
        seed = SCORING_NOISE_SEED + draw
        train_scores = _score_split(
            model, DataLoader(splits["train"], batch_size=64, shuffle=False), device, seed
        )
        train_cell_auc = np.array(
            [
                auc(train_scores["tiles_real"][:, i], train_scores["tiles_fake"][:, i])
                for i in range(train_scores["tiles_real"].shape[1])
            ]
        )
        train_signs = np.where(train_cell_auc >= 0.5, 1.0, -1.0)

        for name in ("train", "val", "test"):
            scores = (
                train_scores
                if name == "train"
                else _score_split(
                    model,
                    DataLoader(splits[name], batch_size=64, shuffle=False),
                    device,
                    seed,
                )
            )
            # Output scale is measured on the raw tile scores, not on the pooled
            # per-sample score: averaging 20 tiles shrinks the spread by roughly an order
            # of magnitude, and it is the tile scale that the gradient penalty constrains
            # and that the hinge margin of +/-1 is compared against.
            tiles = np.concatenate([scores["tiles_real"], scores["tiles_fake"]]).ravel()
            frame = np.concatenate([scores["frame_real"], scores["frame_fake"]])
            per_draw[name].append(
                {
                    "n": int(len(scores["patch_real"])),
                    "patch_auc": auc(scores["patch_real"], scores["patch_fake"]),
                    "frame_auc": auc(scores["frame_real"], scores["frame_fake"]),
                    "patch_d": cohens_d(scores["patch_real"], scores["patch_fake"]),
                    "frame_d": cohens_d(scores["frame_real"], scores["frame_fake"]),
                    "tile_score_sd": float(tiles.std()),
                    "tile_score_absmax": float(np.abs(tiles).max()),
                    "tile_pct_beyond_1": float((np.abs(tiles) > 1).mean() * 100),
                    "frame_score_sd": float(frame.std()),
                    "frame_score_absmax": float(np.abs(frame).max()),
                    **_cell_metrics(scores, train_signs),
                }
            )

    result["n_draws"] = draws
    for name, values in per_draw.items():
        result["splits"][name] = _aggregate_draws(values)
    return result


def _aggregate_draws(values: list[dict[str, Any]]) -> dict[str, Any]:
    """Average each metric over the noise draws and record the spread across them.

    The mean is the estimate to read; ``*_sd`` is the scoring noise on it, which on the
    23/24-sample splits is large enough to swamp several rounds' worth of reported
    differences. ``n`` and ``per_cell_auc`` are averaged element-wise; everything else is
    a scalar.
    """
    out: dict[str, Any] = {"n": values[0]["n"]}
    cells = np.array([v["per_cell_auc"] for v in values]).mean(axis=0)
    for key in values[0]:
        if key in ("n", "per_cell_auc"):
            continue
        series = np.array([v[key] for v in values], dtype=float)
        out[key] = round(float(series.mean()), 4)
        out[f"{key}_sd"] = round(float(series.std(ddof=1)) if len(series) > 1 else 0.0, 4)
    out["per_cell_auc"] = [round(float(v), 4) for v in cells]
    out["cells_above_chance"] = int((cells > 0.5).sum())
    out["mean_abs_effect"] = round(float(np.abs(cells - 0.5).mean()), 4)
    return out


def main() -> None:
    """Score one or more run directories and write the results as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--data", type=Path, default=Path("data_train.parquet"))
    parser.add_argument("--out", type=Path, default=Path("critic_diagnostics.json"))
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = [diagnose(run_dir, args.data, device, args.draws) for run_dir in args.run_dirs]
    with args.out.open("w") as handle:
        json.dump(results, handle, indent=2)
    logger.info("Wrote %s", args.out)

    for result in results:
        print(f"\n=== {result['run']} === ({result['n_draws']} noise draws)")
        print(f"    {result['hparams']}")
        for split, values in result["splits"].items():
            print(
                f"    {split:<5} n={values['n']:<4} "
                f"patch AUC {values['patch_auc']:.3f}±{values['patch_auc_sd']:.3f} "
                f"(d {values['patch_d']:+.2f})  "
                f"frame AUC {values['frame_auc']:.3f}±{values['frame_auc_sd']:.3f} "
                f"(d {values['frame_d']:+.2f})  "
                f"tile sd {values['tile_score_sd']:.3f}  "
                f"max|D| {values['tile_score_absmax']:.2f}  "
                f"sign-corr {values['sign_corrected_auc']:.3f}"
                f"±{values['sign_corrected_auc_sd']:.3f}"
            )


if __name__ == "__main__":
    main()
