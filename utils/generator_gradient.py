"""How much of the generator's update each loss term is responsible for.

``generator_step`` builds ``loss_generator`` from four terms:

    -mean(D_patch(fake))  -mean(D_frame(fake))  + pixel L1  + feature matching

This harness differentiates each of the four separately with respect to the generator's
parameters -- the ones ``configure_optimizers`` hands to the generator's optimiser -- and
reports each term's gradient norm, its share of the total, and its cosine similarity with
the direction the generator actually moves. The share is what answers "does the critic's
signal reach the generator at all", which AUC on its own cannot: a head can separate real
from fake perfectly and still contribute a rounding error to the update.

Measured under the training-time forward path (DiffAugment on, ``self.training`` True) at
a fixed seed, since that is the gradient the generator actually received. Averaged over
whole batches of the train split.

Usage:
    python -m utils.generator_gradient <run_dir> [<run_dir> ...] --out grads.json
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

from swigan.engines.swigan_lit import TTTSWIGAN
from utils.preprocessing import (
    coerce_comma_decimal_columns,
    dataframe_to_rasters,
    fill_all_missing_pixels,
)
from utils.swi_dataset import build_train_val_test_datasets

logger = logging.getLogger(__name__)

# Same convention as utils.critic_diagnostics: fixed, and not the training seed.
SCORING_NOISE_SEED = 1234
TRAIN_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "train_swigan.yaml"

TERMS = ("patch", "frame", "pixel", "feature")


def _flat_grad(loss: torch.Tensor, params: list[torch.nn.Parameter]) -> torch.Tensor:
    """Gradient of one loss term w.r.t. the generator, flattened into one vector."""
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return torch.cat(
        [
            (torch.zeros_like(p) if g is None else g).reshape(-1)
            for p, g in zip(params, grads, strict=True)
        ]
    )


def _term_losses(
    model: TTTSWIGAN, batch: dict[str, torch.Tensor], z: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Return the four additive terms of ``loss_generator``, kept separate.

    ``TTTSWIGAN.generator_step`` returns each term next to their sum, so this is the exact
    forward path the update is built from: DiffAugment with the mask carried through it,
    the coverage-weighted patch mean, the pixel and feature distances over the region's
    cells only.
    """
    outputs = model.generator_step(batch, z)
    return {
        "patch": outputs["patch_term"],
        "frame": outputs["frame_term"],
        "pixel": outputs["pixel_distance_loss"],
        "feature": outputs["feature_loss"],
    }


def decompose(
    run_dir: Path,
    data_path: Path,
    device: torch.device,
    batches: int,
    checkpoint: str = "last.ckpt",
) -> dict[str, Any]:
    """Decompose the generator's gradient over the first ``batches`` train batches.

    ``checkpoint`` names a file under ``<run_dir>/checkpoints``. Pointing it at an
    intermediate epoch is how a loss-weight change is checked mid-run, before the rest of
    the sweep is spent on it.
    """
    logger.info("Decomposing %s", run_dir)

    input_df = pd.read_parquet(data_path)
    input_df = fill_all_missing_pixels(input_df, x_dim_col="x", y_dim_col="y")
    input_df["mask"] = 1.0 * (input_df.loc[:, "scenario"] != 0)

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
    splits, statistics = build_train_val_test_datasets(
        input_maps=feature_maps,
        target_maps=targets,
        timesteps=timestamps,
        mask=mask,
        train_ratio=train_cfg["train_ratio"],
        val_ratio=train_cfg["val_ratio"],
        num_input_steps=train_cfg["input_steps"],
    )

    model = TTTSWIGAN.load_from_checkpoint(
        run_dir / "checkpoints" / checkpoint, loss_fn="l1", map_location=device
    )
    model.to(device)
    # ``generator_step`` also logs RMSE and SMAPE in physical units, which need the
    # training split's standardization statistics; the checkpoint does not carry them.
    model.input_statistics = statistics
    # The generator's gradient is what it received while training, so the training-time
    # forward path is the one to measure: DiffAugment active, stochastic depth active.
    model.train()
    params = [p for p in model.generator.parameters() if p.requires_grad]

    torch.manual_seed(SCORING_NOISE_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SCORING_NOISE_SEED)
    z_generator = torch.Generator(device="cpu").manual_seed(SCORING_NOISE_SEED)

    loader = DataLoader(splits["train"], batch_size=64, shuffle=False)
    norms: dict[str, list[float]] = {t: [] for t in TERMS}
    cosines: dict[str, list[float]] = {t: [] for t in TERMS}
    shares: dict[str, list[float]] = {t: [] for t in TERMS}

    for index, batch in enumerate(loader):
        if index >= batches:
            break
        batch = {key: value.to(device) for key, value in batch.items()}
        z = torch.randn(
            batch["target_maps"].shape[0], model.hparams.z_dim, generator=z_generator
        ).to(device)
        losses = _term_losses(model, batch, z)
        grads = {term: _flat_grad(losses[term], params) for term in TERMS}
        total = sum(grads.values())
        total_norm = total.norm().item()
        norm_sum = sum(g.norm().item() for g in grads.values())
        for term in TERMS:
            norm = grads[term].norm().item()
            norms[term].append(norm)
            shares[term].append(100.0 * norm / norm_sum if norm_sum else float("nan"))
            denominator = norm * total_norm
            cosines[term].append(
                torch.dot(grads[term], total).item() / denominator if denominator else float("nan")
            )
        model.zero_grad(set_to_none=True)

    return {
        "run": run_dir.as_posix(),
        "checkpoint": checkpoint,
        "seed": 61 if "seed61" in run_dir.as_posix() else 60,
        "batches": min(batches, len(loader)),
        "terms": {
            term: {
                "grad_norm": round(float(np.mean(norms[term])), 6),
                "share_pct": round(float(np.mean(shares[term])), 3),
                "share_pct_sd": round(float(np.std(shares[term], ddof=1)), 3),
                "cosine_with_total": round(float(np.mean(cosines[term])), 4),
            }
            for term in TERMS
        },
    }


def main() -> None:
    """Decompose one or more runs' generator gradients and write the results as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--data", type=Path, default=Path("data_train.parquet"))
    parser.add_argument("--out", type=Path, default=Path("generator_gradient.json"))
    parser.add_argument("--batches", type=int, default=6)
    parser.add_argument("--checkpoint", default="last.ckpt")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = [
        decompose(run_dir, args.data, device, args.batches, args.checkpoint)
        for run_dir in args.run_dirs
    ]
    with args.out.open("w") as handle:
        json.dump(results, handle, indent=2)
    logger.info("Wrote %s", args.out)

    for result in results:
        print(f"\n=== {result['run']} === ({result['checkpoint']}, {result['batches']} batches)")
        for term, values in result["terms"].items():
            print(
                f"    {term:<8} share {values['share_pct']:>6.2f}% "
                f"± {values['share_pct_sd']:<5.2f}  "
                f"‖g‖ {values['grad_norm']:>10.4f}  "
                f"cos(total) {values['cosine_with_total']:+.3f}"
            )


if __name__ == "__main__":
    main()
