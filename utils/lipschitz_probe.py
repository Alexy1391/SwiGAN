"""Report which side of 1 the critic's input-gradient norm falls on.

The gradient penalty logs ``(||grad_x D|| - 1)^2``, which is symmetric: it cannot say
whether the critic is too sensitive to its input or not sensitive enough. That matters,
because spectral normalisation caps sensitivity from above while the penalty pushes it
towards 1 from whichever side it is on. If the critic sits *below* 1, the two mechanisms
are pulling against each other and raising ``lambda_penalty`` fights the architecture
rather than the critic.

This probe reproduces the interpolation ``TTTSWIGAN.gradient_penalty`` uses and reports
the distribution of ||grad_x D|| per head.

Usage:
    python -m utils.lipschitz_probe <run_dir> [<run_dir> ...] --out lipschitz.json
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
from torch import autograd
from torch.utils.data import DataLoader

from modules.masking import diff_augment_with_mask, mask_pyramid
from swigan.engines.swigan_lit import DIFF_AUGMENT_POLICY, TTTSWIGAN
from utils.preprocessing import (
    coerce_comma_decimal_columns,
    dataframe_to_rasters,
    fill_all_missing_pixels,
)
from utils.swi_dataset import build_train_val_test_datasets

logger = logging.getLogger(__name__)

SCORING_NOISE_SEED = 1234
TRAIN_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "train_swigan.yaml"


def _grad_norms(
    model: TTTSWIGAN,
    real: torch.Tensor,
    fake: torch.Tensor,
    mask: torch.Tensor,
    head: str,
    augment: bool,
) -> np.ndarray:
    """||grad_x D|| at points interpolated between real and generated maps.

    ``augment`` reproduces the training path: the critic step applies DiffAugment to both
    tensors before the penalty is computed, carrying the region mask through it, so the
    constraint the penalty actually enforced is the augmented one under the union of the
    two augmented masks. Off, this measures the trained critic's sensitivity on clean
    inputs instead. Either way the gradient is projected onto the mask before the norm,
    exactly as ``TTTSWIGAN.gradient_penalty`` does.
    """
    batch_size = real.shape[0]
    real, mask_on_canvas = model.to_critic_canvas(real, mask)
    fake, _ = model.to_critic_canvas(fake, mask)
    mask = mask_on_canvas
    if augment:
        real, mask_real = diff_augment_with_mask(real, mask, DIFF_AUGMENT_POLICY)
        fake, mask_fake = diff_augment_with_mask(fake, mask, DIFF_AUGMENT_POLICY)
        mask = torch.maximum(mask_real, mask_fake)
    else:
        mask = mask.to(real.dtype).expand(batch_size, -1, -1, -1)
    alpha = torch.rand(batch_size, 1, 1, 1, device=real.device)
    interpolated = (alpha * real + (1 - alpha) * fake).clone().detach().requires_grad_(True)

    levels = mask_pyramid(mask, len(model.base_critic.blocks))
    base_output, _ = model.base_critic(interpolated, mask)
    critic = model.patch_critic if head == "patch" else model.frame_critic
    d_interpolated, _ = critic(base_output, weight=levels[-1])

    # Matches gradient_penalty_reduction="mean", which is what every run since round 5
    # uses and the only setting under which lambda means the same thing for both heads:
    # the coverage-weighted tile mean for the patch head, the scalar for the frame head.
    scalar_output = model.patch_scalar(d_interpolated, levels[-1]).sum()
    gradients = autograd.grad(
        outputs=scalar_output, inputs=interpolated, create_graph=False, only_inputs=True
    )[0]
    gradients = (gradients * mask).view(batch_size, -1)
    return gradients.norm(2, dim=-1).detach().cpu().numpy()


def probe(
    run_dir: Path, data_path: Path, device: torch.device, batches: int, augment: bool
) -> dict[str, Any]:
    """Measure the input-gradient norm distribution for both heads of one run."""
    logger.info("Probing %s", run_dir)

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

    torch.manual_seed(SCORING_NOISE_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SCORING_NOISE_SEED)
    z_generator = torch.Generator(device="cpu").manual_seed(SCORING_NOISE_SEED)

    loader = DataLoader(splits["train"], batch_size=64, shuffle=False)
    collected: dict[str, list[np.ndarray]] = {"patch": [], "frame": []}
    for index, batch in enumerate(loader):
        if index >= batches:
            break
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.no_grad():
            z = torch.randn(
                batch["target_maps"].shape[0], model.hparams.z_dim, generator=z_generator
            ).to(device)
            fake = model(
                input_maps=batch["input_maps"],
                input_timestamps=batch["timestamps"],
                mask=batch["mask"],
                noise_vector=z,
            )
        for head in ("patch", "frame"):
            collected[head].append(
                _grad_norms(model, batch["target_maps"], fake, batch["mask"], head, augment)
            )

    result: dict[str, Any] = {
        "run": run_dir.as_posix(),
        "seed": 61 if "seed61" in run_dir.as_posix() else 60,
        "augmented": augment,
        "heads": {},
    }
    for head, chunks in collected.items():
        norms = np.concatenate(chunks)
        result["heads"][head] = {
            "n": int(norms.size),
            "mean": round(float(norms.mean()), 4),
            "median": round(float(np.median(norms)), 4),
            "p05": round(float(np.percentile(norms, 5)), 4),
            "p95": round(float(np.percentile(norms, 95)), 4),
            "pct_below_1": round(float((norms < 1).mean() * 100), 2),
            "penalty": round(float(((norms - 1) ** 2).mean()), 4),
        }
    return result


def main() -> None:
    """Probe one or more runs and write the results as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--data", type=Path, default=Path("data_train.parquet"))
    parser.add_argument("--out", type=Path, default=Path("lipschitz_probe.json"))
    parser.add_argument("--batches", type=int, default=11)
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Apply DiffAugment first, reproducing the training-time penalty.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = [
        probe(run_dir, args.data, device, args.batches, args.augment) for run_dir in args.run_dirs
    ]
    with args.out.open("w") as handle:
        json.dump(results, handle, indent=2)
    logger.info("Wrote %s", args.out)

    for result in results:
        mode = "training path (DiffAugment)" if result["augmented"] else "clean inputs"
        print(f"\n=== {result['run']} === ({mode})")
        for head, v in result["heads"].items():
            print(
                f"    {head:<6} ||grad_x D||  mean {v['mean']:>7.3f}  median {v['median']:>7.3f}  "
                f"[p05 {v['p05']:>6.3f}, p95 {v['p95']:>6.3f}]  "
                f"below 1: {v['pct_below_1']:>6.2f}%  penalty {v['penalty']:.4f}"
            )


if __name__ == "__main__":
    main()
