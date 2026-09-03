"""First-order field statistics of generated maps against the observed ones.

The fresh-critic test (``utils.fresh_critic``) reports how quickly a generator's samples
can be told apart, not what gives them away. These are the cheapest candidates for the
tell, computed over the region of interest only:

* roughness -- the mean absolute first difference between neighbouring pixels, which is
  what an over-smoothed reconstruction or an over-textured adversarial sample gets wrong;
* spatial standard deviation within each map;
* the map mean, which catches a constant bias.

None of them is what the critic reads -- it reads a convolutional stack -- so agreement
here is necessary rather than sufficient. Disagreement between this ordering and the
fresh-critic ordering is itself the finding: it says the tell is structural rather than
first-order.

Usage:
    python -m utils.field_stats <run_dir> [<run_dir> ...] --out field_stats.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from swigan.engines.swigan_lit import TTTSWIGAN
from utils.fresh_critic import _generate, _materialise, build_splits

logger = logging.getLogger(__name__)

SCORING_NOISE_SEED = 1234
DEFAULT_DRAWS = 3


def field_statistics(maps: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    """Roughness, spatial standard deviation and mean, averaged over maps.

    Args:
    ----
        maps: Maps of shape (batch, channels, height, width), in standardised units.
        mask: The region-of-interest mask, broadcastable to ``maps``.

    Returns:
    -------
        The three statistics, each averaged across maps.

    """
    inside = mask.bool()
    diff_x = (maps[..., :, 1:] - maps[..., :, :-1]).abs()
    diff_y = (maps[..., 1:, :] - maps[..., :-1, :]).abs()
    # Only pixel pairs with both ends inside the region contribute.
    pairs_x = (inside[..., :, 1:] & inside[..., :, :-1]).float()
    pairs_y = (inside[..., 1:, :] & inside[..., :-1, :]).float()
    roughness = (
        (diff_x * pairs_x).sum(dim=(-2, -1)) + (diff_y * pairs_y).sum(dim=(-2, -1))
    ) / (pairs_x.sum(dim=(-2, -1)) + pairs_y.sum(dim=(-2, -1)))

    count = inside.float().sum(dim=(-2, -1))
    mean = (maps * inside).sum(dim=(-2, -1)) / count
    variance = (((maps - mean[..., None, None]) ** 2) * inside).sum(dim=(-2, -1)) / count
    return {
        "roughness": round(float(roughness.mean()), 5),
        "spatial_sd": round(float(variance.sqrt().mean()), 5),
        "mean": round(float(mean.mean()), 5),
    }


def measure(
    run_dir: Path, splits: dict[str, Any], device: torch.device, draws: int
) -> dict[str, Any]:
    """Field statistics of one run's generated maps over the train split."""
    logger.info("Measuring %s", run_dir)
    model = TTTSWIGAN.load_from_checkpoint(
        run_dir / "checkpoints" / "last.ckpt", loss_fn="l1", map_location=device
    )
    model.eval().to(device)

    data = _materialise(splits["train"], device)
    indices = torch.arange(len(data["target_maps"]), device=device)
    per_draw = []
    for draw in range(draws):
        seed = SCORING_NOISE_SEED + draw
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        per_draw.append(field_statistics(_generate(model, data, indices), data["mask"]))
    return {
        "run": run_dir.as_posix(),
        "seed": 61 if "seed61" in run_dir.as_posix() else 60,
        "generated": {
            key: round(float(np.mean([d[key] for d in per_draw])), 5) for key in per_draw[0]
        },
    }


def main() -> None:
    """Measure one or more runs' generated maps against the observed ones."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--data", type=Path, default=Path("data_train.parquet"))
    parser.add_argument("--out", type=Path, default=Path("field_stats.json"))
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits = build_splits(args.data)
    observed_data = _materialise(splits["train"], device)
    observed = field_statistics(observed_data["target_maps"], observed_data["mask"])
    results = {
        "observed": observed,
        "runs": [measure(run_dir, splits, device, args.draws) for run_dir in args.run_dirs],
    }
    with args.out.open("w") as handle:
        json.dump(results, handle, indent=2)
    logger.info("Wrote %s", args.out)

    print(
        f"\n{'observed maps':<34} roughness {observed['roughness']:.4f}  "
        f"sd {observed['spatial_sd']:.4f}  mean {observed['mean']:+.4f}"
    )
    for result in results["runs"]:
        values = result["generated"]
        print(
            f"{result['run'].split('/')[0][:34]:<34} roughness {values['roughness']:.4f}  "
            f"sd {values['spatial_sd']:.4f}  mean {values['mean']:+.4f}   "
            f"({values['roughness'] / observed['roughness']:.2f}x observed roughness)"
        )


if __name__ == "__main__":
    main()
