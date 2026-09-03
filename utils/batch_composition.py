"""How much of the critic's measured separation survives real and fake sharing a batch.

Every AUC in this project's analysis comes from ``utils.critic_diagnostics``, which scores
a whole split of observed maps in one pass and a whole split of generated maps in another.
The trunk carries minibatch discrimination (``base_discriminator.py:49``): each sample's
last 64 channels are sums of similarities to the *other samples in its batch*. In a batch
that is entirely real or entirely generated, that statistic is a function of the class
itself, so a critic can separate the two without any per-sample judgement.

This harness scores the same checkpoint twice: once in homogeneous batches, the way the
critic was trained and the way every published number was measured, and once with the two
classes interleaved in one forward pass, which removes that route and leaves only what the
critic can tell about a map on its own. Batch size is matched at 64 in both modes, so the
minibatch sum runs over the same number of neighbours and composition is the only
difference.

This is a measurement question, not a bug: minibatch discrimination is doing what
Salimans et al. designed it to do. But the gap tells you how much of a reported AUC is a
per-sample realism signal and how much is batch-level, and only the former is something a
single generated map can be judged on.

Usage:
    python -m utils.batch_composition <run_dir> [<run_dir> ...] --out batch_composition.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from swigan.engines.swigan_lit import TTTSWIGAN
from utils.critic_diagnostics import auc
from utils.fresh_critic import _generate, _head_logits, _materialise, build_splits

logger = logging.getLogger(__name__)

SCORING_NOISE_SEED = 1234
DEFAULT_DRAWS = 3
BATCH_SIZE = 64

HEADS = ("patch", "frame")


def _as_probe(model: TTTSWIGAN) -> nn.ModuleDict:
    """View the trained critic through the same interface the fresh-critic probe uses."""
    return nn.ModuleDict(
        {
            "base": model.base_critic,
            "patch": model.patch_critic,
            "frame": model.frame_critic,
        }
    )


@torch.no_grad()
def _homogeneous(critic: nn.ModuleDict, maps: torch.Tensor) -> dict[str, np.ndarray]:
    """Score one class at a time, in batches of one class -- the published protocol."""
    collected: dict[str, list[np.ndarray]] = {head: [] for head in HEADS}
    for start in range(0, len(maps), BATCH_SIZE):
        logits = _head_logits(critic, maps[start : start + BATCH_SIZE])
        for head in HEADS:
            collected[head].append(logits[head].cpu().numpy())
    return {head: np.concatenate(value) for head, value in collected.items()}


@torch.no_grad()
def _interleaved(
    critic: nn.ModuleDict, real: torch.Tensor, fake: torch.Tensor
) -> dict[str, np.ndarray]:
    """Score both classes together, half a batch each, so composition carries no label."""
    half = BATCH_SIZE // 2
    collected: dict[str, list[np.ndarray]] = {
        f"{head}_{kind}": [] for head in HEADS for kind in ("real", "fake")
    }
    for start in range(0, len(real), half):
        stop = start + half
        count = len(real[start:stop])
        logits = _head_logits(critic, torch.cat([real[start:stop], fake[start:stop]]))
        for head in HEADS:
            collected[f"{head}_real"].append(logits[head][:count].cpu().numpy())
            collected[f"{head}_fake"].append(logits[head][count:].cpu().numpy())
    return {key: np.concatenate(value) for key, value in collected.items()}


def compare(
    run_dir: Path, splits: dict[str, Any], device: torch.device, draws: int
) -> dict[str, Any]:
    """Score one checkpoint both ways on all three splits, averaged over noise draws."""
    logger.info("Rescoring %s", run_dir)
    model = TTTSWIGAN.load_from_checkpoint(
        run_dir / "checkpoints" / "last.ckpt", loss_fn="l1", map_location=device
    )
    model.eval().to(device)
    critic = _as_probe(model)

    result: dict[str, Any] = {
        "run": run_dir.as_posix(),
        "seed": 61 if "seed61" in run_dir.as_posix() else 60,
        "draws": draws,
        "splits": {},
    }
    for name in ("train", "val", "test"):
        data = _materialise(splits[name], device)
        real = data["target_maps"]
        indices = torch.arange(len(real), device=device)
        homogeneous: dict[str, list[float]] = {head: [] for head in HEADS}
        interleaved: dict[str, list[float]] = {head: [] for head in HEADS}
        for draw in range(draws):
            seed = SCORING_NOISE_SEED + draw
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            fake = _generate(model, data, indices)
            scores_real = _homogeneous(critic, real)
            scores_fake = _homogeneous(critic, fake)
            mixed = _interleaved(critic, real, fake)
            for head in HEADS:
                homogeneous[head].append(auc(scores_real[head], scores_fake[head]))
                interleaved[head].append(auc(mixed[f"{head}_real"], mixed[f"{head}_fake"]))
        result["splits"][name] = {
            "n": int(len(real)),
            **{
                f"{head}_{key}": round(float(np.mean(values[head])), 4)
                for head in HEADS
                for key, values in (("homogeneous", homogeneous), ("interleaved", interleaved))
            },
            **{
                f"{head}_delta": round(
                    float(np.mean(interleaved[head]) - np.mean(homogeneous[head])), 4
                )
                for head in HEADS
            },
        }
    return result


def main() -> None:
    """Rescore one or more runs under both batch compositions and write the results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--data", type=Path, default=Path("data_train.parquet"))
    parser.add_argument("--out", type=Path, default=Path("batch_composition.json"))
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits = build_splits(args.data)
    results = [compare(run_dir, splits, device, args.draws) for run_dir in args.run_dirs]
    with args.out.open("w") as handle:
        json.dump(results, handle, indent=2)
    logger.info("Wrote %s", args.out)

    for result in results:
        print(f"\n=== {result['run']} ===")
        for split, values in result["splits"].items():
            print(
                f"    {split:<5} n={values['n']:<4} "
                + "  ".join(
                    f"{head} homogeneous {values[f'{head}_homogeneous']:.3f} "
                    f"interleaved {values[f'{head}_interleaved']:.3f} "
                    f"({values[f'{head}_delta']:+.3f})"
                    for head in HEADS
                )
            )


if __name__ == "__main__":
    main()
