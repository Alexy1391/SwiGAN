"""Fresh-critic test: how hard one frozen generator's samples are to detect.

Round 8 produced the first real adversarial equilibrium of the whole log -- both heads
lost separation under the rebalance, with nothing in the critic's objective changed, so
the only thing that differed was the generator they faced. That is consistent with two
opposite readings: the generator learned structure this critic cannot see, or it learned
to beat this one opponent. The critic that trained alongside it cannot tell them apart,
because it is the opponent.

This harness asks a critic that has never seen the generator. Freeze the generator, train
a brand-new critic of identical architecture from scratch on real-vs-generated
classification, and record how many steps it needs to separate held-out samples. A
generator whose samples are genuinely harder makes a fresh critic work longer for the same
AUC; one that merely exploited its training opponent does not.

Protocol, and why each piece is fixed rather than matched to the run:

* The probe is one architecture for every run -- ``--probe-aggregation learned`` by
  default, whatever the run itself used. The measurement is a property of the generator,
  so the instrument has to be the same instrument across arms. "learned" is the more
  capable read-out of the two (round 7), so it does not understate how separable a
  generator's samples are.
* The objective is binary cross-entropy, not the Wasserstein loss the run trained under.
  Detection is the question here, and BCE is what makes AUC 1.00 a meaningful saturation
  point instead of an unbounded score.
* No DiffAugment and no random flip/crop. The GAN's critic sees both; a probe measuring
  how distinguishable the generator's output is should see the output.
* The generator runs in eval mode, drawing fresh noise every step during probe training
  and a fixed set of ``--draws`` noise seeds at evaluation, averaged (round 7's
  correction: the generator is stochastic well beyond the z vector).
* At least two probe seeds per generator, per the two-seed rule round 5 set. The probe's
  own initialisation is a nuisance parameter like any other seed in this project.

``--batching`` exists because the trunk contains minibatch discrimination
(``base_discriminator.py:49``), whose features for one sample depend on the other samples
in the batch:

* ``mixed`` (default) puts real and generated maps in the same forward pass, so the
  minibatch statistic cannot act as a label. This is the honest per-sample measurement.
* ``split`` scores them in separate homogeneous batches, which is what ``critic_step``
  (swigan_lit.py:386) and ``utils.critic_diagnostics`` both do. If a probe separates far
  faster under ``split`` than under ``mixed``, part of what every AUC in the log has been
  measuring is batch composition rather than per-sample realism.

Usage:
    python -m utils.fresh_critic <run_dir> [<run_dir> ...] --out fresh_critic.json
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
import torch.nn.functional as F  # noqa: N812
import yaml
from torch import nn
from torch.utils.data import DataLoader

from modules.discriminator.base_discriminator import BaseDiscriminator
from modules.discriminator.frame_discriminator import FrameDiscriminator
from modules.discriminator.patch_gan_discriminator import PatchGANDiscriminator
from swigan.engines.swigan_lit import TTTSWIGAN
from utils.critic_diagnostics import auc
from utils.preprocessing import (
    coerce_comma_decimal_columns,
    dataframe_to_rasters,
    fill_all_missing_pixels,
)
from utils.swi_dataset import build_train_val_test_datasets

logger = logging.getLogger(__name__)

TRAIN_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "train_swigan.yaml"

# Same convention as the other harnesses: fixed, and deliberately not a training seed.
# Draw k evaluates against generator noise seeded with SCORING_NOISE_SEED + k.
SCORING_NOISE_SEED = 1234

# Which of the 709 train-split maps the probe trains on. Fixed so every generator is
# probed against the same partition; the held-out fifth (142 maps) is where the AUC curve
# is read, because it is large enough to resolve the differences this test is looking for.
# The GAN's own val and test splits are scored too, but at n=23 and n=24 they carry the
# +/-0.08 standard error round 7's correction quantified.
PROBE_SPLIT_SEED = 7
PROBE_TRAIN_FRACTION = 0.8

HEADS = ("patch", "frame")
# Reported crossings. 1.00 is the number rounds 2-4 quoted; the lower two are there
# because a generator that never lets the probe saturate still has a curve worth timing.
AUC_THRESHOLDS = (0.9, 0.99, 1.0)


def build_splits(data_path: Path) -> dict[str, Any]:
    """Build the three chronological splits the whole project is measured on.

    Dataset and split settings live in the training config rather than in the checkpoint,
    and have been identical for every run in the analysis (709 / 23 / 24).
    """
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
    return splits


def _materialise(dataset: Any, device: torch.device) -> dict[str, torch.Tensor]:
    """Load a whole split onto the device once, so the probe loop is pure compute."""
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    batch = next(iter(loader))
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def _generate(
    model: TTTSWIGAN,
    data: dict[str, torch.Tensor],
    indices: torch.Tensor,
    batch_size: int = 64,
) -> torch.Tensor:
    """Run the frozen generator over the given samples, drawing noise from the ambient RNG.

    Noise is not seeded here: the caller seeds it, either once per probe run (training,
    where every step should see a new draw) or once per evaluation draw (scoring, where
    the draws have to be the same for every run being compared).
    """
    outputs = []
    for chunk in indices.split(batch_size):
        z = torch.randn(len(chunk), model.hparams.z_dim, device=data["target_maps"].device)
        outputs.append(
            model(
                input_maps=data["input_maps"][chunk],
                input_timestamps=data["timestamps"][chunk],
                mask=data["mask"][chunk],
                noise_vector=z,
            )
        )
    return torch.cat(outputs)


def _build_probe(model: TTTSWIGAN, aggregation: str, device: torch.device) -> nn.ModuleDict:
    """Build a brand-new critic of the run's architecture, freshly initialised.

    Same trunk, same two heads, same channel widths as ``TTTSWIGAN.__init__``. Only the
    patch head's aggregation is pinned rather than copied, so the instrument does not
    change between the runs being compared.
    """
    probe = nn.ModuleDict(
        {
            "base": BaseDiscriminator(
                input_channels=model.hparams.output_channels,
                output_channels=[32, 64, 128],
                mbd_output_channels=64,
            ),
            "patch": PatchGANDiscriminator(input_channels=128 + 64, aggregation=aggregation),
            "frame": FrameDiscriminator(input_channels=128 + 64),
        }
    )
    return probe.to(device)


def _head_logits(probe: nn.ModuleDict, maps: torch.Tensor) -> dict[str, torch.Tensor]:
    """One logit per sample per head. Under "mean" aggregation the grid is averaged."""
    base, _ = probe["base"](maps)
    patch, _ = probe["patch"](base)
    frame, _ = probe["frame"](base)
    return {
        "patch": patch.flatten(1).mean(dim=1),
        "frame": frame.flatten(1).mean(dim=1),
    }


def _batch_logits(
    probe: nn.ModuleDict, real: torch.Tensor, fake: torch.Tensor, batching: str
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Score one batch of real and one of generated maps, as (real, fake) per head.

    Under ``mixed`` the two go through the trunk together, so minibatch discrimination
    sees a batch that is half real; under ``split`` they go separately, as the GAN's own
    critic step does.
    """
    if batching == "mixed":
        logits = _head_logits(probe, torch.cat([real, fake]))
        return {head: (logits[head][: len(real)], logits[head][len(real) :]) for head in HEADS}
    logits_real = _head_logits(probe, real)
    logits_fake = _head_logits(probe, fake)
    return {head: (logits_real[head], logits_fake[head]) for head in HEADS}


@torch.no_grad()
def _score_set(
    probe: nn.ModuleDict,
    real: torch.Tensor,
    fake: torch.Tensor,
    batching: str,
    batch_size: int,
) -> dict[str, float]:
    """Held-out AUC per head over a set of paired real and generated maps."""
    was_training = probe.training
    # eval() so spectral norm does not run a power iteration while measuring.
    probe.eval()
    collected: dict[str, list[np.ndarray]] = {
        f"{head}_{kind}": [] for head in HEADS for kind in ("real", "fake")
    }
    for start in range(0, len(real), batch_size):
        stop = start + batch_size
        scores = _batch_logits(probe, real[start:stop], fake[start:stop], batching)
        for head, (real_scores, fake_scores) in scores.items():
            collected[f"{head}_real"].append(real_scores.cpu().numpy())
            collected[f"{head}_fake"].append(fake_scores.cpu().numpy())
    probe.train(was_training)
    return {
        head: auc(
            np.concatenate(collected[f"{head}_real"]), np.concatenate(collected[f"{head}_fake"])
        )
        for head in HEADS
    }


def _steps_to(curve: list[dict[str, Any]], head: str, split: str, threshold: float) -> int | None:
    """First evaluated step at which this head's AUC on this split reaches ``threshold``."""
    for point in curve:
        if point[split][head] >= threshold:
            return point["step"]
    return None


def probe_generator(
    run_dir: Path,
    splits: dict[str, Any],
    device: torch.device,
    probe_seed: int,
    steps: int,
    eval_every: int,
    draws: int,
    batch_size: int,
    lr: float,
    batching: str,
    aggregation: str,
    checkpoint: str = "last.ckpt",
) -> dict[str, Any]:
    """Train one fresh critic against one frozen generator and return its AUC curve.

    Args:
    ----
        run_dir: The run whose generator is being probed.
        splits: The three chronological splits, from ``build_splits``.
        device: Where to run.
        probe_seed: Seeds the probe's initialisation and its training-time noise draws.
        steps: Number of probe updates.
        eval_every: Evaluate the held-out AUC every this many steps.
        draws: Generator-noise draws averaged at each evaluation.
        batch_size: Real maps per step; the same number of generated maps accompanies them.
        lr: Probe learning rate. The absolute step counts scale with it, so it is held
            fixed across every run being compared and recorded in the output.
        batching: "mixed" or "split" -- see the module docstring.
        aggregation: Patch-head aggregation of the probe, pinned across runs.
        checkpoint: Which checkpoint under ``<run_dir>/checkpoints`` to freeze.

    Returns:
    -------
        The run's configuration, the AUC curve, and the step at which each head first
        crossed each threshold.

    """
    model = TTTSWIGAN.load_from_checkpoint(
        run_dir / "checkpoints" / checkpoint, loss_fn="l1", map_location=device
    )
    model.eval().to(device)
    model.requires_grad_(False)

    data = {name: _materialise(splits[name], device) for name in ("train", "val", "test")}
    count = len(data["train"]["target_maps"])
    order = np.random.default_rng(PROBE_SPLIT_SEED).permutation(count)
    cut = int(PROBE_TRAIN_FRACTION * count)
    fit_index = torch.tensor(np.sort(order[:cut]), device=device)
    held_index = torch.tensor(np.sort(order[cut:]), device=device)

    # Evaluation fakes are generated once per draw, before the probe is seeded, so the
    # probe's own RNG stream is unaffected by how many draws are asked for.
    eval_sets: dict[str, dict[str, Any]] = {
        "heldout_train": {"real": data["train"]["target_maps"][held_index], "fakes": []},
        "gan_heldout": {
            "real": torch.cat([data["val"]["target_maps"], data["test"]["target_maps"]]),
            "fakes": [],
        },
    }
    for draw in range(draws):
        seed = SCORING_NOISE_SEED + draw
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        eval_sets["heldout_train"]["fakes"].append(_generate(model, data["train"], held_index))
        eval_sets["gan_heldout"]["fakes"].append(
            torch.cat(
                [
                    _generate(
                        model,
                        data[name],
                        torch.arange(len(data[name]["target_maps"]), device=device),
                    )
                    for name in ("val", "test")
                ]
            )
        )

    torch.manual_seed(probe_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(probe_seed)
    probe = _build_probe(model, aggregation, device)
    # Betas match the GAN's own critic optimiser; no weight decay, because the probe is a
    # measuring instrument and rounds 3-4 showed what decay does to this architecture.
    optimiser = torch.optim.Adam(probe.parameters(), lr=lr, betas=(0.5, 0.999))
    sampler = torch.Generator(device="cpu").manual_seed(probe_seed)

    def evaluate(step: int) -> dict[str, Any]:
        point: dict[str, Any] = {"step": step}
        for name, contents in eval_sets.items():
            per_draw = [
                _score_set(probe, contents["real"], fake, batching, batch_size)
                for fake in contents["fakes"]
            ]
            point[name] = {
                head: round(float(np.mean([d[head] for d in per_draw])), 4) for head in HEADS
            }
            point[f"{name}_sd"] = {
                head: round(float(np.std([d[head] for d in per_draw], ddof=1)), 4)
                if draws > 1
                else 0.0
                for head in HEADS
            }
        return point

    probe.train()
    curve = [evaluate(0)]
    losses: list[float] = []
    for step in range(1, steps + 1):
        picks = fit_index[torch.randperm(len(fit_index), generator=sampler)[:batch_size].to(device)]
        real = data["train"]["target_maps"][picks]
        fake = _generate(model, data["train"], picks).detach()

        scores = _batch_logits(probe, real, fake, batching)
        loss = sum(
            F.binary_cross_entropy_with_logits(real_scores, torch.ones_like(real_scores))
            + F.binary_cross_entropy_with_logits(fake_scores, torch.zeros_like(fake_scores))
            for real_scores, fake_scores in scores.values()
        )
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        losses.append(float(loss.detach()))

        if step % eval_every == 0:
            curve.append(evaluate(step))

    hparams = model.hparams
    return {
        "run": run_dir.as_posix(),
        "checkpoint": checkpoint,
        "seed": 61 if "seed61" in run_dir.as_posix() else 60,
        "probe_seed": probe_seed,
        "probe": {
            "batching": batching,
            "aggregation": aggregation,
            "lr": lr,
            "batch_size": batch_size,
            "steps": steps,
            "draws": draws,
        },
        "generator_hparams": {
            key: hparams.get(key)
            for key in (
                "patch_aggregation",
                "gradient_penalty_reduction",
                "lambda_penalty",
                "image_distance_weight",
                "feature_matching_weight",
                "weight_decay",
            )
        },
        "n": {name: int(len(contents["real"])) for name, contents in eval_sets.items()},
        "curve": curve,
        "steps_to": {
            split: {
                head: {
                    str(threshold): _steps_to(curve, head, split, threshold)
                    for threshold in AUC_THRESHOLDS
                }
                for head in HEADS
            }
            for split in ("heldout_train", "gan_heldout")
        },
        "final_auc": {
            split: {head: curve[-1][split][head] for head in HEADS}
            for split in ("heldout_train", "gan_heldout")
        },
        "max_auc": {
            split: {head: max(point[split][head] for point in curve) for head in HEADS}
            for split in ("heldout_train", "gan_heldout")
        },
        "final_loss": round(float(np.mean(losses[-20:])), 4),
    }


def main() -> None:
    """Probe one or more frozen generators and write the AUC curves as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--data", type=Path, default=Path("data_train.parquet"))
    parser.add_argument("--out", type=Path, default=Path("fresh_critic.json"))
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--draws", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--probe-seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--batching", choices=("mixed", "split"), default="mixed")
    parser.add_argument("--probe-aggregation", choices=("learned", "mean"), default="learned")
    parser.add_argument("--checkpoint", default="last.ckpt")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits = build_splits(args.data)
    results = []
    for run_dir in args.run_dirs:
        for probe_seed in args.probe_seeds:
            logger.info(
                "Probing %s (probe seed %d, %s batching)", run_dir, probe_seed, args.batching
            )
            results.append(
                probe_generator(
                    run_dir=run_dir,
                    splits=splits,
                    device=device,
                    probe_seed=probe_seed,
                    steps=args.steps,
                    eval_every=args.eval_every,
                    draws=args.draws,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    batching=args.batching,
                    aggregation=args.probe_aggregation,
                    checkpoint=args.checkpoint,
                )
            )
            with args.out.open("w") as handle:
                json.dump(results, handle, indent=2)

    logger.info("Wrote %s", args.out)
    for result in results:
        crossings = result["steps_to"]["heldout_train"]
        print(
            f"\n=== {result['run']} === probe seed {result['probe_seed']}, "
            f"{result['probe']['batching']} batching"
        )
        for head in HEADS:
            steps_099 = crossings[head]["0.99"]
            steps_100 = crossings[head]["1.0"]
            print(
                f"    {head:<5} held-out AUC "
                f"final {result['final_auc']['heldout_train'][head]:.3f} "
                f"max {result['max_auc']['heldout_train'][head]:.3f}  "
                f"steps to 0.99 {steps_099 if steps_099 is not None else '  --'}  "
                f"to 1.00 {steps_100 if steps_100 is not None else '  --'}"
            )


if __name__ == "__main__":
    main()
