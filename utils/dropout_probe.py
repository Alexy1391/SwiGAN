"""Measure what inference-time dropout contributes, on top of ``model.eval()``.

WHY THIS EXISTS. ``article_recherche.pdf`` §8.2.1 (p.37) introduces dropout specifically to stop
the generator ignoring the noise vector, citing Isola et al. [2018]:

    "UNet-based generators may ignore the noise vector and rely solely on input maps, producing
     similar outputs regardless of noise. To mitigate this issue, dropout layers are introduced
     in the last three layers of the downsampling phase and the first three layers of the
     upsampling phase."

The paper is explicit about the inference behaviour of its two *other* stochastic mechanisms --
stochastic depth is off ("During inference, all residual blocks are used", p.37) and the noise
injection is on ("during both training and inference", p.39) -- and says nothing at all about
dropout at inference. That matters, because in pix2pix, which is the citation, dropout IS the
noise source and IS applied at test time; the paper instead keeps z and the injection and adds
dropout on top, so it does not obviously need test-time dropout for diversity.

``inference.py:154`` calls ``model.eval()``, which makes every ``nn.Dropout`` the identity. So the
shipped model's dropout contributes exactly nothing at inference. This script measures what it
WOULD contribute if left active, without retraining anything.

WHAT IT MEASURES. Four arms, crossing the two shipping noise sources with dropout:

    deterministic control   z=0, injection off, dropout off   -- must be exactly 0.00000
    dropout only            z=0, injection off, dropout ON    -- dropout's isolated contribution
    ships today             z std=1, injection indep, off     -- the current configuration
    ships + dropout         z std=1, injection indep, ON      -- what enabling it would buy

Arm 1 is the validity check: if it is not identically zero, some other stochastic path is live in
eval mode and arm 2 is not attributable to dropout. Arm 2 is the number the question turns on.

WHAT IT CANNOT TELL YOU. These checkpoints were trained with dropout active in the first three
DECODER blocks only (9 modules at p=0.3, hardcoded at ``unet_frame_decoder.py:176``); the encoder
side is off because ``spatial_dropout`` is 0.0, so half the paper's prescription was never in the
training graph. This script reads out the half that exists. Whether the full prescription changes
what the generator learns is a training question, not an inference one.
"""

import logging
from contextlib import contextmanager

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from utils.stochasticity_probe import prepare, run_arm, summarise

logger = logging.getLogger(__file__)

# (label, noise_std, injection, dropout). The determinism control runs first so a non-zero
# reading invalidates the rest before any of it is interpreted.
ARMS: list[tuple[str, float, str, bool]] = [
    ("deterministic control  z=0, inj off, drop off", 0.0, "off", False),
    ("dropout only           z=0, inj off, drop ON ", 0.0, "off", True),
    ("ships today            std=1, inj indep, off ", 1.0, "independent", False),
    ("ships + dropout        std=1, inj indep, ON  ", 1.0, "independent", True),
]


@contextmanager
def dropout_active(model: torch.nn.Module, enabled: bool):
    """Put only the ``nn.Dropout`` modules into training mode, leaving everything else in eval.

    This is the precise question being asked: ``model.eval()`` stays in force for batch norm and
    for stochastic depth (whose own ``training`` flag lives on a different module class and is
    untouched here), so any spread that appears is dropout's and nothing else's.
    """
    if not enabled:
        yield
        return
    mods = [m for m in model.modules() if isinstance(m, torch.nn.Dropout) and m.p > 0]
    if not mods:
        raise RuntimeError(
            "no nn.Dropout module with p > 0 -- this checkpoint was trained with dropout "
            "disabled everywhere, so there is nothing to switch on"
        )
    for m in mods:
        m.train()
    try:
        yield
    finally:
        for m in mods:
            m.eval()


@hydra.main(config_path="../config", config_name="inference_swigan", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Run the four arms and print what dropout would add at inference."""
    prepared = prepare(cfg)
    shared = prepared["rollout"]
    model = prepared["model"]
    y_true, bool_mask = prepared["y_true"], prepared["bool_mask"]
    target_std = prepared["target_std"]

    live = [(n, m.p) for n, m in model.generator.named_modules()
            if isinstance(m, torch.nn.Dropout) and m.p > 0]
    logger.info(f"dropout modules with p>0: {len(live)}")
    for name, p in live:
        logger.info(f"    p={p}  {name}")
    if not live:
        logger.warning("nothing to measure -- every dropout in this checkpoint has p=0")

    rows = []
    for label, noise_std, injection, drop in ARMS:
        with dropout_active(model, drop):
            traj = run_arm(noise_std=noise_std, injection=injection, seed=0, **shared)
        stats = summarise(traj, y_true, bool_mask, target_std)
        rows.append((label, stats))
        logger.info(f"{label:46s} spread={stats['spread']:.5f} max_dev={stats['max_dev']:.2e}")

    control = rows[0][1]["spread"]
    print("\n" + "=" * 104)
    print("INFERENCE-TIME DROPOUT".center(104))
    print(f"{len(live)} dropout modules at p>0 · {prepared['num_members']} members · "
          f"{prepared['num_steps']} rollout steps".center(104))
    print("=" * 104)
    print(f"{'arm':46s} {'spread':>9s} {'cover':>8s} {'maxdev':>10s} {'rmse max':>9s} {'rmse mean':>10s}")
    print("-" * 104)
    for label, s in rows:
        print(f"{label:46s} {s['spread']:9.5f} {s['coverage'] * 100:7.1f}% {s['max_dev']:10.2e} "
              f"{s['rmse_max']:9.4f} {s['rmse_mean']:10.4f}")
    print("-" * 104)

    if control > 1e-9:
        print(f"\n!! VALIDITY CHECK FAILED: the deterministic control is {control:.2e}, not 0.")
        print("   Some stochastic path other than dropout is live in eval mode, so the")
        print("   'dropout only' arm below is NOT attributable to dropout. Stop here.")
    else:
        drop_only = rows[1][1]
        ships, ships_drop = rows[2][1], rows[3][1]
        print(f"\ncontrol is exactly 0 -- dropout is the only stochastic path added in arm 2.")
        print(f"\n  dropout's isolated contribution   spread {drop_only['spread']:.5f}"
              f"   ({drop_only['spread'] / ships['spread'] * 100:.0f}% of what ships today)")
        print(f"  adding it to the shipping arm     {ships['spread']:.5f} -> {ships_drop['spread']:.5f}"
              f"   ({ships_drop['spread'] / ships['spread']:.2f}x)")
        print(f"  envelope coverage                 {ships['coverage'] * 100:.1f}% -> "
              f"{ships_drop['coverage'] * 100:.1f}%")
        print(f"  ensemble-mean rmse max            {ships['rmse_max']:.4f} -> {ships_drop['rmse_max']:.4f}"
              f"   ({(ships_drop['rmse_max'] / ships['rmse_max'] - 1) * 100:+.1f}%)")
        print(f"  ensemble-mean rmse mean           {ships['rmse_mean']:.4f} -> {ships_drop['rmse_mean']:.4f}"
              f"   ({(ships_drop['rmse_mean'] / ships['rmse_mean'] - 1) * 100:+.1f}%)")
    print("=" * 104)

    out = cfg["model"].get("saving_path")
    if out:
        np.savez(
            f"{out}_dropout.npz",
            arms=np.array([r[0] for r in rows]),
            spread=np.array([r[1]["spread"] for r in rows]),
            coverage=np.array([r[1]["coverage"] for r in rows]),
            max_dev=np.array([r[1]["max_dev"] for r in rows]),
            rmse_max=np.array([r[1]["rmse_max"] for r in rows]),
            rmse_mean=np.array([r[1]["rmse_mean"] for r in rows]),
            num_dropout_modules=len(live),
        )
        logger.info(f"Saved to '{out}_dropout.npz'.")


if __name__ == "__main__":
    main()