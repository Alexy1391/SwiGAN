"""Measure how loud the paper's noise injection actually is, per residual block.

The paper (Section 8.2.3, "Increasing variability in generated maps") specifies Gaussian noise
after each activation, at both training and inference time, "scaled by a learnable coefficient",
citing Karras et al. [2019] and Karras et al. [2020b]. This script measures the two things that
description leaves open and that decide what the mechanism actually does here:

  1. the ratio of the injected term's magnitude to the activation it is added to, per site --
     StyleGAN's learned scales end up small, so the noise perturbs detail without moving the
     composition; a ratio near 1 means the activation is half noise;
  2. how far the learnable coefficients have travelled from their initialisation, which says
     whether "learnable" is doing any work in this run.

Because the injected noise is standard normal and broadcast across channels, the injected term's
per-channel magnitude is exactly ``|noise_weights[c]|``, so the numerator needs no sampling: only
the activation's own scale has to be measured, via hooks on the two conv blocks of each site.

Inference-only. Same overrides as ``swigan/inference.py``.
"""

import logging

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from utils.stochasticity_probe import prepare

logger = logging.getLogger(__file__)


@hydra.main(config_path="../config", config_name="inference_swigan", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Report the noise-to-activation ratio at all 20 injection sites."""
    prepared = prepare(cfg)
    model = prepared["model"]
    roll = prepared["rollout"]
    device = prepared["device"]

    generator = model.generator
    sites = []  # (label, conv_module, noise_weight_param)
    for idx, block in enumerate(generator.encoder.layers):
        sites.append((f"encoder.{idx}.conv1", block.conv1, block.noise_weights1))
        sites.append((f"encoder.{idx}.conv2", block.conv2, block.noise_weights2))
    for idx, block in enumerate(generator.decoder.blocks):
        sites.append((f"decoder.{idx}.conv1", block.conv1, block.noise_weights1))
        sites.append((f"decoder.{idx}.conv2", block.conv2, block.noise_weights2))

    captured: dict[str, list[float]] = {label: [] for label, _, _ in sites}
    handles = []
    for label, module, _ in sites:
        def hook(_mod, _inp, out, label=label):
            captured[label].append(float(out.detach().float().pow(2).mean().sqrt()))
        handles.append(module.register_forward_hook(hook))

    # One rollout at the shipping configuration, so the activations are the ones inference sees.
    from utils.stochasticity_probe import run_arm

    try:
        run_arm(noise_std=1.0, injection="independent", seed=0, **roll)
    finally:
        for handle in handles:
            handle.remove()

    print("\n" + "=" * 92)
    print("INJECTED NOISE vs ACTIVATION, PER RESIDUAL BLOCK".center(92))
    print("shipping config · eval mode · noise term rms = rms(noise_weights)".center(92))
    print("=" * 92)
    print(f"{'site':22s} {'chans':>6s} {'act rms':>9s} {'noise rms':>10s} {'noise/act':>10s} "
          f"{'neg coefs':>10s}")
    print("-" * 92)
    ratios = []
    for label, _, weight in sites:
        act = float(np.mean(captured[label]))
        noise_rms = float(weight.detach().float().pow(2).mean().sqrt())
        ratio = noise_rms / act
        ratios.append(ratio)
        neg = float((weight.detach() < 0).float().mean())
        print(f"{label:22s} {weight.numel():6d} {act:9.4f} {noise_rms:10.4f} {ratio:10.3f} "
              f"{neg:9.0%}")
    print("-" * 92)
    print(f"{'median across sites':22s} {'':6s} {'':9s} {'':10s} {np.median(ratios):10.3f}")
    print("=" * 92)
    print("noise/act = magnitude of the injected term relative to the activation it is added to.")
    print("StyleGAN initialises this coefficient at zero and lets the network raise it; this")
    print("implementation initialises it at torch.randn, i.e. rms 1.0 with half the coefficients")
    print("negative (modules/generator/unet_frame_encoder.py:68).")
    print("=" * 92 + "\n")

    # How far have the "learnable" coefficients actually moved? Compare against a fresh draw of
    # the same initialiser, and against the conv weights of the same blocks as a control.
    torch.manual_seed(0)
    print(f"{'parameter group':34s} {'rms':>9s} {'|mean|':>9s}")
    print("-" * 56)
    all_w = torch.cat([w.detach().flatten() for _, _, w in sites]).float()
    ref = torch.randn(all_w.numel())
    print(f"{'noise_weights, trained':34s} {all_w.pow(2).mean().sqrt():9.4f} "
          f"{all_w.mean().abs():9.4f}")
    print(f"{'torch.randn, the initialiser':34s} {ref.pow(2).mean().sqrt():9.4f} "
          f"{ref.mean().abs():9.4f}")
    conv = torch.cat(
        [m.weight.detach().flatten() for _, mod, _ in sites for m in mod.modules()
         if isinstance(m, torch.nn.Conv2d)]
    ).float()
    print(f"{'conv weights of the same blocks':34s} {conv.pow(2).mean().sqrt():9.4f} "
          f"{conv.mean().abs():9.4f}")
    print()
    logger.info(f"device={device}")


if __name__ == "__main__":
    main()