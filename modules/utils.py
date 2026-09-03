"""Module containing useful building blocks utils."""

from __future__ import annotations

from enum import Enum
from typing import TypeAlias

import torch
from torch import nn

Channels: TypeAlias = list[int]

NOISE_WEIGHT_INITS = ("randn", "zeros", "randn3")


class ModelFlavour(Enum):
    """Class containing the spatial encoder's/decoder's input/output channels.

    Returns a tuple of lists of ints With the first and second element being
    respectively the encoder and decoder channels.
    """

    Small: tuple[Channels, Channels] = ([8, 16, 32, 64], [64, 32, 16, 8])
    Medium: tuple[Channels, Channels] = ([16, 32, 64, 128], [128, 64, 32, 16])
    Large: tuple[Channels, Channels] = ([32, 64, 128, 256], [256, 128, 64, 32])


def init_noise_weights(num_channels: int, mode: str = "randn") -> nn.Parameter:
    """Build the per-channel gain applied to a block's injected noise field.

    ``mode`` selects the initialization:

    "randn"
        A standard normal draw, which is what every round through 8 trained with. Note
        that this is not an initialization the optimizer goes on to refine: the gains land
        at rms ~1.0 while the conv weights the noise is added to sit at rms 0.017-0.16, and
        over epochs 29->119 of round 8 both families move by the same ~1e-3 -- the distance
        anything travels at ``start_lr`` 1e-5 annealed to 1e-7. A 0.08% relative change
        makes the draw a *fixed random hyperparameter*, 3840 per-channel gains chosen by
        seed and then carried for the whole run.

    "zeros"
        StyleGAN's convention: the block starts as its deterministic self and has to earn
        any noise it uses. Given the drift budget above, expect the gains to stay near zero
        for a 120-epoch run rather than to grow -- so this is closer to "trained with
        injection off" than to "trained with injection free to find its own scale". Pair it
        with ``noise_weight_lr_scale`` if the intent is the latter.

    "randn3"
        A standard normal draw scaled to rms 3.0, for one specific experiment. Round 10's
        arm C (zero init, ``noise_weight_lr_scale`` 10000) fits a saturating approach to an
        asymptote of 1.489 and reaches 98.5% of it, which reads as an attractor -- but every
        run so far started at 0 or rms 1.0, i.e. *below* that value, and an approach from
        below cannot distinguish an attractor from a cosine schedule expiring where it
        happens to expire. Starting above it makes the two readings disagree: a genuine
        attractor pulls the gains back DOWN to ~1.49, while an expiring schedule leaves them
        near 3.0 (or lets AdamW's decoupled ``weight_decay`` walk them toward 0 with no
        preferred stop). Only meaningful paired with a large ``noise_weight_lr_scale`` --
        at scale 1.0 the gains cannot travel, so this is just "randn at 3x" and answers
        nothing.

    Args:
    ----
        num_channels: Number of output channels of the block, one gain each.
        mode: One of ``NOISE_WEIGHT_INITS``.

    Returns:
    -------
        A Parameter of shape (num_channels,).

    """
    if mode not in NOISE_WEIGHT_INITS:
        raise NotImplementedError(
            f"Unknown noise weight init: {mode}. Please provide one of {NOISE_WEIGHT_INITS}."
        )
    if mode == "zeros":
        return nn.Parameter(torch.zeros(num_channels))
    if mode == "randn3":
        return nn.Parameter(torch.randn(num_channels) * 3.0)
    return nn.Parameter(torch.randn(num_channels))


def glorot_init(m: nn.Module) -> None:
    """Util function to initialize all Conv and linear blocks with Glorot."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear, nn.Conv3d)):  # noqa
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def glorot_gru_init(gru_layer: nn.Module) -> None:
    """Util function to initialize all GRU cells with Glorot an orthogonal initialization."""
    for name, param in gru_layer.named_parameters():
        if "weight_ih" in name:
            nn.init.xavier_uniform_(param.data)
        elif "weight_hh" in name:
            nn.init.orthogonal_(param.data)
        elif "bias" in name:
            nn.init.zeros_(param.data)
            # Optional: bias for update gate (helps training)
            n = param.size(0)
            param.data[n // 3 : n // 3 * 2].fill_(1.0)
