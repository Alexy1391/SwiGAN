"""Module containing useful building blocks utils."""

from __future__ import annotations

from enum import Enum
from typing import TypeAlias

import torch
from torch import nn

Channels: TypeAlias = list[int]

NOISE_WEIGHT_INITS = ("randn", "zeros", "randn3")
COHERENT_NOISE_INITS = ("zeros", "randn")


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


def init_coherent_noise_weights(
    num_channels: int, mode: str | float = "zeros"
) -> nn.Parameter:
    """Build the per-channel gain applied to a block's *spatially coherent* noise.

    WHY THIS EXISTS. The field injected by :func:`init_noise_weights` is drawn per pixel, so
    it is speckle: averaging it over a region drives it to zero. Measured on the Corse
    250-epoch arm, the ensemble's region-mean spread divided by its pixel spread is 0.13-0.25,
    while the error it is meant to span sits at 0.63-0.90 -- the truth's error is a coherent
    displacement of the whole field and the model's ensemble is not. Reaching November's
    required region-mean spread of 0.263 SWI through a channel that survives averaging at
    0.25 would take a pixel spread of about 1.3 SWI on a field whose range is roughly 0 to 1,
    so no setting of the speckle gains produces a calibrated region mean. That is why fifty
    epochs of a proper score moved these gains along a random walk: there was no reachable
    direction for the score to push them in.

    The coherent channel is the missing mode. Its draw is constant over height and width, so
    it passes through spatial averaging undiminished and the region-mean ensemble can be as
    wide as the data require without the maps turning to noise.

    ``mode`` selects the initialization:

    a float
        Every gain starts at this constant. THIS IS THE ONE TO FINE-TUNE WITH, and 0.1 is the
        measured starting point -- see the warning about zero below. On the 250-epoch Corse
        weights, 0.05 puts the ensemble's region/pixel spread ratio at 0.41, 0.1 at 0.61 and
        0.2 at 0.77, against the 0.63-0.90 of the error being spanned, so 0.1 sits mid-range
        with room for a proper score to push either way.

    "zeros"
        Contributes exactly nothing at step 0, which makes it the safe thing to load a
        checkpoint with -- and a trap to train from. MEASURED: over 11 epochs of an energy
        fine-tune at ``noise_weight_lr_scale`` 100 the gains went 0.0023 -> 0.0032 rms, a
        growth exponent of 0.435 in the epoch count, extrapolating to 0.006 by epoch 50
        against the ~0.1 the channel needs to do anything. That is a random walk, and the
        reason is structural: at gain zero the coherent perturbation is independent of the
        differences the members already have, so its first-order contribution to the score's
        spread term averages to nothing and the benefit is second order. The origin is a
        plateau, and a proper score cannot bootstrap the channel off it. Use a float instead.

    "randn"
        A standard normal draw at rms 1.0, for training from scratch. Far too large to drop
        on a trained checkpoint: at gain 0.4 the region-mean spread is already 14 times the
        shipped ensemble's.

    Args:
    ----
        num_channels: Number of output channels of the block, one gain each.
        mode: One of ``COHERENT_NOISE_INITS``.

    Returns:
    -------
        A Parameter of shape (num_channels,).

    """
    if isinstance(mode, bool) or not isinstance(mode, (str, float, int)):
        raise NotImplementedError(
            f"Unknown coherent noise init: {mode!r}. Provide one of {COHERENT_NOISE_INITS} "
            "or a float for a constant gain."
        )
    if not isinstance(mode, str):
        return nn.Parameter(torch.full((num_channels,), float(mode)))
    if mode not in COHERENT_NOISE_INITS:
        raise NotImplementedError(
            f"Unknown coherent noise init: {mode}. Please provide one of {COHERENT_NOISE_INITS}, "
            "or a float for a constant gain."
        )
    if mode == "randn":
        return nn.Parameter(torch.randn(num_channels))
    return nn.Parameter(torch.zeros(num_channels))


def draw_coherent_field(
    gains: nn.Parameter | None,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    mask: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Draw a block's spatially coherent noise, or None when the channel is off.

    The field is (batch, channels, 1, 1) -- one draw per sample per channel, constant over
    height and width -- so it broadcasts across the map and survives the spatial averaging
    that reduces the per-pixel field of :func:`init_noise_weights` to nothing. See
    :func:`init_coherent_noise_weights` for why the model needs a channel with that property.

    Args:
    ----
        gains: The block's coherent gains, whose length gives the channel count. None when
            the block was built without the channel, in which case nothing is drawn -- the
            random stream is left untouched so a model without the channel is unaffected.
        batch_size: Number of samples in the batch.
        device: Device to draw on.
        dtype: Dtype of the activations the field is added to.
        mask: Region mask of shape (batch, 1, h, w), or None. When given, the field is zeroed
            outside the region, keeping the block's invariant that everything handed to a
            convolution is zero on the padding.

    Returns:
    -------
        A Tensor broadcastable against (batch, channels, h, w), or None.

    """
    if gains is None:
        return None
    field = torch.randn((batch_size, gains.shape[0], 1, 1), device=device, dtype=dtype)
    if mask is not None:
        field = field * mask.to(dtype)
    return field


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
