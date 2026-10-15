"""Loading and driving the generator's spatially coherent noise channel at scoring time.

WHY THIS EXISTS. ``modules/utils.py`` builds the channel at whatever gain the trainer asks
for, and ``swigan/train.py`` can fine-tune it. Neither lets a scoring run SWEEP the gain, and
the gain is the one parameter of the channel that was never chosen the way this project chooses
parameters. The 250-epoch report set it to 0.1 from the coherence table of
:func:`modules.utils.init_coherent_noise_weights` -- a property of the weights rather than a
score on the scored window, so not test-fitted, but not selected either, while every other
shipped parameter of that report was picked on fourteen calibration windows with a
leave-one-window-out correction. :mod:`utils.dispersion_select` sweeps the gain through
:func:`coherent_gains` and :mod:`swigan.inference` applies the winner through the same context
manager, so the configuration that ships is the configuration that was scored.

AMPLITUDE ZERO IS THE CHANNEL OFF, not a zero gain. A zero gain would still draw the field and
advance the random stream, so the shipped ensemble would stop being reproducible bit for bit
from a model built without the channel -- and the grid's baseline point has to BE the shipped
ensemble, not a statistical equal of it. :func:`coherent_gains` detaches the parameters
instead, which is exactly the state a checkpoint written before the channel existed loads into.

AMPLITUDE ``None`` LEAVES THE CHECKPOINT'S OWN GAINS ALONE. That is the fine-tuned channel's
twenty trained vectors, which a constant would overwrite; it is also what a dispersion spec
written before this axis existed means, so those specs keep scoring what they scored.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from torch import nn

from modules.utils import COHERENT_GAIN_ATTRIBUTES
from swigan.engines.swigan_lit import TTTSWIGAN

logger = logging.getLogger(__file__)

GAIN_KEY = "coherent_noise_weights"


def coherent_gain_sites(model: nn.Module) -> list[tuple[nn.Module, str]]:
    """Every ``(block, attribute)`` pair holding one of the channel's per-channel gain vectors.

    Empty when the model was built without the channel, which is what every checkpoint before
    2026-09-18 holds.
    """
    return [
        (module, attribute)
        for module in model.modules()
        for attribute in COHERENT_GAIN_ATTRIBUTES
        if isinstance(getattr(module, attribute, None), nn.Parameter)
    ]


def checkpoint_carries_coherent_gains(checkpoint_path: str | Path) -> bool:
    """Whether the checkpoint's own state dict holds the channel's gains."""
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["state_dict"]
    return any(GAIN_KEY in key for key in state)


@contextmanager
def coherent_gains(model: nn.Module, amplitude: float | None) -> Iterator[None]:
    """Run the block with every coherent gain driven to ``amplitude``.

    ``None`` leaves the model's own gains untouched and yields immediately, which is the
    no-op that a model without the channel also takes. ``0.0`` detaches the gains, so
    :func:`modules.utils.draw_coherent_field` returns ``None``, nothing is drawn and the
    random stream is the one a model built without the channel sees -- the shipped ensemble,
    bit for bit, rather than something that merely scores the same. Any positive value fills
    all 3 840 gains with that constant, which is what
    :func:`modules.utils.init_coherent_noise_weights` does for a float init.

    The gains' values and the Parameter objects themselves are both restored on exit, so a
    checkpoint's trained channel survives a sweep that overwrote it.
    """
    if amplitude is None:
        yield
        return
    if amplitude < 0:
        raise ValueError(f"coherent amplitude must be non-negative, got {amplitude}")
    sites = coherent_gain_sites(model)
    if not sites:
        raise RuntimeError(
            "this model has no spatially coherent noise channel, so its amplitude cannot be "
            "set -- load the checkpoint through load_with_coherent_channel()"
        )
    saved = [getattr(module, attribute) for module, attribute in sites]
    values = [gain.detach().clone() for gain in saved]
    try:
        with torch.no_grad():
            for (module, attribute), gain in zip(sites, saved, strict=True):
                if amplitude == 0:
                    setattr(module, attribute, None)
                else:
                    gain.fill_(float(amplitude))
        yield
    finally:
        with torch.no_grad():
            for (module, attribute), gain, original in zip(sites, saved, values, strict=True):
                gain.copy_(original)
                setattr(module, attribute, gain)


def load_with_coherent_channel(
    checkpoint_path: str | Path, init: str | float = 0.0, **overrides: Any
) -> TTTSWIGAN:
    """Load a checkpoint into a model that HAS the channel, however the checkpoint was trained.

    A checkpoint written before the channel existed holds none of its gains, so turning the
    channel on means ``strict=False``. That is only safe if those gains are the ONLY thing
    missing: anything else absent is a real mismatch and has to raise rather than be silently
    replaced by a fresh init. This is ``swigan/train.py``'s guard applied to the scoring path.

    ``init`` is what the fresh gains start at and is immaterial to a caller that drives them
    through :func:`coherent_gains` afterwards; it matters only for a checkpoint that already
    carries gains, where it is ignored because the saved ones load over it.
    """
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["state_dict"]
    saved_keys = set(state)
    pretrained = any(GAIN_KEY in key for key in saved_keys)
    model = TTTSWIGAN.load_from_checkpoint(
        checkpoint_path, strict=pretrained, coherent_noise_init=init, **overrides
    )
    if not pretrained:
        missing = [key for key in model.state_dict() if key not in saved_keys]
        unexplained = [key for key in missing if GAIN_KEY not in key]
        if unexplained:
            raise ValueError(
                f"'{checkpoint_path}' is missing {len(unexplained)} parameters that are not "
                f"coherent-noise gains, e.g. {unexplained[:3]}. Loading it would silently "
                "re-initialise trained weights."
            )
        logger.info(
            f"'{checkpoint_path}' predates the coherent noise channel: {len(missing)} gain "
            f"vectors added at '{init}'. Every other parameter came from the checkpoint."
        )
    else:
        gains = torch.cat([state[k].flatten() for k in saved_keys if GAIN_KEY in k])
        logger.info(
            f"'{checkpoint_path}' carries its own coherent channel: {len(gains)} gains at "
            f"rms {float(gains.pow(2).mean().sqrt()):.4f}."
        )
    return model
