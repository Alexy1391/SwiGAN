"""Tests of driving the coherent channel's amplitude at scoring and shipping time.

``utils/dispersion_select.py`` sweeps the amplitude of the generator's spatially coherent noise
channel over the calibration windows and ``swigan/inference.py`` applies the winner, both
through :func:`utils.coherent_channel.coherent_gains`. The selection is only worth what these
properties are: amplitude 0 IS the shipped ensemble rather than something that scores like it,
a sweep never leaves the model holding the last grid point's gains, and a checkpoint written
before the channel existed loads with the channel attached and nothing else re-initialised.
"""

from __future__ import annotations

from pathlib import Path

import lightning
import pytest
import torch

from swigan.engines.swigan_lit import TTTSWIGAN
from tests.test_coherent_noise import (
    CANVAS,
    COHERENT,
    DECODER_CHANNELS,
    ENCODER_CHANNELS,
    NUM_INPUT_CHANNELS,
    build_generator,
    coherence,
    corse_mask,
    set_gains,
)
from utils.coherent_channel import (
    checkpoint_carries_coherent_gains,
    coherent_gain_sites,
    coherent_gains,
    load_with_coherent_channel,
)


def coherent_parameters(model: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
    """Collect the channel's gain vectors as the model currently exposes them, by name."""
    return {
        name: param
        for name, param in model.named_parameters()
        if name.rsplit(".", 1)[-1] in COHERENT
    }


def run(generator: torch.nn.Module, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """One forward pass from a fixed random state, so two runs differ only by the model."""
    torch.manual_seed(1)
    with torch.no_grad():
        return generator(inputs, mask=mask)


def test_amplitude_zero_is_the_shipped_model_bit_for_bit() -> None:
    """The grid's baseline has to BE the shipped ensemble, not a statistical equal of it.

    A zero gain is not enough: it still draws the coherent field, which advances the random
    stream, so every per-pixel field drawn after it changes and the output moves. Detaching
    the gains is what leaves the stream -- and the output -- exactly as a model built without
    the channel produces them.
    """
    torch.manual_seed(0)
    shipped = build_generator(None)
    with_channel = build_generator(0.1)
    incompatible = with_channel.load_state_dict(shipped.state_dict(), strict=False)
    assert len(incompatible.missing_keys) == 20
    assert not incompatible.unexpected_keys
    shipped.eval()
    with_channel.eval()
    mask = corse_mask(4)
    inputs = torch.randn(4, NUM_INPUT_CHANNELS, CANVAS, CANVAS) * mask

    reference = run(shipped, inputs, mask)
    with coherent_gains(with_channel, 0.0):
        detached = run(with_channel, inputs, mask)
    assert torch.equal(reference, detached)

    set_gains(with_channel, white=1.0, coherent=0.0)
    set_gains(shipped, white=1.0, coherent=0.0)
    zero_gain = run(with_channel, inputs, mask)
    assert not torch.equal(run(shipped, inputs, mask), zero_gain), (
        "a zero gain should shift the random stream; if it does not, this test cannot tell "
        "detaching from zeroing"
    )


def test_a_positive_amplitude_drives_every_gain_and_widens_the_region_mean() -> None:
    """All 20 vectors take the constant, and the amplitude is a working one."""
    torch.manual_seed(0)
    generator = build_generator("zeros")
    with coherent_gains(generator, 0.25):
        gains = coherent_parameters(generator)
        assert len(gains) == 20
        assert all(torch.all(g == 0.25) for g in gains.values())
        driven = coherence(generator)
    with coherent_gains(generator, 0.0):
        off = coherence(generator)
    assert driven > 2 * off, f"region/pixel {driven:.3f} at 0.25 against {off:.3f} off"


def test_a_sweep_restores_the_checkpoints_own_gains() -> None:
    """Trained gains survive a sweep that overwrote them -- values and Parameter identity.

    Identity matters as much as the values: the optimizer's param groups hold the Parameter
    objects, so a sweep that swapped them would quietly disconnect a later fine-tune.
    """
    torch.manual_seed(0)
    generator = build_generator("randn")  # non-constant, as a fine-tuned channel's are
    before = {n: (p, p.detach().clone()) for n, p in coherent_parameters(generator).items()}

    for amplitude in (0.0, 0.3, 0.05, 0.0):
        with coherent_gains(generator, amplitude):
            live = coherent_parameters(generator)
            if amplitude == 0.0:
                assert live == {}, "amplitude 0 should detach every gain"
            else:
                assert len(live) == 20

    with pytest.raises(RuntimeError, match="interrupted"):
        with coherent_gains(generator, 0.0):
            raise RuntimeError("interrupted")

    after = coherent_parameters(generator)
    assert after.keys() == before.keys()
    for name, (param, values) in before.items():
        assert after[name] is param, name
        assert torch.equal(after[name].detach(), values), name


def test_none_leaves_the_model_alone_and_the_channel_is_required_otherwise() -> None:
    """None is the no-op every older spec means; a number on a model with no channel is a bug."""
    without = build_generator(None)
    with coherent_gains(without, None):
        pass
    assert coherent_gain_sites(without) == []
    with pytest.raises(RuntimeError, match="no spatially coherent noise channel"):
        with coherent_gains(without, 0.1):
            pass

    trained = build_generator("randn")
    values = {n: p.detach().clone() for n, p in coherent_parameters(trained).items()}
    with coherent_gains(trained, None):
        assert all(
            torch.equal(p.detach(), values[n]) for n, p in coherent_parameters(trained).items()
        )
    with pytest.raises(ValueError, match="non-negative"):
        with coherent_gains(trained, -0.1):
            pass


def small_model(coherent_noise_init: str | float | None) -> TTTSWIGAN:
    """Build a five-block Lightning module on Corse's 23x11 grid, as the tests above use."""
    return TTTSWIGAN(
        input_channels=NUM_INPUT_CHANNELS,
        output_channels=1,
        input_map_dims=[23, 11],
        encoder_channels=ENCODER_CHANNELS,
        decoder_channels=DECODER_CHANNELS,
        timestamps_dim=2,
        spatial_dropout=0.0,
        apply_center_block=True,
        z_dim=4,
        lr=1e-5,
        weight_decay=0.0,
        loss_fn="l1",
        optim=torch.optim.AdamW,
        normalization="batchnorm",
        coherent_noise_init=coherent_noise_init,
    )


def save_checkpoint(model: TTTSWIGAN, path: Path, drop: str | None = None) -> None:
    """Write what ``load_from_checkpoint`` reads, optionally without one parameter."""
    state = {k: v for k, v in model.state_dict().items() if k != drop}
    torch.save(
        {
            "state_dict": state,
            "hyper_parameters": dict(model.hparams),
            "pytorch-lightning_version": lightning.__version__,
        },
        path,
    )


def test_a_checkpoint_from_before_the_channel_loads_with_it(tmp_path: Path) -> None:
    """The 250-epoch weights hold no coherent gains, and the sweep needs them to exist."""
    torch.manual_seed(0)
    source = small_model(None)
    path = tmp_path / "base.ckpt"
    save_checkpoint(source, path)
    assert not checkpoint_carries_coherent_gains(path)

    loaded = load_with_coherent_channel(path, loss_fn="l1")
    assert len(coherent_gain_sites(loaded)) == 20
    for name, before in source.state_dict().items():
        assert torch.equal(before, loaded.state_dict()[name]), name


def test_a_checkpoint_missing_anything_else_is_refused(tmp_path: Path) -> None:
    """strict=False is only licensed for the gains; a real hole must not be re-initialised."""
    torch.manual_seed(0)
    source = small_model(None)
    dropped = next(k for k in source.state_dict() if k.startswith("generator.") and "conv" in k)
    path = tmp_path / "holed.ckpt"
    save_checkpoint(source, path, drop=dropped)
    with pytest.raises(ValueError, match="not coherent-noise gains"):
        load_with_coherent_channel(path, loss_fn="l1")


def test_a_checkpoint_with_its_own_channel_keeps_its_gains(tmp_path: Path) -> None:
    """The init is for gains the checkpoint lacks; trained ones load over it, strictly."""
    torch.manual_seed(0)
    source = small_model("randn")
    path = tmp_path / "finetuned.ckpt"
    save_checkpoint(source, path)
    assert checkpoint_carries_coherent_gains(path)

    loaded = load_with_coherent_channel(path, init=0.0, loss_fn="l1")
    trained = coherent_parameters(source)
    assert len(trained) == 20
    for name, param in coherent_parameters(loaded).items():
        assert torch.equal(param.detach(), trained[name].detach()), name
