"""Tests of the spatially coherent noise channel.

The generator's original stochastic channels are a latent vector and a field drawn per pixel
inside each block. That field is speckle: averaged over a region it goes to zero, so the
region-mean ensemble is narrow however large the gains get. Measured on the Corse 250-epoch
arm, the ensemble's region-mean spread divided by its pixel spread is 0.13-0.25, while the
error it is meant to span sits at 0.63-0.90 -- the truth's error is a coherent displacement of
the whole field. These tests pin that the new channel supplies exactly that missing mode, that
it costs nothing when it is off, and that a checkpoint written before it existed still loads.
"""

from __future__ import annotations

from pathlib import Path

import lightning
import pytest
import torch

from modules.generator.unet_generator import UNetGenerator
from swigan.engines.swigan_lit import NOISE_GAIN_ATTRIBUTES, TTTSWIGAN

CANVAS = 48
ENCODER_CHANNELS = [8, 16, 16, 16, 16]
DECODER_CHANNELS = [16, 16, 16, 16, 8]
NUM_INPUT_CHANNELS = 12
COHERENT = ("coherent_noise_weights1", "coherent_noise_weights2")


def corse_mask(batch_size: int = 1) -> torch.Tensor:
    """Corse's footprint on the canvas: an ellipse of 140 cells in its 23x11 bounding box."""
    yy, xx = torch.meshgrid(torch.arange(23), torch.arange(11), indexing="ij")
    radius = ((yy - 11) / 11.5) ** 2 + ((xx - 5) / 5.5) ** 2
    box = torch.zeros(23 * 11)
    box[radius.flatten().argsort()[:140]] = 1.0
    mask = torch.zeros(batch_size, 1, CANVAS, CANVAS)
    mask[..., 12:35, 18:29] = box.view(23, 11)
    return mask


def build_generator(coherent_noise_init: str | None) -> UNetGenerator:
    """Build a small five-block generator on the 48x48 canvas ``TTTSWIGAN.forward`` pads to."""
    return UNetGenerator(
        input_dim=NUM_INPUT_CHANNELS,
        output_dim=1,
        noise_dim=4,
        encoder_channels=ENCODER_CHANNELS,
        decoder_channels=DECODER_CHANNELS,
        dropout=0.0,
        normalization="batchnorm",
        apply_center_block=True,
        coherent_noise_init=coherent_noise_init,
    )


def set_gains(generator: UNetGenerator, white: float, coherent: float) -> None:
    """Drive the two noise families to fixed amplitudes so their effects can be told apart."""
    with torch.no_grad():
        for name, param in generator.named_parameters():
            attribute = name.rsplit(".", 1)[-1]
            if attribute in COHERENT:
                param.fill_(coherent)
            elif attribute in ("noise_weights1", "noise_weights2"):
                param.fill_(white)


def coherence(generator: UNetGenerator, members: int = 24) -> float:
    """Region-mean spread over pixel spread for an ensemble on one fixed input.

    1.0 means every member differs from the others by a displacement of the whole field, so
    the spread survives spatial averaging intact. 0 means independent per-cell noise, which
    the region mean averages away. This is the quantity ``utils/spread_shape.py`` reports as
    the region and pixel rows of the acceptance test.
    """
    generator.eval()
    mask = corse_mask(members)
    torch.manual_seed(0)
    inputs = torch.randn(1, NUM_INPUT_CHANNELS, CANVAS, CANVAS).expand(members, -1, -1, -1)
    with torch.no_grad():
        out = generator(inputs * mask, mask=mask)[:, 0]
    cells = mask[:, 0] > 0
    per_member = out[cells].view(members, -1)
    # Spread across members, of the region mean and of each cell.
    region_spread = per_member.mean(dim=1).std().item()
    pixel_spread = per_member.std(dim=0).mean().item()
    return region_spread / max(pixel_spread, 1e-12)


def test_the_channel_is_absent_unless_asked_for() -> None:
    """Default builds carry no new parameters, so existing checkpoints still load strictly."""
    without = build_generator(None)
    with_channel = build_generator("zeros")
    assert not [n for n, _ in without.named_parameters() if "coherent" in n]
    added = [n for n, _ in with_channel.named_parameters() if "coherent" in n]
    assert len(added) == 20, added
    assert [n for n, _ in with_channel.named_parameters() if "coherent" not in n] == [
        n for n, _ in without.named_parameters()
    ]


def test_zeros_init_leaves_the_output_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    """At the "zeros" init the channel must contribute exactly nothing.

    Replacing the draw with an enormous constant field proves it: if any of the output moved,
    the gains were not the only thing holding the channel shut, and a fine-tune would start by
    damaging the checkpoint it loaded.
    """
    generator = build_generator("zeros")
    generator.eval()
    mask = corse_mask(2)
    inputs = torch.randn(2, NUM_INPUT_CHANNELS, CANVAS, CANVAS) * mask

    torch.manual_seed(0)
    with torch.no_grad():
        quiet = generator(inputs, mask=mask)

    import modules.generator.unet_frame_decoder as decoder_module
    import modules.generator.unet_frame_encoder as encoder_module

    def enormous(gains, batch_size, device, dtype, mask=None):  # noqa: ANN001, ANN202
        if gains is None:
            return None
        # Draw and discard, so this consumes exactly the random numbers the real function
        # does and the per-pixel fields drawn after it are the same ones. Without that the
        # two runs diverge through the white channel and the test proves nothing.
        field = torch.randn((batch_size, gains.shape[0], 1, 1), device=device, dtype=dtype)
        field = torch.full_like(field, 1e4)
        return field if mask is None else field * mask.to(dtype)

    monkeypatch.setattr(encoder_module, "draw_coherent_field", enormous)
    monkeypatch.setattr(decoder_module, "draw_coherent_field", enormous)
    torch.manual_seed(0)
    with torch.no_grad():
        shouted = generator(inputs, mask=mask)

    assert torch.equal(quiet, shouted)


def test_the_coherent_channel_survives_spatial_averaging() -> None:
    """The point of the whole exercise: speckle averages away, the coherent channel does not.

    The numbers to beat are the measured ones. The shipped ensemble's region/pixel ratio is
    0.13-0.25 and the error it has to span is 0.63-0.90, so a channel worth adding has to sit
    up with the truth rather than down with the speckle.
    """
    torch.manual_seed(0)
    white_only = build_generator("zeros")
    set_gains(white_only, white=1.0, coherent=0.0)

    torch.manual_seed(0)
    coherent_only = build_generator("zeros")
    set_gains(coherent_only, white=0.0, coherent=1.0)

    speckle = coherence(white_only)
    whole_field = coherence(coherent_only)

    # Thresholds are the measured bands, not round numbers: the shipped ensemble sits at
    # 0.13-0.25 and the error it has to span at 0.63-0.90. The coherent channel measures 0.775
    # here and the per-pixel one 0.061, so both assertions clear their band with room.
    assert speckle < 0.30, f"the per-pixel channel should average away, got {speckle:.3f}"
    assert whole_field > 0.55, f"the coherent channel should survive, got {whole_field:.3f}"
    assert whole_field > 3 * speckle


def test_the_coherent_gains_join_the_scaled_param_group() -> None:
    """They are useless at the shared rate, for the reason the white gains are."""
    model = TTTSWIGAN(
        input_channels=NUM_INPUT_CHANNELS,
        output_channels=1,
        input_map_dims=[23, 11],
        encoder_channels=ENCODER_CHANNELS,
        decoder_channels=DECODER_CHANNELS,
        timestamps_dim=2,
        spatial_dropout=0.0,
        apply_center_block=True,
        z_dim=4,
        lr=3e-6,
        weight_decay=0.0,
        loss_fn="l1",
        optim=torch.optim.AdamW,
        normalization="batchnorm",
        max_epochs=50,
        min_lr=1e-7,
        coherent_noise_init="zeros",
        noise_weight_lr_scale=100.0,
    )
    (optimizer_generator, _), _ = model.configure_optimizers()
    conv_group, gain_group = optimizer_generator.param_groups
    assert gain_group["lr"] == 3e-6 * 100.0
    assert conv_group["lr"] == 3e-6

    expected = {
        id(p)
        for n, p in model.generator.named_parameters()
        if n.rsplit(".", 1)[-1] in NOISE_GAIN_ATTRIBUTES
    }
    assert len(expected) == 40, "20 per-pixel gains and 20 coherent ones"
    assert {id(p) for p in gain_group["params"]} == expected
    assert not ({id(p) for p in conv_group["params"]} & expected)


def test_a_checkpoint_from_before_the_channel_loads_into_a_model_with_it(tmp_path: Path) -> None:
    """Fine-tuning the 250-epoch arm means loading weights that have no coherent gains."""
    source = build_generator(None)
    path = tmp_path / "base.ckpt"
    torch.save(
        {
            "state_dict": {f"generator.{k}": v for k, v in source.state_dict().items()},
            "pytorch-lightning_version": lightning.__version__,
        },
        path,
    )
    saved = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    target = build_generator("zeros")
    prefixed = {f"generator.{k}": v for k, v in target.state_dict().items()}

    missing = [k for k in prefixed if k not in saved]
    assert missing, "the checkpoint should be missing something"
    assert all("coherent_noise_weights" in k for k in missing), missing
    assert len(missing) == 20

    incompatible = target.load_state_dict(
        {k[len("generator.") :]: v for k, v in saved.items()}, strict=False
    )
    assert not incompatible.unexpected_keys
    # Everything the checkpoint did carry arrived unchanged, and the gains are off.
    for name, before in source.state_dict().items():
        assert torch.equal(before, target.state_dict()[name]), name
    for name, param in target.named_parameters():
        if name.rsplit(".", 1)[-1] in COHERENT:
            assert torch.count_nonzero(param) == 0


def test_a_float_init_starts_the_channel_off_the_plateau() -> None:
    """Zero is a stationary point for the channel, so a fine-tune has to start away from it.

    At gain zero the coherent perturbation is independent of the differences the members
    already have, so its first-order contribution to a proper score's spread term averages to
    nothing -- measured, an energy fine-tune walked the gains from 0.0023 to 0.0032 rms over
    11 epochs, an exponent of 0.435, going nowhere. A float init puts every gain at a constant
    where the gradient is real and the score can push it either way.
    """
    generator = build_generator(0.1)
    gains = [
        param
        for name, param in generator.named_parameters()
        if name.rsplit(".", 1)[-1] in COHERENT
    ]
    assert len(gains) == 20
    assert all(torch.allclose(g, torch.full_like(g, 0.1)) for g in gains)
    # And it is a working amplitude, not just a number that threads through.
    assert coherence(generator) > coherence(build_generator(0.0))


def test_an_unknown_init_is_refused() -> None:
    """A typo must not silently leave the channel out or switched off."""
    with pytest.raises(NotImplementedError, match="coherent noise init"):
        build_generator("smallish")
