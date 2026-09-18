"""Tests of the mask-aware generator path.

The generator normalizes over space in 37 places -- 27 BatchNorms in its convolution blocks
and the global average pool inside each of its 10 scSE channel gates -- and every one of them
ran over the whole padded canvas. These tests pin the three properties that make
``generator_masked_norm`` trustworthy: that it is exactly the old generator when it is off,
that the statistics no longer depend on how much of the canvas the region covers, and that
with it on the generator cannot read the padding at all.
"""

from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch import nn

from modules.base_conv_blocks import MaskedSequential, single_conv_block
from modules.generator.unet_generator import UNetGenerator
from modules.masking import (
    INSTANCE_NORM_MIN_DIM,
    MaskedBatchNorm2d,
    MaskedGroupNorm2d,
    MaskedInstanceNorm2d,
    MaskedSpatialNorm2d,
    binary,
    default_num_groups,
    generator_mask_pyramid,
    masked_batch_statistics,
    masked_group_norm,
    masked_mean,
)
from modules.scse import SCSEModule

# The 48x48 canvas ``TTTSWIGAN.forward`` pads to, and the five-block encoder of
# config/train_swigan.yaml scaled down so the tests stay cheap.
CANVAS = 48
ENCODER_CHANNELS = [8, 16, 16, 16, 16]
DECODER_CHANNELS = [16, 16, 16, 16, 8]
NUM_INPUT_CHANNELS = 20


def corse_mask(batch_size: int = 4, cells: int = 140) -> torch.Tensor:
    """Corse's footprint on the canvas: ``cells`` of 2304, in its 23x11 bounding box."""
    yy, xx = torch.meshgrid(torch.arange(23), torch.arange(11), indexing="ij")
    radius = ((yy - 11) / 11.5) ** 2 + ((xx - 5) / 5.5) ** 2
    box = torch.zeros(23 * 11)
    box[radius.flatten().argsort()[:cells]] = 1.0
    mask = torch.zeros(batch_size, 1, CANVAS, CANVAS)
    mask[..., 12:35, 18:29] = box.view(23, 11)
    return mask


def running_mean(layer: MaskedBatchNorm2d) -> torch.Tensor:
    """Return the running mean, which ``track_running_stats`` guarantees is not None."""
    assert layer.running_mean is not None
    return layer.running_mean


def running_var(layer: MaskedBatchNorm2d) -> torch.Tensor:
    """Return the running variance, which ``track_running_stats`` guarantees is not None."""
    assert layer.running_var is not None
    return layer.running_var


def build_generator(seed: int = 0, normalization: str = "batchnorm") -> UNetGenerator:
    """Build the generator of the training config with small channel counts."""
    torch.manual_seed(seed)
    return UNetGenerator(
        input_dim=NUM_INPUT_CHANNELS,
        output_dim=1,
        noise_dim=32,
        encoder_channels=ENCODER_CHANNELS,
        decoder_channels=DECODER_CHANNELS,
        dropout=0.0,
        normalization=normalization,
        apply_center_block=True,
    )


def test_masked_batch_norm_is_batch_norm_under_a_full_mask() -> None:
    """Values, running statistics and the no-mask path all match ``nn.BatchNorm2d``."""
    masked, plain = MaskedBatchNorm2d(8), nn.BatchNorm2d(8)
    plain.load_state_dict(masked.state_dict())
    values = torch.randn(4, 8, 6, 6)
    full = torch.ones(4, 1, 6, 6)

    masked.train(), plain.train()
    torch.testing.assert_close(masked(values, full), plain(values))
    torch.testing.assert_close(masked.running_mean, plain.running_mean)
    torch.testing.assert_close(masked.running_var, plain.running_var)

    masked.eval(), plain.eval()
    torch.testing.assert_close(masked(values, full), plain(values))
    # Without a mask the layer is the one it replaces, whatever the mode.
    torch.testing.assert_close(masked(values), plain(values))


def test_masked_batch_norm_statistics_come_from_the_mask_only() -> None:
    """The statistics are those of the valid cells, and the padding comes out zero."""
    values = torch.randn(3, 4, 5, 5)
    mask = torch.zeros(3, 1, 5, 5)
    mask[..., 1:4, 1:4] = 1.0
    mean, var, count = masked_batch_statistics(values, mask)

    inside = values[..., 1:4, 1:4].transpose(0, 1).reshape(4, -1)
    torch.testing.assert_close(mean, inside.mean(dim=1))
    torch.testing.assert_close(var, inside.var(dim=1, unbiased=False))
    torch.testing.assert_close(count, torch.full((4,), float(3 * 3 * 3)))

    out = MaskedBatchNorm2d(4).train()(values, mask)
    assert torch.equal(out * (1 - mask), torch.zeros_like(out))


def test_masked_batch_norm_does_not_depend_on_the_coverage() -> None:
    """The same region on a 5x5 grid and on a 40x40 canvas normalizes to the same values.

    This is the property the whole change exists for: unmasked, the two differ by
    1/sqrt(coverage), so a batch mixing two regions normalizes each by the other's fill.
    """
    region = torch.randn(2, 3, 3, 3)

    tight = torch.randn(2, 3, 5, 5)
    tight_mask = torch.zeros(2, 1, 5, 5)
    tight[..., 1:4, 1:4], tight_mask[..., 1:4, 1:4] = region, 1.0

    canvas = torch.randn(2, 3, 40, 40) * 7.0 + 3.0  # arbitrary content in the padding
    canvas_mask = torch.zeros(2, 1, 40, 40)
    canvas[..., 20:23, 20:23], canvas_mask[..., 20:23, 20:23] = region, 1.0

    layer = MaskedBatchNorm2d(3).train()
    tight_out = layer(tight, tight_mask)[..., 1:4, 1:4]
    tight_stats = (running_mean(layer).clone(), running_var(layer).clone())
    layer = MaskedBatchNorm2d(3).train()
    canvas_out = layer(canvas, canvas_mask)[..., 20:23, 20:23]

    torch.testing.assert_close(tight_out, canvas_out)
    torch.testing.assert_close(tight_stats[0], running_mean(layer))
    torch.testing.assert_close(tight_stats[1], running_var(layer))

    plain = nn.functional.batch_norm(canvas, None, None, None, None, True, 0.0, 1e-5)[
        ..., 20:23, 20:23
    ]
    assert not torch.allclose(plain, canvas_out, atol=1e-3), "the unmasked norm must differ"


def test_masked_batch_norm_running_statistics_drive_the_eval_pass() -> None:
    """``eval`` reads the masked running statistics, which is what rollout scores with."""
    layer = MaskedBatchNorm2d(3, momentum=1.0).train()
    values, mask = torch.randn(4, 3, 8, 8), torch.zeros(4, 1, 8, 8)
    mask[..., 2:5, 2:5] = 1.0
    layer(values, mask)

    mean, var, count = masked_batch_statistics(values, mask)
    torch.testing.assert_close(running_mean(layer), mean)
    torch.testing.assert_close(running_var(layer), var * count / (count - 1.0))

    layer.eval()
    out = layer(values, mask)
    expected = (values - running_mean(layer).view(1, -1, 1, 1)) / torch.sqrt(
        running_var(layer).view(1, -1, 1, 1) + layer.eps
    )
    expected = expected * layer.weight.view(1, -1, 1, 1) + layer.bias.view(1, -1, 1, 1)
    torch.testing.assert_close(out, expected * mask)
    assert torch.equal(out * (1 - mask), torch.zeros_like(out))


def test_masked_reductions_never_produce_nan_on_an_empty_mask() -> None:
    """A region cut away entirely contributes zero, not NaN."""
    values = torch.randn(2, 3, 6, 6)
    empty = torch.zeros(2, 1, 6, 6)
    for out in (MaskedBatchNorm2d(3).train()(values, empty), masked_mean(values, empty)):
        assert torch.isfinite(out).all()
        assert torch.equal(out, torch.zeros_like(out))


def test_scse_channel_gate_pools_over_the_region_only() -> None:
    """The channel gate's pool is the coverage-weighted mean, and is the plain one when full."""
    module = SCSEModule(in_channels=16).eval()
    values, weight = torch.randn(2, 16, 8, 8), torch.rand(2, 1, 8, 8)

    torch.testing.assert_close(module(values), module(values, torch.ones(2, 1, 8, 8)))

    pooled = masked_mean(values, weight)[..., None, None]
    expected = pooled
    for layer in list(module.cSE)[1:]:
        expected = layer(expected)
    torch.testing.assert_close(module.channel_gate(values, weight), expected)
    assert not torch.allclose(module.channel_gate(values, weight), module.cSE(values))


def test_conv_block_keeps_the_state_dict_of_the_sequential_it_replaces() -> None:
    """``single_conv_block`` is still an ``nn.Sequential`` with the same keys."""
    block = single_conv_block(3, 4, 3, "batchnorm")
    assert isinstance(block, MaskedSequential | nn.Sequential)
    assert set(block.state_dict()) == set(
        nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4)).state_dict()
    )


@pytest.mark.parametrize("num_blocks", [3, 5])
def test_pyramid_levels_have_the_encoder_output_shapes(num_blocks: int) -> None:
    """Each pyramid level lands on the resolution of the encoder block it weights."""
    mask = corse_mask(batch_size=2)
    generator = UNetGenerator(
        input_dim=NUM_INPUT_CHANNELS,
        output_dim=1,
        noise_dim=32,
        encoder_channels=ENCODER_CHANNELS[:num_blocks],
        decoder_channels=DECODER_CHANNELS[-num_blocks:],
        dropout=0.0,
        normalization="batchnorm",
        apply_center_block=True,
    )
    features, levels = generator.encoder(torch.randn(2, NUM_INPUT_CHANNELS, CANVAS, CANVAS), mask)
    assert len(levels) == num_blocks + 1
    for feature, level in zip(features, levels, strict=True):
        assert feature.shape[-2:] == level.shape[-2:]


# Only "batchnorm" is exercised: on the 48x48 canvas the five-block encoder bottoms out at
# 1x1, and InstanceNorm rejects a single spatial element. That is not new -- the generator at
# HEAD raises the same ValueError -- so the generator's "instancenorm" option has never run on
# this canvas, and `config/train_swigan.yaml` sets "batchnorm".
@pytest.mark.parametrize("training", [True, False])
def test_the_generator_is_unchanged_when_no_mask_is_given(training: bool) -> None:
    """``generator_masked_norm: false`` is the generator rounds 1-12 trained, bit for bit."""
    generator = build_generator().train(training)
    maps, latent = torch.randn(4, NUM_INPUT_CHANNELS, CANVAS, CANVAS), torch.randn(4, 32)
    mask = corse_mask()

    # The twin swaps every MaskedBatchNorm2d back for the stock layer it subclasses, which
    # is the only substantive difference from the module tree at HEAD: MaskedSequential
    # without a mask steps through its children exactly as nn.Sequential does, and
    # SCSEModule without weights calls its own cSE.
    twin = copy.deepcopy(generator)
    for module in twin.modules():
        for name, child in list(module.named_children()):
            if isinstance(child, MaskedBatchNorm2d):
                plain = nn.BatchNorm2d(
                    child.num_features,
                    eps=child.eps,
                    momentum=child.momentum,
                    affine=child.affine,
                    track_running_stats=child.track_running_stats,
                )
                plain.load_state_dict(child.state_dict())
                setattr(module, name, plain.train(training))

    torch.manual_seed(1)
    unmasked = generator(maps, latent)
    torch.manual_seed(1)
    reference = twin(maps, latent)
    assert torch.equal(unmasked, reference), "the no-mask path must be bit-identical"
    assert all(
        torch.equal(a, b) for a, b in zip(generator.buffers(), twin.buffers(), strict=True)
    ), "and must leave the same running statistics behind"

    torch.manual_seed(1)
    masked = generator(maps * mask, latent, mask=mask)
    assert not torch.allclose(masked * mask, unmasked * mask), "the mask must change something"


@pytest.mark.parametrize("training", [True, False])
def test_the_masked_generator_cannot_read_the_padding(training: bool) -> None:
    """Whatever is written outside the mask, the output is the same, everywhere.

    Re-zeroing and the masked norm together make this exact: the statistics do not see the
    padding, and no convolution is ever handed a non-zero cell outside the region -- not the
    noise fields, not the transposed convolutions' biases, not the timestamp channels.
    """
    generator = build_generator().train(training)
    mask = corse_mask()
    latent = torch.randn(4, 32)
    maps = torch.randn(4, NUM_INPUT_CHANNELS, CANVAS, CANVAS) * mask + 0.7 * (1 - mask)
    junk = maps + torch.randn_like(maps) * 50.0 * (1 - mask)

    torch.manual_seed(1)
    out = generator(maps, latent, mask=mask)
    torch.manual_seed(1)
    out_junk = generator(junk, latent, mask=mask)
    assert torch.equal(out, out_junk)

    torch.manual_seed(1)
    leaky = generator(maps, latent)
    torch.manual_seed(1)
    leaky_junk = generator(junk, latent)
    assert not torch.allclose(leaky * mask, leaky_junk * mask), "unmasked, the padding leaks in"


def test_every_block_writes_zero_outside_its_level_of_the_pyramid() -> None:
    """The invariant the masked norm depends on, checked at every encoder and decoder block."""
    generator = build_generator().train()
    mask = corse_mask()
    levels = generator_mask_pyramid(mask, len(ENCODER_CHANNELS))
    outside: list[tuple[str, float]] = []

    def record(level: torch.Tensor, name: str) -> Callable[..., None]:
        padding = 1.0 - binary(level)

        def hook(module: nn.Module, inputs: Any, output: torch.Tensor) -> None:  # noqa: ARG001
            assert output.shape[-2:] == padding.shape[-2:]
            outside.append((name, float((output * padding).abs().max())))

        return hook

    handles = [
        block.register_forward_hook(record(levels[i], f"encoder.{i}"))
        for i, block in enumerate(generator.encoder.layers)
    ]
    handles += [
        block.register_forward_hook(record(levels[::-1][i + 1], f"decoder.{i}"))
        for i, block in enumerate(generator.decoder.blocks)
    ]
    output = generator(torch.randn(4, NUM_INPUT_CHANNELS, CANVAS, CANVAS) * mask, mask=mask)
    for handle in handles:
        handle.remove()

    assert len(outside) == len(ENCODER_CHANNELS) + len(DECODER_CHANNELS)
    assert all(value == 0.0 for _, value in outside), (
        f"leaked: {sorted(outside, key=lambda r: -r[1])[:3]}"
    )
    assert torch.equal(output * (1 - mask), torch.zeros_like(output))


def test_the_masked_generator_backpropagates() -> None:
    """Every parameter gets a finite gradient through the masked path."""
    generator = build_generator().train()
    mask = corse_mask()
    output = generator(torch.randn(4, NUM_INPUT_CHANNELS, CANVAS, CANVAS) * mask, mask=mask)
    output.pow(2).mean().backward()

    gradients = [p.grad for p in generator.parameters()]
    assert all(g is not None for g in gradients)
    assert all(torch.isfinite(g).all() for g in gradients if g is not None)


# --- masked GroupNorm, and the InstanceNorm fallback it backs ------------------------------
#
# ``nn.InstanceNorm2d`` cannot run on the generator's 1x1 bottleneck: its reduction is over
# (h, w) per sample and channel, so at one cell the variance is 0 and PyTorch raises rather
# than return zeros. The masked form does not raise -- it computes the statistics itself --
# and so returns exact zeros, which the UNet's skip connections hide. These tests pin the
# switch that avoids both: GroupNorm below ``INSTANCE_NORM_MIN_DIM``, InstanceNorm above.


def test_masked_group_norm_is_group_norm_under_a_full_mask() -> None:
    """With every cell valid this is ``nn.GroupNorm(affine=False)``."""
    values = torch.randn(4, 16, 6, 6)
    mask = torch.ones(4, 1, 6, 6)
    torch.testing.assert_close(
        masked_group_norm(mask, values, num_groups=4),
        nn.GroupNorm(4, 16, affine=False)(values),
    )


def test_masked_group_norm_statistics_come_from_the_mask_only() -> None:
    """Junk on the padding does not move the region's normalized values."""
    values = torch.randn(2, 16, 8, 8)
    mask = torch.zeros(2, 1, 8, 8)
    mask[..., 2:5, 2:5] = 1.0
    polluted = values + 50.0 * (1 - mask)

    torch.testing.assert_close(
        masked_group_norm(mask, values, 4), masked_group_norm(mask, polluted, 4)
    )


def masked_instance_norm_of(mask: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Run :class:`MaskedInstanceNorm2d` on ``values``, for contrast with the group form."""
    return MaskedInstanceNorm2d()(values, mask)


def test_masked_group_norm_survives_a_single_valid_cell() -> None:
    """One valid cell still yields a variance, because the group's channels supply it.

    This is the whole point of the fallback: :func:`masked_instance_norm` returns exact
    zeros here, on the 1x1 bottleneck and on any coarse level a small region reaches with a
    single cell.
    """
    for height, width in [(1, 1), (3, 3)]:
        values = torch.randn(4, 16, height, width)
        mask = torch.zeros(4, 1, height, width)
        mask[..., height // 2, width // 2] = 1.0

        grouped = masked_group_norm(mask, values, 4)
        instanced = masked_instance_norm_of(mask, values)

        assert grouped.abs().sum() > 0.0, f"group norm collapsed at {height}x{width}"
        assert instanced.abs().sum() == 0.0, f"instance norm was not degenerate at {height}"


@pytest.mark.parametrize(
    ("num_channels", "expected_groups", "expected_width"),
    [(8, 1, 8), (16, 1, 16), (64, 2, 32), (128, 4, 32), (256, 8, 32)],
)
def test_default_num_groups_targets_thirty_two_channels_per_group(
    num_channels: int, expected_groups: int, expected_width: int
) -> None:
    """The generator's channel counts all land on groups of at least two channels."""
    groups = default_num_groups(num_channels)
    assert groups == expected_groups
    assert num_channels // groups == expected_width


def test_masked_group_norm_refuses_a_group_of_one_channel() -> None:
    """A one-channel group is InstanceNorm again, so it is refused at construction."""
    with pytest.raises(ValueError, match="channel"):
        MaskedGroupNorm2d(4, num_groups=4)
    with pytest.raises(ValueError, match="does not divide"):
        MaskedGroupNorm2d(16, num_groups=5)


@pytest.mark.parametrize("size", [INSTANCE_NORM_MIN_DIM - 1, INSTANCE_NORM_MIN_DIM, 48])
def test_masked_spatial_norm_switches_on_the_map_size(size: int) -> None:
    """At or above the threshold it is the instance branch; below it, the group branch."""
    layer = MaskedSpatialNorm2d(16)
    values = torch.randn(2, 16, size, size)
    mask = torch.ones(2, 1, size, size)

    expected = (
        layer.group(values, mask)
        if size < INSTANCE_NORM_MIN_DIM
        else layer.instance(values, mask)
    )
    torch.testing.assert_close(layer(values, mask), expected)


def test_masked_spatial_norm_branches_split_the_generator_above_the_12_level() -> None:
    """The 48 and 24 levels take the instance branch; 12 down to 1 take the group branch.

    ``INSTANCE_NORM_MIN_DIM`` is one number, so the parametrized test above follows it
    wherever it moves. This one pins what the number was chosen for: which of the generator's
    levels end up on either side of the cut, and how many of its 27 spatial norms that is.
    A threshold of 9 would pull the five norms at 12x12 across, where a 140-cell region has
    15 valid cells left and InstanceNorm's divisor carries 20% relative error.
    """
    generator = build_generator(normalization="instancenorm")
    mask = corse_mask(batch_size=2)
    sizes: list[int] = []

    def record(module: nn.Module, args: tuple[Any, ...], output: torch.Tensor) -> None:
        sizes.append(int(args[0].shape[-1]))

    for module in generator.modules():
        if isinstance(module, MaskedSpatialNorm2d):
            module.register_forward_hook(record)

    generator(torch.randn(2, NUM_INPUT_CHANNELS, CANVAS, CANVAS) * mask, mask=mask)

    per_level = Counter(sizes)
    assert dict(per_level) == {48: 4, 24: 5, 12: 5, 6: 5, 3: 5, 1: 3}

    instance = sum(count for size, count in per_level.items() if size >= INSTANCE_NORM_MIN_DIM)
    assert (instance, sum(per_level.values()) - instance) == (9, 18)
    assert min(size for size in per_level if size >= INSTANCE_NORM_MIN_DIM) == 24


def test_masked_spatial_norm_adds_no_state_dict_keys() -> None:
    """Both branches are parameter-free, so a block's keys are unchanged."""
    assert MaskedSpatialNorm2d(16).state_dict() == {}
    block = single_conv_block(3, 16, 3, "instancenorm")
    assert set(block.state_dict()) == set(
        nn.Sequential(nn.Conv2d(3, 16, 3, padding=1)).state_dict()
    )


def test_masked_spatial_norm_treats_every_sample_in_a_batch_alike() -> None:
    """The branch follows the map's shape, so a differing cell count cannot split a batch.

    Two regions of the same size can reach a coarse level with different numbers of valid
    cells purely through where they land on the stride grid. A count-driven fallback would
    normalize them differently; this one does not.
    """
    layer = MaskedSpatialNorm2d(16)
    values = torch.randn(4, 16, 3, 3)
    mask = torch.zeros(4, 1, 3, 3)
    mask[:2, :, 1, 1] = 1.0  # one valid cell
    mask[2:, :, 1, 1:3] = 1.0  # two valid cells

    output = layer(values, mask)
    for sample in range(4):
        assert output[sample].abs().sum() > 0.0, f"sample {sample} collapsed to zero"


@pytest.mark.parametrize("normalization", ["instancenorm", "groupnorm"])
@pytest.mark.parametrize("masked", [True, False])
def test_the_generator_runs_at_the_bottleneck_with_spatial_normalization(
    normalization: str, masked: bool
) -> None:
    """The 1x1 bottleneck no longer raises, and no longer collapses to zero either.

    Plain ``nn.InstanceNorm2d`` raises ``Expected more than 1 spatial element`` here, and
    the masked form used to return exact zeros for the whole center block.
    """
    generator = build_generator(normalization=normalization).eval()
    mask = corse_mask()
    inputs = torch.randn(4, NUM_INPUT_CHANNELS, CANVAS, CANVAS) * mask

    outputs, _ = generator.encoder(inputs, mask if masked else None)

    assert outputs[-1].shape[-2:] == (1, 1)
    assert outputs[-1].abs().sum() > 0.0, "the center block collapsed to zero"
    assert torch.isfinite(generator(inputs, mask=mask if masked else None)).all()


@pytest.mark.parametrize("normalization", ["instancenorm", "groupnorm"])
def test_the_spatially_normalized_generator_cannot_read_the_padding(
    normalization: str,
) -> None:
    """Filling the padding with junk leaves the masked output untouched.

    The same property :func:`test_the_masked_generator_cannot_read_the_padding` pins for
    BatchNorm, for the two spatial norms. The seed is reset before each call because the
    encoder blocks draw a fresh noise field on every forward, in eval as in train.
    """
    generator = build_generator(normalization=normalization).eval()
    mask = corse_mask()
    latent = torch.randn(4, 32)
    maps = torch.randn(4, NUM_INPUT_CHANNELS, CANVAS, CANVAS) * mask + 0.7 * (1 - mask)
    junk = maps + torch.randn_like(maps) * 50.0 * (1 - mask)

    torch.manual_seed(1)
    out = generator(maps, latent, mask=mask)
    torch.manual_seed(1)
    out_junk = generator(junk, latent, mask=mask)
    assert torch.equal(out, out_junk)


def test_masked_group_norm_does_not_depend_on_the_coverage() -> None:
    """The same region on a small map and a large one normalizes to the same values."""
    torch.manual_seed(0)
    region = torch.randn(1, 16, 5, 5)

    small_mask = torch.ones(1, 1, 5, 5)
    small = masked_group_norm(small_mask, region, 4)

    large = torch.zeros(1, 16, 24, 24)
    large[..., 7:12, 7:12] = region
    large_mask = torch.zeros(1, 1, 24, 24)
    large_mask[..., 7:12, 7:12] = 1.0
    normalized = masked_group_norm(large_mask, large, 4)

    torch.testing.assert_close(small, normalized[..., 7:12, 7:12])
