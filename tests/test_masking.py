"""Tests of the mask-aware critic path.

The critic path reduces over space in several places, and each reduction is weighted by
one coverage pyramid derived from the region mask. These tests pin the identities that
make that pyramid trustworthy, the geometry that keeps it aligned with the trunk on any
grid, and the end-to-end training steps on Corse's 23x11 grid, on the 48x48 canvas the
generator pads to, and on the 36x44 Grand Est grid.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from modules.diff_augment import DiffAugment
from modules.discriminator.base_discriminator import BaseDiscriminator
from modules.discriminator.frame_discriminator import FrameDiscriminator
from modules.discriminator.patch_gan_discriminator import PatchGANDiscriminator
from modules.masking import (
    diff_augment_with_mask,
    mask_pyramid,
    masked_instance_norm,
    masked_mean,
    region_extent,
)
from swigan.engines.swigan_lit import DIFF_AUGMENT_POLICY, TTTSWIGAN

NUM_INPUT_CHANNELS = 11 + 8  # eleven covariates and eight months of SWI history


def ellipse_mask(height: int, width: int, fill: float = 0.55) -> torch.Tensor:
    """Build an elliptical region covering ``fill`` of the grid, shape (1, 1, H, W)."""
    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    scale = math.sqrt(4 * fill / math.pi)
    cy, cx = (height - 1) / 2, (width - 1) / 2
    ry, rx = height / 2 * scale, width / 2 * scale
    return (((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0).float()[None, None]


def make_batch(height: int, width: int, batch_size: int = 4) -> dict[str, torch.Tensor]:
    """Build a synthetic batch shaped like one from ``SWIDataset``."""
    mask = ellipse_mask(height, width).expand(batch_size, -1, -1, -1).contiguous()
    return {
        "input_maps": torch.randn(batch_size, NUM_INPUT_CHANNELS, height, width) * mask,
        "target_maps": torch.randn(batch_size, 1, height, width) * mask,
        "timestamps": torch.randint(0, 12, (batch_size,)),
        "mask": mask,
    }


def build_model(
    height: int,
    width: int,
    aggregation: str = "mean",
    normalization: str | None = "instancenorm",
    canvas: bool = False,
) -> TTTSWIGAN:
    """Build a small ``TTTSWIGAN`` with the training config's loss settings."""
    model = TTTSWIGAN(
        input_channels=NUM_INPUT_CHANNELS,
        output_channels=1,
        input_map_dims=[height, width],
        encoder_channels=[8, 16, 32, 32, 32],
        decoder_channels=[32, 32, 32, 16, 8],
        timestamps_dim=5,
        spatial_dropout=0.0,
        apply_center_block=True,
        z_dim=8,
        lr=1e-5,
        weight_decay=1e-4,
        loss_fn="l1",
        optim=torch.optim.AdamW,
        normalization="batchnorm",
        patch_critic_loss="wasserstein",
        patch_aggregation=aggregation,
        gradient_penalty_reduction="mean",
        num_critic_iterations_per_epoch=1,
        lambda_penalty=1.0,
        image_distance_weight=100.0,
        feature_matching_weight=10.0,
        max_epochs=1,
        critic_normalization=normalization,
        critic_on_padded_canvas=canvas,
    )
    model.input_statistics = {
        "targets_mean": np.zeros((1, 1, 1, 1), dtype=np.float32),
        "targets_std": np.ones((1, 1, 1, 1), dtype=np.float32),
    }
    return model


def test_masked_reductions_reduce_to_unmasked_ones_under_a_full_mask() -> None:
    """With every cell valid the masked helpers are the plain ones."""
    x = torch.randn(3, 5, 9, 7)
    ones = torch.ones(3, 1, 9, 7)
    assert torch.allclose(masked_instance_norm(ones, x), F.instance_norm(x), atol=1e-5)
    assert torch.allclose(masked_mean(x, ones), x.mean((-2, -1)), atol=1e-6)


def test_masked_mean_is_the_weighted_mean_and_never_nan() -> None:
    """Fractional weights rank cells by coverage; an empty mask gives 0, not NaN."""
    x = torch.randn(3, 5, 9, 7)
    w = torch.rand(3, 1, 9, 7)
    expected = (x * w).sum((-2, -1)) / w.sum((-2, -1))
    assert torch.allclose(masked_mean(x, w), expected, atol=1e-5)
    assert torch.equal(masked_mean(x, torch.zeros_like(w)), torch.zeros(3, 5))


def test_masked_instance_norm_statistics_come_from_the_mask_only() -> None:
    """Zero mean over the valid cells and exactly zero outside them."""
    x = torch.randn(3, 5, 9, 7)
    m = (torch.rand(3, 1, 9, 7) > 0.5).float()
    out = masked_instance_norm(m, x)
    assert float((out * (1 - m)).abs().max()) == 0.0
    inside_mean = (out * m).sum((-2, -1)) / m.sum((-2, -1))
    assert float(inside_mean.abs().max()) < 1e-5


@pytest.mark.parametrize(
    ("height", "width", "trunk", "tiles"),
    [
        (23, 11, (3, 2), (2, 1)),
        (36, 44, (5, 6), (4, 5)),
        (48, 48, (6, 6), (5, 5)),
        (64, 64, (8, 8), (7, 7)),
    ],
)
def test_pyramid_levels_have_the_trunk_output_shapes(
    height: int, width: int, trunk: tuple[int, int], tiles: tuple[int, int]
) -> None:
    """Each coverage level is exactly the shape of the block it weights, on any grid."""
    critic = BaseDiscriminator(1, [8, 16, 32], mbd_output_channels=4, mask_channel=True).eval()
    mask = ellipse_mask(height, width).expand(2, -1, -1, -1)
    levels = mask_pyramid(mask, 3)
    with torch.no_grad():
        out, features = critic(torch.randn(2, 1, height, width) * mask, mask)
        patch, _ = PatchGANDiscriminator(36, grid_shape=tiles).eval()(out, weight=levels[-1])
        frame, _ = FrameDiscriminator(36).eval()(out, weight=levels[-1])
    assert tuple(out.shape[-2:]) == trunk
    assert all(f.shape[-2:] == lv.shape[-2:] for f, lv in zip(features, levels[1:4], strict=True))
    assert tuple(levels[-1].shape[-2:]) == tiles
    assert tuple(patch.shape[-2:]) == tiles
    assert frame.shape == (2, 1)


def test_diff_augment_carries_the_mask_with_the_data() -> None:
    """After translation and cutout the maps still vanish exactly outside the mask."""
    torch.manual_seed(0)
    mask = ellipse_mask(23, 11).expand(16, -1, -1, -1)
    maps = torch.randn(16, 1, 23, 11) * mask
    aug_maps, aug_mask = diff_augment_with_mask(maps, mask, DIFF_AUGMENT_POLICY)
    assert float((aug_maps * (1 - aug_mask)).abs().max()) == 0.0
    assert set(aug_mask.unique().tolist()) <= {0.0, 1.0}
    _, other_mask = diff_augment_with_mask(maps, mask, DIFF_AUGMENT_POLICY)
    assert not torch.equal(aug_mask, other_mask), "two calls draw two augmentations"


def test_pixel_loss_does_not_dilute_with_the_fill_fraction() -> None:
    """The same region scores the same on its native grid and on a 48x48 canvas."""
    model = build_model(23, 11)
    mask = ellipse_mask(23, 11).expand(2, -1, -1, -1)
    a = torch.randn(2, 1, 23, 11) * mask
    b = torch.randn(2, 1, 23, 11) * mask
    pa, pb, pm = (torch.zeros(2, 1, 48, 48) for _ in range(3))
    pa[..., 8:31, 8:19], pb[..., 8:31, 8:19], pm[..., 8:31, 8:19] = a, b, mask
    native = float(model.pixel_distance(a, b, mask))
    canvas = float(model.pixel_distance(pa, pb, pm))
    assert abs(native - canvas) < 1e-6
    assert abs(float(F.l1_loss(a, b)) - float(F.l1_loss(pa, pb))) > 0.1, "the plain L1 does dilute"


def test_the_canvas_scores_the_windows_the_native_grid_drops() -> None:
    """Corse on its 23x11 grid is 2 tiles; on the generator's 48x48 canvas it is 20."""
    mask = ellipse_mask(23, 11)
    native_tiles = mask_pyramid(mask, 3)[-1]
    model = build_model(23, 11, canvas=True)
    _, canvas_mask = model.to_critic_canvas(torch.zeros(1, 1, 23, 11), mask)
    canvas_tiles = mask_pyramid(canvas_mask, 3)[-1]
    assert int((native_tiles > 0).sum()) == 2
    assert int((canvas_tiles > 0).sum()) > 2
    assert tuple(canvas_mask.shape[-2:]) == (48, 48)


@pytest.mark.parametrize(
    ("height", "width", "aggregation", "normalization", "canvas"),
    [
        (23, 11, "mean", "instancenorm", False),
        (23, 11, "mean", "instancenorm", True),
        (23, 11, "learned", None, True),
        (36, 44, "mean", "instancenorm", False),
    ],
)
def test_training_steps_run_end_to_end(
    height: int, width: int, aggregation: str, normalization: str | None, canvas: bool
) -> None:
    """Critic and generator steps produce finite losses and gradients on every parameter."""
    torch.manual_seed(0)
    model = build_model(height, width, aggregation, normalization, canvas)
    batch = make_batch(height, width)
    z = torch.randn(4, 8)

    model.train()
    critic_out = model.critic_step(batch, z)
    critic_out["total_loss_critic"].backward()
    critic_params = [
        p
        for head in (model.base_critic, model.patch_critic, model.frame_critic)
        for p in head.parameters()
    ]
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in critic_params)
    assert math.isfinite(float(critic_out["gradient_penalty"]))
    model.zero_grad(set_to_none=True)

    generator_out = model.generator_step(batch, z)
    generator_out["loss_generator"].backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.generator.parameters()
    )
    for key in ("patch_term", "frame_term", "pixel_distance_loss", "feature_loss"):
        assert math.isfinite(float(generator_out[key]))

    model.eval()
    with torch.no_grad():
        model.critic_step(batch, z)
        model.generator_step(batch, z)
        prediction = model(batch["input_maps"], batch["timestamps"], batch["mask"], z)
    assert prediction.shape == (4, 1, height, width)
    assert float((prediction * (1 - batch["mask"])).abs().max()) == 0.0


def place_on_canvas(mask: torch.Tensor, size: int = 48) -> torch.Tensor:
    """Put a (1, 1, H, W) region on a ``size`` x ``size`` canvas, centred as forward pads."""
    height, width = mask.shape[-2:]
    canvas = torch.zeros(1, 1, size, size)
    top, left = (size - height) // 2, (size - width) // 2
    canvas[..., top : top + height, left : left + width] = mask
    return canvas


def cutout_losses(mask: torch.Tensor, on_region: bool, draws: int = 2000) -> torch.Tensor:
    """Fraction of the region removed by the cutout, over ``draws`` independent draws."""
    torch.manual_seed(0)
    batch = mask.expand(draws, -1, -1, -1).contiguous()
    _, augmented = diff_augment_with_mask(batch.clone(), batch, "cutout", on_region)
    return 1.0 - augmented.flatten(1).sum(1) / mask.sum()


def test_region_extent_is_the_bounding_box_of_each_sample() -> None:
    """The extent is (top, left, height, width), per sample, and survives the canvas."""
    # The ellipse does not touch the corners of its grid, so its box is smaller than the
    # grid -- which is the point: the extent follows the region, not the raster.
    mask = ellipse_mask(23, 11)
    assert region_extent(mask).tolist() == [[2, 1, 19, 9]]
    # Placed on the canvas at (12, 18), the same box moves by the offset and keeps its size.
    assert region_extent(place_on_canvas(mask)).tolist() == [[14, 19, 19, 9]]

    batch = torch.cat([place_on_canvas(mask), torch.zeros(1, 1, 48, 48)])
    assert region_extent(batch).tolist() == [[14, 19, 19, 9], [0, 0, 0, 0]], "empty mask is zero"


def test_the_canvas_does_not_change_a_region_calibrated_cutout() -> None:
    """The whole point: the same island loses the same distribution on either grid."""
    mask = ellipse_mask(23, 11)
    native = cutout_losses(mask, on_region=True)
    canvas = cutout_losses(place_on_canvas(mask), on_region=True)
    assert abs(float(native.mean()) - float(canvas.mean())) < 0.01
    assert abs(float(native.max()) - float(canvas.max())) < 0.01


def test_the_stock_cutout_guts_the_region_once_it_sits_on_a_canvas() -> None:
    """The defect being fixed, pinned so it cannot come back unnoticed."""
    mask = ellipse_mask(23, 11)
    stock_native = cutout_losses(mask, on_region=False)
    stock_canvas = cutout_losses(place_on_canvas(mask), on_region=False)
    fixed_canvas = cutout_losses(place_on_canvas(mask), on_region=True)

    # On the native grid the map is the region, so a 30% box can never take much more.
    assert float(stock_native.max()) < 0.25
    # On the canvas the same policy cuts 14x14 out of a 23x11 island.
    assert float(stock_canvas.max()) > 0.6
    assert float((stock_canvas > 0.5).float().mean()) > 0.02
    # Calibrated on the region, the canvas draw is back under the native ceiling.
    assert float(fixed_canvas.max()) < 0.25
    assert float((fixed_canvas > 0.5).float().mean()) == 0.0


def test_region_calibration_keeps_the_cutout_hitting_the_region() -> None:
    """Sizing alone is not the fix: the centre has to be drawn over the region too."""
    mask = place_on_canvas(ellipse_mask(23, 11))
    stock = cutout_losses(mask, on_region=False)
    fixed = cutout_losses(mask, on_region=True)
    # The stock policy misses the island entirely on most draws; the calibrated one rarely does.
    assert float((stock == 0).float().mean()) > 0.5
    assert float((fixed == 0).float().mean()) < 0.2
    # Yet it is not a heavier augmentation on average -- it is the same one, better placed.
    assert abs(float(stock.mean()) - float(fixed.mean())) < 0.05


def test_region_calibrated_translation_scales_with_the_region() -> None:
    """A shift drawn on the region cannot exceed the region's own 20%, on any canvas."""
    torch.manual_seed(0)
    mask = place_on_canvas(ellipse_mask(23, 11)).expand(512, -1, -1, -1).contiguous()
    _, augmented = diff_augment_with_mask(mask.clone(), mask, "translation", True)
    extents = region_extent(augmented)
    # The island is never clipped by the canvas edge, so its box keeps its 19x9 size, and
    # the top-left corner moves by at most int(19 * 0.2 + 0.5) = 4 rows, int(9 * 0.2 + 0.5)
    # = 2 columns -- the region's own 20%, not the canvas's int(48 * 0.2 + 0.5) = 10.
    assert set(extents[:, 2].tolist()) == {19}
    assert set(extents[:, 3].tolist()) == {9}
    assert int((extents[:, 0] - 14).abs().max()) == 4
    assert int((extents[:, 1] - 19).abs().max()) == 2


def test_the_stock_policy_is_untouched_by_the_switch() -> None:
    """`on_region=False` reproduces the augmentation rounds 1-12 trained on, bit for bit."""
    mask = ellipse_mask(36, 44).expand(8, -1, -1, -1).contiguous()
    maps = torch.randn(8, 1, 36, 44) * mask

    torch.manual_seed(7)
    expected = DiffAugment(torch.cat([maps, mask], dim=1).contiguous(), policy=DIFF_AUGMENT_POLICY)
    torch.manual_seed(7)
    got_maps, got_mask = diff_augment_with_mask(maps, mask, DIFF_AUGMENT_POLICY, False)
    assert torch.equal(got_maps, expected[:, :-1])
    assert torch.equal(got_mask, expected[:, -1:])
