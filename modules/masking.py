"""Mask-aware reductions for training on a zero-padded canvas.

Every map this project trains on is a rectangular raster of which only the region's cells
carry data: 55% of the Corse and Grand Est grids, and 3-6% once a region is placed on a
common canvas. The critic path reduces over space in several places -- the InstanceNorm
statistics, the patch critic's tile mean, MiniBatch Discrimination, the gradient penalty
norm, the pixel and feature-matching losses -- and each unmasked reduction treats the
padding as data. The helpers here make every one of those reductions mask-weighted, driven
by one pyramid of coverage weights computed with the critic trunk's own stride pattern.

The generator reduces over space too. Its 37 BatchNorms take mean and variance over the
whole padded canvas, so on Corse -- 140 real cells out of 48x48 -- 94% of every statistic is
padding. Where the padding sits near zero the variance is diluted by the coverage fraction,
and dividing by that smaller sigma makes the region's normalized activations 1/sqrt(c) times
*too large*: measured on this generator at Corse's 6.1%, 3.8x at the first encoder block,
2.5x at its second convolution, decaying with depth to a median 1.1x over the 24 spatial
norms, because the deeper padding carries a bias-filled plateau whose own between-group
variance partly offsets the dilution.

The size of that factor is the smaller half of the problem. It *depends on the coverage*, so
a batch mixing Grand Est with Corse normalizes the same map differently depending on who it
is batched with. :class:`MaskedBatchNorm2d` and :func:`generator_mask_pyramid` remove that
dependence; the ten scSE gates need their global average pool masked with the same weights,
and each block re-zeroes its output so the padding plateau a norm's affine term leaves
behind never feeds the next convolution. The three are one change, not three: re-zeroing
alone leaves the statistics diluted, and a masked norm alone writes an out-of-range constant
into the padding that the next 3x3 conv immediately bleeds back across the coast.

Once every reduction is masked, a zero-padding convolution at the image border is the same
operation as an explicit zero canvas, so the canvas size becomes a batching and compute
choice rather than a modelling one.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa
from torch import nn

from modules.diff_augment import DiffAugment


def mask_pyramid(mask: torch.Tensor, num_trunk_blocks: int = 3) -> list[torch.Tensor]:
    """Coverage weights of the region at every resolution of the critic.

    Level 0 is the mask itself. Levels ``1..num_trunk_blocks`` follow the trunk's 3x3
    stride-2 pad-1 convolutions with an average pool of the same geometry, so each cell
    holds the fraction of real pixels under it. The last level follows the heads' 2x2
    stride-1 convolution and is the tile grid the patch and frame critics score.

    Averaging rather than max-pooling keeps a graded weight: a coastal tile that is 20% land
    counts a fifth of an inland one. Padded cells at the canvas border count as empty
    (``count_include_pad=True``), so a cell hanging off the edge covers less. ``weight > 0``
    is the binary "any data under this cell" mask the normalization layers use.

    Args:
    ----
        mask: Region mask of shape (B, 1, H, W) with values in {0, 1}.
        num_trunk_blocks: Number of stride-2 blocks in ``BaseDiscriminator``.

    Returns:
    -------
        ``num_trunk_blocks + 2`` tensors: the mask, one level per trunk block, and the tile
        grid.

    """
    levels = [mask]
    weight = mask
    for _ in range(num_trunk_blocks):
        weight = F.avg_pool2d(weight, kernel_size=3, stride=2, padding=1)
        levels.append(weight)
    levels.append(F.avg_pool2d(weight, kernel_size=2, stride=1))
    return levels


def generator_mask_pyramid(mask: torch.Tensor, num_blocks: int) -> list[torch.Tensor]:
    """Coverage weights of the region at every resolution of the UNet generator.

    Level 0 is the mask itself, the resolution the first encoder block and the last decoder
    block work at. Level ``k`` follows the encoder's ``k`` downsampling convolutions -- 2x2,
    stride 2, no padding -- with an average pool of the same geometry, so each cell holds the
    fraction of real pixels under it and an odd row or column is dropped exactly where the
    strided convolution drops it. Level ``num_blocks`` is the center block's resolution.

    The decoder walks the same list backwards: its ``ConvTranspose2d(2, stride=2)`` doubles
    the resolution to the one its skip connection arrives at, so decoder block ``i`` writes
    its output under ``levels[num_blocks - 1 - i]``. Sharing one pyramid is what keeps an
    encoder level and the decoder level it feeds normalized over the same cells.

    Args:
    ----
        mask: Region mask of shape (B, 1, H, W) with values in {0, 1}.
        num_blocks: Number of downsampling blocks in ``UNetFrameEncoder``.

    Returns:
    -------
        ``num_blocks + 1`` tensors, finest first.

    """
    levels = [mask]
    weight = mask
    for _ in range(num_blocks):
        weight = F.avg_pool2d(weight, kernel_size=2, stride=2)
        levels.append(weight)
    return levels


def masked_mean(
    values: torch.Tensor,
    weight: torch.Tensor,
    dims: tuple[int, ...] = (-2, -1),
    eps: float = 1e-6,
) -> torch.Tensor:
    """Weighted mean of ``values`` over ``dims``.

    ``weight`` is broadcast against ``values`` (typically (B, 1, h, w) against (B, C, h, w)),
    so reducing over the channel dimension counts every channel. A sample whose weights sum
    to zero -- a region entirely removed by the cutout augmentation -- contributes 0 rather
    than NaN.

    Args:
    ----
        values: The tensor to reduce.
        weight: Non-negative weights, broadcastable to ``values``.
        dims: Dimensions to reduce over.
        eps: Floor on the weight total.

    Returns:
    -------
        ``values`` reduced over ``dims``.

    """
    weight = weight.to(values.dtype).expand_as(values)
    return (values * weight).sum(dims) / weight.sum(dims).clamp_min(eps)


def masked_instance_norm(
    mask: torch.Tensor, values: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    """InstanceNorm2d whose statistics come from the valid cells only.

    Mean and biased variance are taken per sample and channel over the cells where ``mask``
    is non-zero, exactly as ``nn.InstanceNorm2d`` takes them over the whole map, and the
    output is zeroed outside the mask so that padding activations never leak back into the
    next convolution. With an all-ones mask this is ``F.instance_norm``.

    Args:
    ----
        mask: Binary mask of shape (B, 1, h, w) at the resolution of ``values``.
        values: Activations of shape (B, C, h, w).
        eps: Numerical floor on the variance, as in ``nn.InstanceNorm2d``.

    Returns:
    -------
        The normalized activations, zero outside the mask.

    """
    mask = mask.to(values.dtype)
    count = mask.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
    mean = (values * mask).sum(dim=(-2, -1), keepdim=True) / count
    var = (((values - mean) ** 2) * mask).sum(dim=(-2, -1), keepdim=True) / count
    return (values - mean) / torch.sqrt(var + eps) * mask


class MaskedInstanceNorm2d(nn.Module):
    """``nn.InstanceNorm2d`` (no affine, no running stats) restricted to the valid cells."""

    accepts_mask = True

    def __init__(self, eps: float = 1e-5) -> None:
        """Initialize the layer.

        Args:
        ----
            eps: Numerical floor on the variance.

        """
        super().__init__()
        self.eps = eps

    def forward(self, values: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Normalize ``values``; without a mask this is plain instance normalization."""
        if mask is None:
            return F.instance_norm(values, eps=self.eps)
        return masked_instance_norm(mask, values, eps=self.eps)


#: Smallest map dimension at which InstanceNorm's spatial reduction is trusted.
#:
#: The generator's five stride-2 downsamples take the 48x48 canvas to 24, 12, 6, 3 and 1, so
#: this threshold chooses how far down that pyramid a per-(sample, channel) reduction is
#: still worth having. 17 admits 48 and 24 and stops above 12, which puts 9 of the
#: generator's 27 spatial norms on the instance branch and 18 on the group branch.
#:
#: InstanceNorm estimates sigma from the valid cells of one channel of one sample, so its
#: sample size is the region's cell count at that level and nothing else. Bootstrapping that
#: estimate over the valid cells, on this generator's activations at initialization, gives
#: its relative standard error. The Corse column is that footprint on the canvas carrying
#: Grand Est fields, so it varies the geometry and holds the data fixed; and because training
#: makes activations more spatially correlated, which lowers the effective sample size, these
#: are a floor rather than an estimate:
#:
#:     level    map     Grand Est (896 cells)       Corse (140 cells)
#:                        n   instance   group      n   instance   group
#:       0     48x48     896     3.1%     0.4%     140     7.2%     1.1%
#:       1     24x24     249     5.2%     1.0%      42    12.2%     2.5%
#:       2     12x12      72     9.0%     1.8%      15    20.0%     4.3%
#:       3      6x6       24    15.6%     3.2%       8    27.9%     6.1%
#:       4      3x3        8    25.7%     5.3%       3    53.4%     8.9%
#:       5      1x1        1   degenerate 12.1%      1   degenerate 12.4%
#:
#: The cut sits above 12x12 because that is the level where the smaller region crosses 20%,
#: and because the norms are affine-free: there is no gamma to absorb a per-sample error in
#: the divisor, so it passes into the next convolution as multiplicative noise. At a single
#: cell InstanceNorm is degenerate -- variance 0, so every activation normalizes to exactly
#: 0. Below this dimension :class:`MaskedSpatialNorm2d` reduces over a group of channels as
#: well, which multiplies the sample size by the group width and stays defined down to 1x1.
#:
#: This started at 33, which came from the critic: a 3x3 stride-2 pad-1 convolution takes a
#: dimension to ``ceil(d / 2)``, so the trunk's three blocks take 33 -> 17 -> 9 -> 5 while
#: 32 -> 16 -> 8 -> 4, and 33 is the smallest input still reaching the 5 rows the frame
#: head's old fixed tail needed. That is a constraint on the critic's geometry rather than a
#: statement about how many cells a variance needs, and it left level 1 on the group branch
#: at a cost the table above does not show: the injected noise is added *after* the norm, so
#: the norm decides what a noise gain means. Group normalization leaves the per-channel
#: post-norm std spanning 0.62 to 1.17 (p10-p90, measured at 24x24), so one gain is 1.8x
#: louder relative to signal on a quiet channel than on a loud one; the instance branch pins
#: every channel at exactly 1 and makes the gain a signal-to-noise ratio.
INSTANCE_NORM_MIN_DIM = 17

#: Channels per group :func:`default_num_groups` aims for, the usual GroupNorm convention.
DEFAULT_CHANNELS_PER_GROUP = 32


def default_num_groups(
    num_channels: int, channels_per_group: int = DEFAULT_CHANNELS_PER_GROUP
) -> int:
    """Return the group count that puts ``channels_per_group`` channels in each group.

    When ``channels_per_group`` does not divide ``num_channels`` the widest divisor below it
    is used instead, because a group's width is what the variance estimate draws on: the
    generator's 64/128/256-channel blocks all land on groups of exactly 32.

    Args:
    ----
        num_channels: Number of channels the norm will see.
        channels_per_group: Preferred group width.

    Returns:
    -------
        A group count that divides ``num_channels``.

    """
    for group_width in range(min(channels_per_group, num_channels), 1, -1):
        if num_channels % group_width == 0:
            return num_channels // group_width
    return 1


def masked_group_norm(
    mask: torch.Tensor, values: torch.Tensor, num_groups: int, eps: float = 1e-5
) -> torch.Tensor:
    """GroupNorm whose statistics come from the valid cells only.

    Mean and biased variance are taken per sample and per channel group over the cells where
    ``mask`` is non-zero, and the output is zeroed outside the mask so padding activations
    never leak into the next convolution. With an all-ones mask this is
    ``nn.GroupNorm(num_groups, C, affine=False)``.

    Reducing over the group's channels as well as over space is what keeps this defined
    where :func:`masked_instance_norm` is not: a group of ``C / num_groups`` channels
    contributes that many values per valid cell, so one valid cell -- the generator's 1x1
    bottleneck, or a small region at the coarse end of the pyramid -- still yields a
    variance.

    Args:
    ----
        mask: Binary mask of shape (B, 1, h, w) at the resolution of ``values``.
        values: Activations of shape (B, C, h, w).
        num_groups: Number of channel groups; must divide C.
        eps: Numerical floor on the variance, as in ``nn.GroupNorm``.

    Returns:
    -------
        The normalized activations, zero outside the mask.

    """
    batch, channels, height, width = values.shape
    mask = mask.to(values.dtype)
    grouped = values.view(batch, num_groups, channels // num_groups, height, width)
    weight = mask.view(batch, 1, 1, height, width).expand_as(grouped)
    count = weight.sum(dim=(2, 3, 4), keepdim=True).clamp_min(1.0)
    mean = (grouped * weight).sum(dim=(2, 3, 4), keepdim=True) / count
    var = (((grouped - mean) ** 2) * weight).sum(dim=(2, 3, 4), keepdim=True) / count
    normalized = (grouped - mean) / torch.sqrt(var + eps)
    return normalized.view(batch, channels, height, width) * mask


class MaskedGroupNorm2d(nn.Module):
    """``nn.GroupNorm`` (no affine) restricted to the valid cells.

    Parameter-free, so it adds no state-dict keys over the ``nn.InstanceNorm2d`` it stands
    in for, and ``forward`` without a mask is ``F.group_norm``.
    """

    accepts_mask = True

    def __init__(self, num_channels: int, num_groups: int | None = None, eps: float = 1e-5) -> None:
        """Initialize the layer.

        Args:
        ----
            num_channels: Number of channels the norm will see.
            num_groups: Number of channel groups, or None for :func:`default_num_groups`.
            eps: Numerical floor on the variance.

        Raises:
        ------
            ValueError: If ``num_groups`` does not divide ``num_channels``, or if it would
                leave fewer than two channels in a group. A group of one channel is
                InstanceNorm again, which is exactly the degeneracy this class exists to
                avoid, so it is refused here rather than silently normalizing a 1x1 map to
                zero.

        """
        super().__init__()
        if num_groups is None:
            num_groups = default_num_groups(num_channels)
        if num_channels % num_groups:
            raise ValueError(
                f"num_groups={num_groups} does not divide num_channels={num_channels}."
            )
        if num_channels // num_groups < 2:
            raise ValueError(
                f"num_groups={num_groups} leaves {num_channels // num_groups} channel(s) per "
                f"group for num_channels={num_channels}. A group of one channel reduces over "
                "space alone, which has no variance on a 1x1 map -- the generator's "
                "bottleneck -- so every activation there would normalize to exactly 0. Use "
                "fewer groups, or normalization=None for this block."
            )
        self.num_channels = num_channels
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, values: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Normalize ``values``; without a mask this is plain group normalization."""
        if mask is None:
            return F.group_norm(values, self.num_groups, eps=self.eps)
        return masked_group_norm(mask, values, self.num_groups, eps=self.eps)


class MaskedSpatialNorm2d(nn.Module):
    """InstanceNorm where the map is large enough for it, masked GroupNorm where it is not.

    The generator's five stride-2 downsamples take the 48x48 canvas to 24, 12, 6, 3 and
    finally 1, and ``nn.InstanceNorm2d`` cannot run on that last map at all: its reduction
    is over (h, w) per sample and channel, so at 1x1 it has one number, a variance of 0, and
    PyTorch raises ``Expected more than 1 spatial element`` rather than return zeros. The
    masked form does not raise -- it computes the statistics by hand -- which is worse,
    because it returns exact zeros and the UNet's skip connections carry the map around the
    dead bottleneck without a word.

    So this layer switches on the map's shape: at or above :data:`INSTANCE_NORM_MIN_DIM` in
    both dimensions it is :class:`MaskedInstanceNorm2d`, below it :class:`MaskedGroupNorm2d`.
    On the 48x48 canvas that is the 48 and 24 levels on the instance branch and 12 down to 1
    on the group branch. Two properties make the shape the right thing to switch on rather
    than the mask's cell count, which is the other obvious choice:

    * every sample in a batch takes the same branch. Two regions of the same size can differ
      in valid cells at the coarse levels purely through where they land on the stride grid
      -- measured, Corse's 140-cell footprint reaches the 3x3 level with anywhere from 2 to 6
      cells depending on offset alone -- so a count-driven branch would normalize two
      identical regions differently, which is the coverage dependence the mask pyramid exists
      to remove;
    * it is a property of the architecture, not of the data, so it is fixed for a training
      run and cannot change under augmentation.

    The cost is that the branch does depend on the canvas: a region on a native grid below 17
    would take the group branch at level 0 where the same region on the 48x48 canvas takes
    the instance branch. Everything in this project pads to 48 before the generator, so the
    two never disagree in practice, but a change of canvas size is a change of normalization.
    """

    accepts_mask = True

    def __init__(
        self,
        num_channels: int,
        min_instance_dim: int = INSTANCE_NORM_MIN_DIM,
        num_groups: int | None = None,
        eps: float = 1e-5,
    ) -> None:
        """Initialize the layer.

        Args:
        ----
            num_channels: Number of channels the norm will see.
            min_instance_dim: Smallest map dimension handed to the instance branch.
            num_groups: Groups for the group branch, or None for :func:`default_num_groups`.
            eps: Numerical floor on the variance, shared by both branches.

        """
        super().__init__()
        self.instance = MaskedInstanceNorm2d(eps=eps)
        self.group = MaskedGroupNorm2d(num_channels, num_groups=num_groups, eps=eps)
        self.min_instance_dim = min_instance_dim

    def forward(self, values: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Normalize ``values`` with whichever branch its spatial size calls for.

        Args:
        ----
            values: Activations of shape (B, C, h, w).
            mask: Binary mask of shape (B, 1, h, w) at the resolution of ``values``, or None.

        Returns:
        -------
            The normalized activations, zero outside the mask when one is given.

        """
        if min(values.shape[-2], values.shape[-1]) < self.min_instance_dim:
            return self.group(values, mask)
        return self.instance(values, mask)


def masked_batch_statistics(
    values: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-channel mean, biased variance and cell count over the valid cells of a batch.

    ``nn.BatchNorm2d`` takes these over every (sample, row, column) triple; this takes them
    over the triples where ``mask`` is non-zero, which is the same quantity when the mask is
    full. The count is returned because the running-variance update needs Bessel's
    correction against the number of cells that actually contributed, not against ``B*H*W``.

    Args:
    ----
        values: Activations of shape (B, C, h, w).
        mask: Binary mask of shape (B, 1, h, w), broadcastable to ``values``.

    Returns:
    -------
        The mean, the biased variance and the cell count, each of shape (C,). A channel with
        no valid cell gets mean 0, variance 0 and count 1 rather than a NaN.

    """
    mask = mask.to(values.dtype).expand_as(values)
    count = mask.sum(dim=(0, -2, -1)).clamp_min(1.0)
    mean = (values * mask).sum(dim=(0, -2, -1)) / count
    centered = values - mean.view(1, -1, 1, 1)
    var = ((centered**2) * mask).sum(dim=(0, -2, -1)) / count
    return mean, var, count


class MaskedBatchNorm2d(nn.BatchNorm2d):
    """``nn.BatchNorm2d`` whose statistics come from the region's cells only.

    Same parameters and buffers as ``nn.BatchNorm2d`` -- ``weight``, ``bias``,
    ``running_mean``, ``running_var``, ``num_batches_tracked`` -- so a state dict is
    interchangeable with the layer it replaces, and ``forward`` without a mask is exactly
    ``nn.BatchNorm2d.forward``. Given a mask it differs in three ways:

    * mean and variance are taken over the cells where the mask is non-zero, so they no
      longer depend on how much of the canvas the region happens to cover;
    * the running statistics are updated with those masked values, because rollout and
      inference score in ``eval`` mode and an unmasked running path would diverge from
      training exactly where the headline metrics are computed;
    * the output is zeroed outside the mask, so that the constant the affine term leaves
      on the padding plateau -- a value nowhere near the data's range -- is not bled back
      across the coast by the next 3x3 convolution.

    The last point is why the mask cannot be passed to only some of these layers: a masked
    norm without the re-zeroing is worse at the boundary than no masking at all.
    """

    accepts_mask = True

    def forward(self, values: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Normalize ``values``; without a mask this is plain batch normalization.

        Args:
        ----
            values: Activations of shape (B, C, h, w).
            mask: Binary mask of shape (B, 1, h, w) at the resolution of ``values``, or None.

        Returns:
        -------
            The normalized activations, zero outside the mask.

        """
        if mask is None:
            return super().forward(values)
        self._check_input_dim(values)

        # Same bookkeeping as ``nn.modules.batchnorm._BatchNorm.forward``: count the batch,
        # and read momentum=None as a cumulative moving average over batches seen.
        momentum = 0.0 if self.momentum is None else self.momentum
        if self.training and self.track_running_stats and self.num_batches_tracked is not None:
            self.num_batches_tracked.add_(1)
            if self.momentum is None:
                momentum = 1.0 / float(self.num_batches_tracked)

        # As in ``_BatchNorm.forward``: the batch's own statistics when training, or when
        # the layer was built without running buffers.
        running_mean, running_var = self.running_mean, self.running_var
        if self.training or running_mean is None or running_var is None:
            mean, var, count = masked_batch_statistics(values, mask)
            if self.training and running_mean is not None and running_var is not None:
                with torch.no_grad():
                    # Bessel's correction against the cells that contributed, not B*H*W.
                    unbiased = var * count / (count - 1.0).clamp_min(1.0)
                    running_mean.mul_(1 - momentum).add_(momentum * mean.detach())
                    running_var.mul_(1 - momentum).add_(momentum * unbiased.detach())
        else:
            mean, var = running_mean, running_var

        out = (values - mean.view(1, -1, 1, 1)) / torch.sqrt(var.view(1, -1, 1, 1) + self.eps)
        if self.affine:
            out = out * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return out * mask.to(out.dtype)


def binary(weight: torch.Tensor) -> torch.Tensor:
    """Turn coverage weights into the "any data under this cell" mask, same dtype."""
    return (weight > 0).to(weight.dtype)


def region_extent(mask: torch.Tensor) -> torch.Tensor:
    """Per-sample bounding box of the region, as ``(top, left, height, width)``.

    Shape (B, 4), integer. A sample whose mask is empty gets a zero-sized box at the
    origin, which every consumer here treats as "augment nothing".

    Args:
    ----
        mask: Region mask of shape (B, 1, H, W).

    Returns:
    -------
        The bounding boxes, of shape (B, 4).

    """
    present = mask.squeeze(1) > 0
    rows, cols = present.any(dim=-1), present.any(dim=-2)

    def _span(flags: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n = flags.shape[-1]
        idx = torch.arange(n, device=flags.device).expand_as(flags)
        lo = torch.where(flags, idx, torch.full_like(idx, n)).amin(dim=-1)
        hi = torch.where(flags, idx, torch.full_like(idx, -1)).amax(dim=-1)
        empty = ~flags.any(dim=-1)
        return torch.where(empty, torch.zeros_like(lo), lo), torch.where(
            empty, torch.zeros_like(hi), hi - lo + 1
        )

    top, height = _span(rows)
    left, width = _span(cols)
    return torch.stack([top, left, height, width], dim=-1)


def diff_augment_with_mask(
    maps: torch.Tensor, mask: torch.Tensor, policy: str, on_region: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """DiffAugment ``maps`` and carry ``mask`` through the same translation and cutout.

    The mask rides along as an extra channel so that it is shifted with the data and zeroed
    under the cutout, which is the right reading of a cut-out cell: a cell with no data.
    Translation and cutout are sampled per call, so real and fake maps augmented in two
    calls end up under two different masks; the critic scores each under its own, and the
    gradient penalty uses their union.

    Because the mask is already here, the region's own extent is available at the one point
    that decides how large an augmentation is. ``on_region`` uses it: see
    :func:`modules.diff_augment.rand_cutout`. The default is the stock policy, which sizes
    both transforms against the map it is handed -- correct on a native grid, where the map
    *is* the region, and far too strong once the same region sits on a 48x48 canvas. On
    Corse the stock cutout is a 14x14 box against a 23x11 island: it removes more than half
    of it on 6.2% of draws, against a 15% ceiling on the native grid.

    Args:
    ----
        maps: Maps of shape (B, C, H, W).
        mask: Region mask of shape (B, 1, H, W) or (1, 1, H, W).
        policy: The DiffAugment policy string, e.g. ``"translation,cutout"``.
        on_region: Whether to size and place the augmentations on each sample's region
            bounding box instead of on the whole map.

    Returns:
    -------
        The augmented maps and the augmented mask.

    """
    mask = mask.to(maps.dtype).expand(maps.shape[0], -1, -1, -1)
    extent = region_extent(mask) if on_region else None
    stacked = DiffAugment(torch.cat([maps, mask], dim=1).contiguous(), policy=policy, extent=extent)
    return stacked[:, :-1].contiguous(), stacked[:, -1:].contiguous()
