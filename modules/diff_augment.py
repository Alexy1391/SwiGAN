"""Differentiable augmentation for Discriminator regularization.

See https://arxiv.org/abs/2006.10738
"""

import torch
import torch.nn.functional as F  # noqa


def DiffAugment(  # noqa
    x: torch.Tensor,
    policy: str = "",
    channels_first: bool = True,
    extent: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply differentiable augmentation to the input given a policy.

    Copied from https://github.com/mit-han-lab/data-efficient-gans

    Args:
    ----
        x: Input Tensor to apply diff augment.
        policy: A comma separated string specifying the transformations to apply.
        channels_first: If true expects a tensor of shape (batch, channels, H, W),
            else (batch, H, W, channels).
        extent: Optional per-sample region bounding box of shape (B, 4), as
            ``(top, left, height, width)``. When given, translation and cutout are sized
            and placed relative to that box instead of the whole map -- see
            :func:`rand_translation` and :func:`rand_cutout`.

    Returns:
    -------
        The augmented input.

    """
    if policy:
        if not channels_first:
            raise NotImplementedError("Only support channels_first=True")
        for p in policy.split(","):
            for f in AUGMENT_FNS[p]:
                x = f(x, extent=extent)
        x = x.contiguous()
    return x


def rand_translation(
    x: torch.Tensor, ratio: float = 0.2, extent: torch.Tensor | None = None
) -> torch.Tensor:
    """Slight modification of the Vanilla DiffAugment translation.

    Handles spatio-temporal data by applying the same transformation to
    all timesteps and all channels of a single batch element.

    Args:
    ----
        x: Input Tensor to apply diff augment.
        ratio: Ratio of the input maps along which the translation will be performed.
        extent: Optional per-sample region bounding box of shape (B, 4), as
            ``(top, left, height, width)``. When given, the shift is ``ratio`` of the
            region's own height and width rather than of the map's, so placing a region on
            a larger canvas no longer widens the translation. The shift is still applied to
            the whole map, and is still drawn per sample.

    Returns:
    -------
        The augmented input.

    """
    if extent is None:
        shift_x, shift_y = int(x.size(-2) * ratio + 0.5), int(x.size(-1) * ratio + 0.5)
        translation_x = torch.randint(
            -shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device
        )
        translation_y = torch.randint(
            -shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device
        )
    else:
        # Per-sample bounds, so torch.randint's scalar low/high cannot be used: draw on
        # [0, 1) and scale into [-shift, shift] instead.
        shift = (extent[:, 2:].to(x.dtype) * ratio + 0.5).long().view(x.size(0), 2, 1, 1)
        span = 2 * shift + 1
        draw = (torch.rand(x.size(0), 2, 1, 1, device=x.device) * span).long().clamp_max(2 * shift)
        translation_x, translation_y = (draw - shift).unbind(dim=1)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(-2), dtype=torch.long, device=x.device),
        torch.arange(x.size(-1), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(-2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(-1) + 1)
    if x.ndim == 4:  # (batch_size, channels, H, W)
        x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
        x = (
            x_pad.permute(0, 2, 3, 1)
            .contiguous()[grid_batch, grid_x, grid_y]
            .permute(0, 3, 1, 2)
            .contiguous()
        )
    else:  # (batch_size, timesteps, channels, H, W)
        x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        x = (
            x_pad.permute(0, 3, 4, 1, 2)
            .contiguous()[grid_batch, grid_x, grid_y]
            .permute(0, 3, 4, 1, 2)
            .contiguous()
        )

    return x


def rand_cutout(
    x: torch.Tensor, ratio: float = 0.3, extent: torch.Tensor | None = None
) -> torch.Tensor:
    """Slight modification of the Vanilla DiffAugment cutout.

    Handles spatio-temporal data by applying the same transformation to
    all timesteps and all channels of a single batch element.

    Args:
    ----
        x: Input Tensor to apply diff augment.
        ratio: Ratio of the input maps along which the translation will be performed.
        extent: Optional per-sample region bounding box of shape (B, 4), as
            ``(top, left, height, width)``. When given, the box is ``ratio`` of the
            region's own height and width, and its centre is drawn uniformly over the
            region rather than over the map. Both halves are needed: sizing alone would
            keep the centre uniform over the canvas, which on Corse's 48x48 canvas would
            drop the fraction of draws that touch the island at all from 1 to about 0.1 --
            a quieter augmentation rather than a better-calibrated one.

    Returns:
    -------
        The augmented input.

    """
    if extent is not None:
        return _region_cutout(x, ratio, extent)

    cutout_size = int(x.size(-2) * ratio + 0.5), int(x.size(-1) * ratio + 0.5)
    offset_x = torch.randint(
        0, x.size(-2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device
    )
    offset_y = torch.randint(
        0, x.size(-1) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device
    )
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(-2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(-1) - 1)
    mask = torch.ones(x.size(0), x.size(-2), x.size(-1), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    if x.ndim == 4:  # (batch_size, channels, H, W)
        x = x * mask.unsqueeze(1)
    else:  # (batch_size, timesteps, channels, H, W)
        x = x * mask.unsqueeze(1).unsqueeze(1)
    return x


def _region_cutout(x: torch.Tensor, ratio: float, extent: torch.Tensor) -> torch.Tensor:
    """Cutout sized and placed on each sample's region box, for :func:`rand_cutout`.

    Written as a comparison against the box bounds rather than as the scatter the canvas
    path uses, because the box size varies per sample here and a single ``meshgrid`` of
    ``cutout_size`` cannot express that. The two also differ at the boundary: the scatter
    clamps out-of-range *indices*, which folds the overhanging part of the box onto the
    border row, while this truncates the box at the region's edge.
    """
    origin, size = extent[:, :2], extent[:, 2:]
    box = (size.to(x.dtype) * ratio + 0.5).long()
    # Same upper bound as the canvas path, on the region's own span: an even-sided box can
    # start one cell past the far edge so that both edges are reachable.
    span = size + (1 - box % 2)
    centre = origin + (torch.rand_like(span, dtype=x.dtype) * span).long().clamp_max(span - 1)

    start = centre - box // 2
    lo, hi = origin, origin + size
    start = torch.minimum(torch.maximum(start, lo), hi - 1)
    end = torch.minimum(start + box, hi)
    # An empty box (a region too small for `ratio` to reach one cell) cuts nothing.
    end = torch.where(box > 0, end, start)

    rows = torch.arange(x.size(-2), device=x.device).view(1, -1, 1)
    cols = torch.arange(x.size(-1), device=x.device).view(1, 1, -1)
    y0, x0 = start[:, 0].view(-1, 1, 1), start[:, 1].view(-1, 1, 1)
    y1, x1 = end[:, 0].view(-1, 1, 1), end[:, 1].view(-1, 1, 1)
    cut = (rows >= y0) & (rows < y1) & (cols >= x0) & (cols < x1)

    mask = (~cut).to(x.dtype)
    if x.ndim == 4:  # (batch_size, channels, H, W)
        return x * mask.unsqueeze(1)
    return x * mask.unsqueeze(1).unsqueeze(1)  # (batch_size, timesteps, channels, H, W)


AUGMENT_FNS = {
    "translation": [rand_translation],
    "cutout": [rand_cutout],
}
