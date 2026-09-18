"""Module containing Spatial and Channel wise Squeeze Excitation Network."""

from __future__ import annotations

import torch
from torch import nn

from modules.masking import masked_mean


class SCSEModule(nn.Module):
    """Implement a Spatial and Channel Squeeze Excitation (scSE) Module.

    Incorporates Attention mechanism in the decoding phase.
    See <https://arxiv.org/abs/1803.02579> for more details.

    The channel gate opens with a global average over the map. On a region padded onto a
    common canvas that average is mostly padding -- 94% of it on Corse's 48x48 canvas -- so
    the per-channel gate is set by the sea rather than by the island. Given the coverage
    weights of :func:`modules.masking.generator_mask_pyramid`, the pool becomes a
    coverage-weighted mean over the region's cells instead, which is the same number when
    the mask is full. The two gates are multiplicative, so the output stays zero wherever the
    input is and no extra re-zeroing is needed.
    """

    accepts_mask = True

    def __init__(self, in_channels: int, reduction: int = 8) -> None:
        """Initialize the input class.

        Args:
        ----
            in_channels: Number of input channels.
            reduction: Reduction factor to apply. Reduces the number of input_channels
                to input_channels // reduction.

        """
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.BatchNorm2d(in_channels // reduction),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def channel_gate(self, x: torch.Tensor, weight: torch.Tensor | None) -> torch.Tensor:
        """Run the channel branch, pooling over the region's cells when given weights.

        ``self.cSE[0]`` is the global average pool; with weights it is replaced by the
        coverage-weighted mean and the rest of the branch -- the two 1x1 convolutions and
        their BatchNorm, which see a 1x1 map and so normalize over the batch alone -- runs
        unchanged on the pooled vector.

        Args:
        ----
            x: Activations of shape (B, C, h, w).
            weight: Coverage weights of shape (B, 1, h, w), or None for the plain pool.

        Returns:
        -------
            The per-channel gate, of shape (B, C, 1, 1).

        """
        if weight is None:
            return self.cSE(x)
        gate = masked_mean(x, weight)[..., None, None]
        for layer in list(self.cSE)[1:]:
            gate = layer(gate)
        return gate

    def forward(self, x: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass.

        Args:
        ----
            x: The input tensor. Should be of shape (batch_size, channels, height, width).
            weight: Coverage weights of shape (batch_size, 1, height, width) at the
                resolution of ``x``, or None for the unmasked global pool.

        Returns:
        -------
            A tensor of shape (batch_size, channels, height, width)

        """
        return x * self.channel_gate(x, weight) + x * self.sSE(x)
