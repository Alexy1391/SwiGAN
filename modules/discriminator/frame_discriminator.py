"""Module containing the frame level discriminator."""

from __future__ import annotations

import torch
from torch import nn

from modules.base_conv_blocks import MaskedDiscriminatorConvBlock
from modules.masking import binary, masked_mean
from modules.utils import glorot_init


class FrameDiscriminator(nn.Module):
    """The frame level discriminator.

    Two convolutions at the tile resolution, then a coverage-weighted global average pool
    over the tiles that have data under them, then one spectral-normed linear readout. The
    pool replaces the fixed 2x2 stride-2 tail that only fitted a 5x6 trunk output: the head
    is now independent of the grid size, and the padding tiles do not enter the score.
    """

    def __init__(self, input_channels: int, normalization: str | None = "instancenorm") -> None:
        """Initialize the input parameters.

        Args:
        ----
            input_channels: Number of input channels.
            normalization: Passed to the convolution blocks; see
                ``MaskedDiscriminatorConvBlock``.

        """
        super().__init__()
        self.conv1 = MaskedDiscriminatorConvBlock(
            in_channels=input_channels,
            out_channels=input_channels * 2,
            kernel_size=2,
            dropout=0.0,
            normalization=normalization,
            padding=0,
            stride=1,
        )
        self.conv2 = MaskedDiscriminatorConvBlock(
            in_channels=input_channels * 2,
            out_channels=input_channels * 2,
            kernel_size=3,
            dropout=0.0,
            normalization=normalization,
            padding=1,
            stride=1,
        )
        self.readout = nn.utils.spectral_norm(nn.Linear(input_channels * 2, 1))
        self.apply(glorot_init)

    def forward(
        self, inputs: torch.Tensor, weight: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass.

        Args:
        ----
            inputs: The trunk output, of shape (batch_size, channels, height, width).
            weight: Coverage weights of the tile grid (the last level of ``mask_pyramid``),
                shape (batch_size, 1, h, w). None means every tile has data under it.

        Returns:
        -------
            A tensor of shape (batch_size, 1) and the two intermediate feature maps.

        """
        features = []
        if weight is None:
            weight = torch.ones(
                inputs.shape[0],
                1,
                inputs.shape[-2] - 1,
                inputs.shape[-1] - 1,
                device=inputs.device,
                dtype=inputs.dtype,
            )
        mask = binary(weight)
        out = self.conv1(inputs, mask)
        features.append(out)
        out = self.conv2(out, mask)
        features.append(out)
        pooled = masked_mean(out, weight, dims=(-2, -1))  # (B, C)
        return self.readout(pooled), features
