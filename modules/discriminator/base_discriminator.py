"""Module containing the base discriminator."""

from __future__ import annotations

import torch
from torch import nn

from modules.base_conv_blocks import MaskedDiscriminatorConvBlock
from modules.discriminator.minibatch_discrimination import MiniBatchDiscrimination
from modules.masking import binary, mask_pyramid
from modules.utils import glorot_init


class BaseDiscriminator(nn.Module):
    """The network for the common parts of the discriminators.

    Three 3x3 stride-2 blocks followed by MiniBatch Discrimination. The trunk is fully
    convolutional and mask-aware: given the region mask it derives the coverage pyramid of
    :func:`modules.masking.mask_pyramid`, normalizes each level over the region's cells
    only, zeroes the padding at every level, and optionally reads the mask as an extra input
    channel. Nothing here depends on the grid size, so the same trunk runs on Grand Est's
    36x44 grid (output 5x6), Corse's 23x11 (3x2) or a 64x64 canvas (8x8).
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: list[int],
        mbd_output_channels: int | None = None,
        mask_channel: bool = False,
        normalization: str | None = "instancenorm",
    ) -> None:
        """Initialize the input arguments.

        Args:
        ----
            input_channels: The number of map channels. The mask channel, when enabled, is
                added on top.
            output_channels: The number of output channels of each block.
            mbd_output_channels: The number of output channels of the MiniBatch
                Discrimination module. If None, no Minibatch Discrimination is used.
            mask_channel: Whether to feed the region mask as an extra input channel. Real
                and generated maps share the mask, so it carries no real/fake label; what
                it gives the critic is the difference between "zero because padding" and
                "zero because the standardized value is the mean", which are otherwise
                the same number.
            normalization: "instancenorm" for the masked InstanceNorm2d, or None for no
                normalization at all (spectral norm stays on every convolution).

        """
        super().__init__()
        self.mask_channel = mask_channel
        channels = [input_channels + int(mask_channel)] + output_channels
        self.blocks = nn.ModuleList(
            [
                MaskedDiscriminatorConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    dropout=0.0,
                    normalization=normalization,
                    padding=1,
                    stride=2,
                )
                for in_channels, out_channels in zip(channels[:-1], channels[1:], strict=False)
            ]
        )
        self.mbd: MiniBatchDiscrimination | None
        if mbd_output_channels is not None:
            self.mbd = MiniBatchDiscrimination(
                in_features=output_channels[-1], out_features=mbd_output_channels, kernel_dims=10
            )
        else:
            self.mbd = None
        self.apply(glorot_init)

    def forward(
        self, inputs: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass.

        Args:
        ----
            inputs: Input tensor of shape (batch_size, channels, height, width).
            mask: Region mask of shape (batch_size, 1, height, width) or
                (1, 1, height, width). None means every cell is data, which reproduces the
                unmasked trunk.

        Returns:
        -------
            The trunk output of shape (batch_size, channels + mbd_channels, h, w) and the
            output of each block, for feature matching.

        """
        if mask is None:
            mask = torch.ones_like(inputs[:, :1])
        mask = mask.to(inputs.dtype).expand(inputs.shape[0], -1, -1, -1)
        levels = mask_pyramid(mask, len(self.blocks))

        features = []
        out = torch.cat([inputs, mask], dim=1) if self.mask_channel else inputs
        for block, weight in zip(self.blocks, levels[1 : len(self.blocks) + 1], strict=True):
            out = block(out, binary(weight))
            features.append(out)
        if self.mbd is not None:
            out = self.mbd(out, binary(levels[len(self.blocks)]))
        return out.contiguous(), features
