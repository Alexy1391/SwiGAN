"""Encoder for the UNET SWIGAN variant."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torchvision.ops import stochastic_depth

from modules.base_conv_blocks import single_conv_block
from modules.generator.center_block import CenterBlock
from modules.scse import SCSEModule
from modules.utils import glorot_init, init_noise_weights


class UNetEncoderBlock(nn.Module):
    """A single upscale decoding block.

    Uses skip connections, transpose convolutions and scSE attention modules.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float,
        normalization: str | None,
        prob: float = 1.0,
        noise_weight_init: str = "randn",
    ) -> None:
        """Initialize the input parameters.

        Args:
        ----
            in_channels: Number of input channels.
            out_channels: The number of output channels.
            dropout: Dropout rate.
            normalization: normalization: The type of normalization to apply.
                If None, no normalization is applied. Supported normalization are
                "instancenorm" for InstanceNorm2D, "batchnorm" for BatchNorm2D.
            prob: Survival probability for stochastic depth.
            noise_weight_init: How to initialize the per-channel noise gains. See
                ``modules.utils.init_noise_weights``.

        """
        super().__init__()
        self.proj_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.conv1 = single_conv_block(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            dropout=dropout,
            normalization=normalization,
            padding=1,
        )
        self.conv2 = single_conv_block(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            dropout=dropout,
            normalization=normalization,
            padding=1,
        )
        self.attention = SCSEModule(in_channels=out_channels)
        self.noise_weights1 = init_noise_weights(out_channels, noise_weight_init)
        self.noise_weights2 = init_noise_weights(out_channels, noise_weight_init)
        self.prob = prob

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
        ----
            inputs: The input Tensor. Should be of shape
                (batch_size, channels, height, width).

        Returns:
        -------
            A Tensor of shape (batch_size, out_channels, 2 * height, 2 * width).

        """
        noise = torch.randn(
            (inputs.shape[0], 1, inputs.shape[-2], inputs.shape[-1]), device=inputs.device
        )
        out = inputs
        x = self.proj_layer(out)
        out = self.conv1(out)
        out = out + noise * self.noise_weights1[None, :, None, None]

        out = self.conv2(out)
        out = out + noise * self.noise_weights2[None, :, None, None]

        out = self.attention(out)
        out = stochastic_depth(out, 1 - self.prob, mode="batch", training=self.training)
        out = out + x
        return out


class UNetFrameEncoder(nn.Module):
    """The frame encoder applied to each map individually."""

    def __init__(
        self,
        in_channels: int,
        out_channels: list[int],
        dropout: float,
        normalization: str | None = "batchnorm",
        apply_center_block: bool = False,
        noise_weight_init: str = "randn",
        encoder_late_dropout: float = 0.0,
    ) -> None:
        """Initialize the module.

        Args:
        ----
            in_channels: Number of input channels.
            out_channels: Number of output_channels.
            dropout: Dropout rate.
            normalization: normalization: The type of normalization to apply.
                If None, no normalization is applied. Supported normalization are
                "instancenorm" for InstanceNorm2D, "batchnorm" for BatchNorm2d.
            apply_center_block: Whether to apply the center block at the end of the encoding.
            noise_weight_init: How to initialize the per-channel noise gains. See
                ``modules.utils.init_noise_weights``.
            encoder_late_dropout: Dropout rate for the *last three* downsampling blocks only.
                ``article_recherche.pdf`` §8.2.1 puts dropout in "the last three layers of the
                downsampling phase and the first three layers of the upsampling phase", citing
                Isola et al. [2018] on UNet generators that ignore the noise vector. The
                upsampling half is already hardcoded at ``unet_frame_decoder.py:176``; this is
                the downsampling half, which rounds 1-10 ran without. 0.0 leaves the encoder
                exactly as those rounds had it.

        """
        super().__init__()
        in_channel = in_channels
        layers = []
        downsample_layers = []
        self.probs = list(np.arange(1.0, 0.5, -0.5 / (len(out_channels) - 1))) + [0.5]
        late_from = len(out_channels) - 3
        for idx, out in enumerate(out_channels):
            # Only the last three blocks are affected, and only when the rate is set, so a
            # non-zero ``dropout`` keeps applying everywhere exactly as it did before.
            block_dropout = dropout
            if encoder_late_dropout > 0.0 and idx >= late_from:
                block_dropout = encoder_late_dropout

            layers.append(
                UNetEncoderBlock(
                    in_channels=in_channel,
                    out_channels=out,
                    dropout=block_dropout,
                    normalization=normalization,
                    prob=self.probs[idx],
                    noise_weight_init=noise_weight_init,
                )
            )

            downsample_layers.append(
                single_conv_block(
                    in_channels=out,
                    out_channels=out,
                    kernel_size=2,
                    stride=2,
                    dropout=block_dropout,
                    normalization=normalization,
                    padding=0,
                )
            )
            in_channel = out
        self.layers = nn.ModuleList(layers)
        self.downsample_layers = nn.ModuleList(downsample_layers)

        if apply_center_block:
            self.center = CenterBlock(
                out_channels[-1],
                out_channels[-1],
                dropout,
                normalization,
            )
        else:
            self.center = nn.Identity()
        self.apply(glorot_init)

    def forward(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass.

        Args:
        ----
            inputs: Input tensor. Should be of size
                (batch_size, channels, height, width)

        Returns:
        -------
            A Tensor of shape (batch_size, out_channels)

        """
        out = inputs
        outputs = []
        for _, (block, downsample) in enumerate(
            zip(self.layers, self.downsample_layers, strict=True)
        ):
            out = block(out)
            outputs.append(out)
            out = downsample(out)
        out = self.center(out)
        outputs.append(out)
        return outputs
