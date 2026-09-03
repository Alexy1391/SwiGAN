"""Module containing decoder for the UNET generator module."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F  # noqa
from torch import nn
from torchvision.ops import stochastic_depth

from modules.base_conv_blocks import single_conv_block
from modules.scse import SCSEModule
from modules.utils import glorot_init, init_noise_weights


class UNetDecoderBlock(nn.Module):
    """A single upscale decoding block.

    Uses skip connections, transpose convolutions and scSE attention modules.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int,
        dropout: float,
        normalization: str | None,
        prob: float = 1.0,
        additional_pad: tuple[int, ...] | None = None,
        noise_weight_init: str = "randn",
    ) -> None:
        """Initialize the input parameters.

        Args:
        ----
            in_channels: Number of input channels.
            out_channels: The number of output channels.
            skip_channels: The number of channels of the tensor from the
                downsampling stage that will be appended to the input of the
                current block.
            dropout: Dropout rate.
            normalization: normalization: The type of normalization to apply.
                If None, no normalization is applied. Supported normalization are
                "instancenorm" for InstanceNorm2D, "batchnorm" for BatchNorm2D.
            prob: Survival probability for stochastic depth.
            additional_pad: Additional padding to apply to the input to match the
                skip input.
            noise_weight_init: How to initialize the per-channel noise gains. See
                ``modules.utils.init_noise_weights``.

        """
        super().__init__()
        self.inner_upscale = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )

        self.upscale = nn.Sequential(
            nn.Dropout(dropout),
            nn.ConvTranspose2d(
                in_channels,
                in_channels,
                kernel_size=2,
                stride=2,
            ),
            nn.LeakyReLU(0.2),
        )
        self.conv1 = single_conv_block(
            in_channels + skip_channels,
            out_channels,
            dropout=dropout,
            normalization=normalization,
            kernel_size=3,
            padding=1,
        )
        self.conv2 = single_conv_block(
            out_channels,
            out_channels,
            dropout=dropout,
            normalization=normalization,
            kernel_size=3,
            padding=1,
        )
        self.attention = SCSEModule(in_channels=out_channels)
        self.noise_weights1 = init_noise_weights(out_channels, noise_weight_init)
        self.noise_weights2 = init_noise_weights(out_channels, noise_weight_init)
        self.prob = prob
        self.additional_pad = additional_pad

    def forward(
        self, inputs: torch.Tensor, skip_inputs: list[torch.Tensor] | None = None
    ) -> torch.Tensor:
        """Forward pass.

        Args:
        ----
            inputs: The input Tensor. Should be of shape
                (batch_size, channels, height, width).
            skip_inputs: List containing the outputs of the downsampling phase.

        Returns:
        -------
            A Tensor of shape (batch_size, out_channels, 2 * height, 2 * width).

        """
        # if self.training: # Only add noise during training
        x = self.inner_upscale(inputs)
        out = self.upscale(inputs)
        noise = torch.randn((out.shape[0], 1, out.shape[-2], out.shape[-1]), device=inputs.device)
        if self.additional_pad is not None:
            x = F.pad(x, self.additional_pad)
            out = F.pad(out, self.additional_pad)
            noise = F.pad(noise, self.additional_pad)

        if skip_inputs is not None:
            out = torch.concat([out, skip_inputs], dim=1)
        out = self.conv1(out)
        out = out + noise * self.noise_weights1[None, :, None, None]

        out = self.conv2(out)
        out = out + noise * self.noise_weights2[None, :, None, None]

        out = self.attention(out)
        out = stochastic_depth(out, 1 - self.prob, mode="batch", training=self.training)
        return out + x


class UNetFrameDecoder(nn.Module):
    """The frame decoder applied to each map individually."""

    def __init__(
        self,
        output_dim: int,
        head_channels: int,
        encoder_channels: list[int],
        decoder_channels: list[int],
        dropout: float = 0.3,
        normalization: str | None = "batchnorm",
        noise_weight_init: str = "randn",
    ) -> None:
        """Initialize input parameters.

        Args:
        ----
            output_dim: Number of output channels.
            head_channels: Number of channels of the center block in the UNet model.
            encoder_channels: List containing the number of channels in the downsampling phase.
            decoder_channels: A list of output channels of the upsampling stage.
            dropout: The dropout rate.
            normalization: normalization: The type of normalization to apply.
                If None, no normalization is applied. Supported normalization are
                "instancenorm" for InstanceNorm2D, "batchnorm" for BatchNorm2D.
            noise_weight_init: How to initialize the per-channel noise gains. See
                ``modules.utils.init_noise_weights``.

        """
        super().__init__()
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = encoder_channels + [0]
        out_channels = decoder_channels
        self.probs = list(np.arange(0.5, 1.0, 0.5 / (len(in_channels) - 1))) + [1.0]

        # combine decoder keyword arguments
        blocks: list[nn.Module] = [
            UNetDecoderBlock(
                in_channels=in_channels[i],
                skip_channels=skip_channels[i],
                out_channels=out_channels[i],
                dropout=0.3 if i < 3 else dropout,
                normalization=normalization,
                prob=self.probs[i],
                additional_pad=(1, 0, 1, 0) if i == 0 else None,  # Needed to harmonize the shapes
                noise_weight_init=noise_weight_init,
            )
            for i in range(len(in_channels))
        ]
        self.head = single_conv_block(
            in_channels=out_channels[-1],
            out_channels=output_dim,
            kernel_size=1,
            dropout=dropout,
            normalization=None,
            padding=0,
            activation=None,
        )

        self.blocks = nn.ModuleList(blocks)

        self.apply(glorot_init)
        self.output_dim = output_dim

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """Forward pass.

        Args:
        ----
            features: List containing the output of each encoder block
                from the downsampling stage.

        Returns:
        -------
            A Tensor of shape (batch_size, output_dim, H, W).

        """
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]
        x = head
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)
        return self.head(x)
