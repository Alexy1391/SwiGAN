"""Module containing decoder for the UNET generator module."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F  # noqa
from torch import nn
from torchvision.ops import stochastic_depth

from modules.base_conv_blocks import single_conv_block
from modules.masking import binary
from modules.scse import SCSEModule
from modules.utils import (
    draw_coherent_field,
    glorot_init,
    init_coherent_noise_weights,
    init_noise_weights,
)


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
        coherent_noise_init: str | float | None = None,
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
                "batchnorm", "instancenorm" or "groupnorm"; see
                :func:`modules.base_conv_blocks.single_conv_block`.
            prob: Survival probability for stochastic depth.
            additional_pad: Additional padding to apply to the input to match the
                skip input.
            noise_weight_init: How to initialize the per-channel noise gains. See
                ``modules.utils.init_noise_weights``.
            coherent_noise_init: How to initialize the per-channel gains of the spatially
                coherent noise channel, or None to leave the channel out of the model
                entirely -- no parameters are created, so the state dict is the one the
                existing checkpoints hold. See
                ``modules.utils.init_coherent_noise_weights``.

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
        # Registered only when asked for, so a model built without the channel keeps exactly
        # the parameter names the existing checkpoints hold and loads them strictly.
        if coherent_noise_init is None:
            self.coherent_noise_weights1 = None
            self.coherent_noise_weights2 = None
        else:
            self.coherent_noise_weights1 = init_coherent_noise_weights(
                out_channels, coherent_noise_init
            )
            self.coherent_noise_weights2 = init_coherent_noise_weights(
                out_channels, coherent_noise_init
            )
        self.prob = prob
        self.additional_pad = additional_pad

    def forward(
        self,
        inputs: torch.Tensor,
        skip_inputs: list[torch.Tensor] | None = None,
        weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        With ``weight`` -- the coverage of the level this block *writes*, which is the level
        its skip connection arrives at -- the two transposed convolutions are re-zeroed
        before ``conv1`` reads them, the injected noise is masked, and the residual is
        re-zeroed on the way out. Both ``ConvTranspose2d`` have a bias, so their padding is a
        non-zero constant that a 3x3 kernel would otherwise carry back across the coast.

        Args:
        ----
            inputs: The input Tensor. Should be of shape
                (batch_size, channels, height, width).
            skip_inputs: List containing the outputs of the downsampling phase.
            weight: Coverage weights at the block's *output* resolution, of shape
                (batch_size, 1, 2 * height, 2 * width), or None to run the block exactly as
                rounds 1-12 did.

        Returns:
        -------
            A Tensor of shape (batch_size, out_channels, 2 * height, 2 * width).

        """
        mask = None if weight is None else binary(weight)
        # if self.training: # Only add noise during training
        x = self.inner_upscale(inputs)
        out = self.upscale(inputs)
        noise = torch.randn((out.shape[0], 1, out.shape[-2], out.shape[-1]), device=inputs.device)
        if self.additional_pad is not None:
            x = F.pad(x, self.additional_pad)
            out = F.pad(out, self.additional_pad)
            noise = F.pad(noise, self.additional_pad)
        if mask is not None:
            out = out * mask
            noise = noise * mask
        # Constant over height and width, so it survives the spatial averaging that reduces
        # `noise` to nothing -- no padding needed, it broadcasts to whatever the block writes.
        coherent = draw_coherent_field(
            self.coherent_noise_weights1, out.shape[0], inputs.device, inputs.dtype, mask
        )

        if skip_inputs is not None:
            out = torch.concat([out, skip_inputs], dim=1)
        out = self.conv1(out, mask)
        out = out + noise * self.noise_weights1[None, :, None, None]
        if coherent is not None:
            out = out + coherent * self.coherent_noise_weights1[None, :, None, None]

        out = self.conv2(out, mask)
        out = out + noise * self.noise_weights2[None, :, None, None]
        if coherent is not None:
            out = out + coherent * self.coherent_noise_weights2[None, :, None, None]

        out = self.attention(out, weight)
        out = stochastic_depth(out, 1 - self.prob, mode="batch", training=self.training)
        out = out + x
        if mask is not None:
            out = out * mask
        return out


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
        coherent_noise_init: str | float | None = None,
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
                "batchnorm", "instancenorm" or "groupnorm"; see
                :func:`modules.base_conv_blocks.single_conv_block`.
            noise_weight_init: How to initialize the per-channel noise gains. See
                ``modules.utils.init_noise_weights``.
            coherent_noise_init: How to initialize the per-channel gains of the spatially
                coherent noise channel, or None to leave the channel out of the model
                entirely -- no parameters are created, so the state dict is the one the
                existing checkpoints hold. See
                ``modules.utils.init_coherent_noise_weights``.

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
                coherent_noise_init=coherent_noise_init,
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

    def forward(
        self, features: list[torch.Tensor], levels: list[torch.Tensor] | None = None
    ) -> torch.Tensor:
        """Forward pass.

        Args:
        ----
            features: List containing the output of each encoder block
                from the downsampling stage.
            levels: The encoder's coverage pyramid, finest first, as returned by
                ``UNetFrameEncoder.forward``. Reversed here so that decoder block ``i``
                normalizes over the same cells as the encoder level it is fed by. None runs
                the decoder unmasked.

        Returns:
        -------
            A Tensor of shape (batch_size, output_dim, H, W).

        """
        features = features[::-1]  # reverse channels to start from head of encoder
        # Coarsest first, aligned with `features`: weights[i + 1] is the resolution decoder
        # block i writes at, and weights[-1] the one the 1x1 head writes at.
        weights: list[torch.Tensor | None] = (
            [None] * (len(features)) if levels is None else list(levels[::-1])
        )

        head = features[0]
        skips = features[1:]
        x = head
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip, weights[i + 1])
        head_weight = weights[len(self.blocks)]
        return self.head(x, None if head_weight is None else binary(head_weight))
