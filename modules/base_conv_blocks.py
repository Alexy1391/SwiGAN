"""Module containing basic conv blocks for building the models."""

from __future__ import annotations

import torch
from torch import nn

from modules.masking import (
    MaskedBatchNorm2d,
    MaskedGroupNorm2d,
    MaskedInstanceNorm2d,
    MaskedSpatialNorm2d,
)


class MaskedSequential(nn.Sequential):
    """``nn.Sequential`` that hands the region mask to the layers that take one.

    Layers advertise themselves with an ``accepts_mask`` class attribute --
    :class:`modules.masking.MaskedBatchNorm2d`, :class:`~modules.masking.MaskedInstanceNorm2d`
    and this class do -- and everything else is called as usual. When a mask is given the
    block's output is zeroed outside it, so a normalization's affine term never leaves a
    constant on the padding for the next convolution to bleed back inward. Called without a
    mask this is ``nn.Sequential`` exactly, which is what every layer here does when
    ``generator_masked_norm`` is off.

    The child modules keep their positional names, so the state dict is the one the plain
    ``nn.Sequential`` version of this block wrote.
    """

    accepts_mask = True

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Run the layers in order, threading ``mask`` through the mask-aware ones.

        Args:
        ----
            inputs: Activations of shape (B, C_in, h, w).
            mask: Binary mask of shape (B, 1, h', w') at the *output* resolution, or None.

        Returns:
        -------
            The block's output, zero outside the mask when one is given.

        """
        for module in self:
            if mask is not None and getattr(module, "accepts_mask", False):
                inputs = module(inputs, mask)
            else:
                inputs = module(inputs)
        if mask is None:
            return inputs
        return inputs * mask.to(inputs.dtype)


def single_conv_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int | tuple[int],
    normalization: str | None,
    dropout: float = 0.3,
    stride: int = 1,
    padding: str | int = 1,
    activation: bool | None = True,
) -> nn.Module:
    """Single convolutional building block.

    A single 2D convolutional building block with normalization,
    dropout and leakyrelu activation.

    Args:
    ----
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Kernel size for the convolution.
        normalization: The type of normalization to apply. If None, no normalization is
            applied. "batchnorm" is :class:`~modules.masking.MaskedBatchNorm2d`.
            "instancenorm" is :class:`~modules.masking.MaskedSpatialNorm2d`, which is
            InstanceNorm on maps of at least
            :data:`~modules.masking.INSTANCE_NORM_MIN_DIM` in both dimensions and masked
            GroupNorm below that -- plain InstanceNorm cannot run on the generator's 1x1
            bottleneck at all. "groupnorm" is
            :class:`~modules.masking.MaskedGroupNorm2d` at every resolution.
        dropout: Dropout rate.
        stride: Stride of the convolutions.
        padding: Padding to add.
        activation: Whether to use LeakyReLu activation with 0.2 negative slope.

    Returns:
    -------
        A :class:`MaskedSequential`. Called as ``block(x)`` it is the plain
        convolution-normalization-activation-dropout stack rounds 1-12 trained; called as
        ``block(x, mask)`` the normalization takes its statistics over the region's cells
        only and the output is zeroed outside the mask.

    """
    layers: list[nn.Module] = [
        nn.Conv2d(
            in_channels,
            out_channels,
            stride=stride,
            kernel_size=kernel_size,
            padding=padding,
        ),
    ]
    if normalization is None:
        pass
    elif normalization == "batchnorm":
        layers.append(MaskedBatchNorm2d(out_channels))
    elif normalization == "instancenorm":
        layers.append(MaskedSpatialNorm2d(out_channels))
    elif normalization == "groupnorm":
        layers.append(MaskedGroupNorm2d(out_channels))
    else:
        raise NotImplementedError(
            f"Unknown normalization: {normalization}. Please provide one of 'batchnorm', "
            "'instancenorm', 'groupnorm', None."
        )

    if activation:
        layers.append(nn.LeakyReLU(0.2))

    layers.append(nn.Dropout(dropout))

    return MaskedSequential(*layers)


def single_discriminator_conv_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int | tuple[int],
    normalization: str | None,
    dropout: float = 0.3,
    stride: int = 1,
    padding: str | int = 1,
    activation: bool = True,
) -> nn.Module:
    """Single convolutional building block for the discriminator.

    Applies spectral normalization to avoid mregularize the Discriminator.
    See https://arxiv.org/abs/1802.05957 .

    A single 2D convolutional building block with normalization,
    dropout and leakyrelu activation.

    Args:
    ----
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Kernel size for the convolution.
        normalization: The type of normalization to apply.
            If None, no normalization is applied. Supported normalization are
            "instancenorm" for InstanceNorm2D, "batchnorm" for BatchNorm2D.
        dropout: Dropout rate.
        stride: Stride of the convolutions.
        padding: Padding to add.
        activation: Whether to use LeakyReLu activation with 0.2 negative slope.

    Returns:
    -------
        The convolution block.

    """
    layers: list[nn.Module] = [
        nn.utils.spectral_norm(
            nn.Conv2d(
                in_channels,
                out_channels,
                stride=stride,
                kernel_size=kernel_size,
                padding=padding,
            )
        ),
    ]
    if normalization is None:
        pass
    elif normalization == "batchnorm":
        layers.append(nn.BatchNorm2d(out_channels))
    elif normalization == "instancenorm":
        layers.append(nn.InstanceNorm2d(out_channels))
    else:
        raise NotImplementedError(f"Unknown normalization: {normalization}")

    if activation:
        layers.append(nn.LeakyReLU(0.2))

    layers.append(nn.Dropout(dropout))

    return nn.Sequential(*layers)


def single_conv3d_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int | tuple[int, int, int],
    normalization: str | None,
    dropout: float = 0.3,
    stride: int = 1,
    padding: str | int = 1,
    activation: bool = True,
) -> nn.Module:
    """Single convolutional building block.

    A single 3D convolutional building block with normalization,
    dropout and leakyrelu activation.

    Args:
    ----
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Kernel size for the convolution.
        normalization: The type of normalization to apply.
            If None, no normalization is applied. Supported normalization are
            "instancenorm" for InstanceNorm2D, "batchnorm" for BatchNorm2D.
        dropout: Dropout rate.
        stride: Stride of the convolutions.
        padding: Padding to add.
        activation: Whether to use LeakyReLu activation with 0.2 negative slope.

    Returns:
    -------
        The convolution block.

    """
    layers: list[nn.Module] = [
        nn.Conv3d(
            in_channels,
            out_channels,
            stride=stride,
            kernel_size=kernel_size,
            padding=padding,
        ),
    ]
    if normalization is None:
        pass
    elif normalization == "batchnorm":
        layers.append(nn.BatchNorm3d(out_channels))
    elif normalization == "instancenorm":
        layers.append(nn.InstanceNorm3d(out_channels))
    else:
        raise NotImplementedError(f"Unknown normalization: {normalization}")

    if activation:
        layers.append(nn.LeakyReLU(0.2))

    layers.append(nn.Dropout(dropout))

    return nn.Sequential(*layers)


def single_discriminator_conv3d_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int | tuple[int, int, int],
    normalization: str | None,
    dropout: float = 0.3,
    stride: int = 1,
    padding: str | int = 1,
    activation: bool = True,
) -> nn.Module:
    """Single convolutional building block.

    A single 3D convolutional building block with normalization,
    dropout and leakyrelu activation.

    Applies spectral normalization to avoid mregularize the Discriminator.
    See https://arxiv.org/abs/1802.05957 .

    Args:
    ----
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Kernel size for the convolution.
        normalization: The type of normalization to apply.
            If None, no normalization is applied. Supported normalization are
            "instancenorm" for InstanceNorm2D, "batchnorm" for BatchNorm2D.
        dropout: Dropout rate.
        stride: Stride of the convolutions.
        padding: Padding to add.
        activation: Whether to use LeakyReLu activation with 0.2 negative slope.

    Returns:
    -------
        The convolution block.

    """
    layers: list[nn.Module] = [
        nn.utils.spectral_norm(
            nn.Conv3d(
                in_channels,
                out_channels,
                stride=stride,
                kernel_size=kernel_size,
                padding=padding,
            )
        ),
    ]
    if normalization is None:
        pass
    elif normalization == "batchnorm":
        layers.append(nn.BatchNorm3d(out_channels))
    elif normalization == "instancenorm":
        layers.append(nn.InstanceNorm3d(out_channels))
    else:
        raise NotImplementedError(f"Unknown normalization: {normalization}")

    if activation:
        layers.append(nn.LeakyReLU(0.2))

    layers.append(nn.Dropout(dropout))

    return nn.Sequential(*layers)


class MaskedDiscriminatorConvBlock(nn.Module):
    """A critic convolution block whose normalization only sees the region's cells.

    Same layers as :func:`single_discriminator_conv_block` -- spectral-normed convolution,
    normalization, LeakyReLU(0.2), dropout -- but the normalization is the masked
    InstanceNorm of :mod:`modules.masking`, and the output is zeroed outside the mask so that
    the padding activations of one level never feed the convolution of the next. Without a
    mask it behaves exactly like the unmasked block.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        normalization: str | None,
        dropout: float = 0.0,
        stride: int = 1,
        padding: str | int = 1,
        activation: bool = True,
    ) -> None:
        """Initialize the block.

        Args:
        ----
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Kernel size for the convolution.
            normalization: Either "instancenorm" for the masked InstanceNorm2d, or None.
                BatchNorm mixes samples and is not supported here: a critic under a
                gradient penalty is meant to score each map on its own.
            dropout: Dropout rate.
            stride: Stride of the convolution.
            padding: Padding of the convolution.
            activation: Whether to apply LeakyReLU with 0.2 negative slope.

        """
        super().__init__()
        self.conv = nn.utils.spectral_norm(
            nn.Conv2d(
                in_channels,
                out_channels,
                stride=stride,
                kernel_size=kernel_size,
                padding=padding,
            )
        )
        if normalization is None:
            self.norm: nn.Module | None = None
        elif normalization == "instancenorm":
            self.norm = MaskedInstanceNorm2d()
        else:
            raise NotImplementedError(
                f"Unknown critic normalization: {normalization}. "
                "Please provide one of 'instancenorm', None."
            )
        self.activation = nn.LeakyReLU(0.2) if activation else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass.

        Args:
        ----
            inputs: Activations of shape (B, C_in, h, w).
            mask: Binary mask of shape (B, 1, h', w') at the *output* resolution, or None.

        Returns:
        -------
            Activations of shape (B, C_out, h', w'), zero outside the mask.

        """
        out = self.conv(inputs)
        if self.norm is not None:
            out = self.norm(out, mask)
        out = self.dropout(self.activation(out))
        if mask is not None:
            out = out * mask.to(out.dtype)
        return out
