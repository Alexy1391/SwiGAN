"""Contains the modules for the image encoder."""

from __future__ import annotations

from modules.base_conv_blocks import MaskedSequential, single_conv_block


class CenterBlock(MaskedSequential):
    """Center convolutional block to apply at the end of the unet downsampling stage.

    A :class:`~modules.base_conv_blocks.MaskedSequential`, so ``block(x, mask)`` normalizes
    over the region's cells and ``block(x)`` is the plain two-convolution stack.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float,
        normalization: str | None,
    ) -> None:
        """Initialize the class.

        Args:
        ----
            in_channels: Number of input channels.
            out_channels: Number of output_channels.
            dropout: Dropout rate.
            normalization: normalization: The type of normalization to apply.
                If None, no normalization is applied. Supported normalization are
                "batchnorm", "instancenorm" or "groupnorm"; see
                :func:`modules.base_conv_blocks.single_conv_block`.

        """
        conv1 = single_conv_block(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            normalization=normalization,
            activation=True,
            dropout=dropout,
        )
        conv2 = single_conv_block(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            normalization=normalization,
            activation=False,
            dropout=dropout,
        )
        super().__init__(conv1, conv2)
