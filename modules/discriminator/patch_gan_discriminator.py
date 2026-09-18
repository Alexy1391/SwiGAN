"""Module containing the patchGAN discriminator."""

from __future__ import annotations

import torch
from torch import nn

from modules.base_conv_blocks import MaskedDiscriminatorConvBlock, single_discriminator_conv_block
from modules.masking import binary
from modules.utils import glorot_init


class PatchGANDiscriminator(nn.Module):
    """The patchGAN discriminator, scoring one tile per cell of the trunk's 2x2 windows."""

    def __init__(
        self,
        input_channels: int,
        aggregation: str = "mean",
        grid_shape: tuple[int, int] = (4, 5),
        normalization: str | None = "instancenorm",
    ) -> None:
        """Initialize the input parameters.

        Args:
        ----
            input_channels: Number of input channels.
            aggregation: How the tile grid becomes the scalar the critic loss consumes.
                "mean" returns the raw grid and leaves the reduction to the loss, which
                takes the coverage-weighted mean of Eq. 3.5 over the tiles that have data
                under them. Because the final conv is 1x1 and the WGAN loss is linear,
                that mean commutes through the conv -- mean_l(W.h_l + b) = W.mean_l(h_l) + b
                -- so the whole grid collapses to one linear readout of the masked-pooled
                features and the tiles contribute a single degree of freedom.
                "learned" reduces the grid inside the network instead, through a
                non-linear MLP over the tile scores. The critic still exposes one scalar
                function, so the Kantorovich-Rubinstein duality and the gradient penalty
                stay valid, but d(f)/d(h_l) now varies by position. Note that this makes
                the readout position-specific, which is only meaningful while every map
                puts its region at the same place on the grid.
            grid_shape: (H, W) of the tile grid produced by the final conv, only used by
                the "learned" aggregator. Derive it from the map size with
                ``mask_pyramid`` rather than assuming the (4, 5) of the 36x44 grid.
            normalization: Passed to the convolution block; see
                ``MaskedDiscriminatorConvBlock``.

        """
        super().__init__()

        if aggregation not in ("mean", "learned"):
            raise NotImplementedError(
                f"Unknown patch aggregation: {aggregation}. "
                f"Please provide one of 'mean', 'learned'."
            )
        self.aggregation = aggregation

        self.conv = MaskedDiscriminatorConvBlock(
            in_channels=input_channels,
            out_channels=input_channels * 2,
            kernel_size=2,
            dropout=0.0,
            normalization=normalization,
            padding=0,
            stride=1,
        )

        self.final_conv = single_discriminator_conv_block(
            in_channels=input_channels * 2,
            out_channels=1,
            kernel_size=1,
            dropout=0.0,
            normalization=None,
            padding=0,
            stride=1,
            activation=False,
        )

        if aggregation == "learned":
            num_tiles = grid_shape[0] * grid_shape[1]
            # Spectral norm on both layers for the same reason it is on every conv in
            # this head: without it the aggregator is an unconstrained scale knob on
            # the critic's output, which is the one thing the gradient penalty is here
            # to anchor. LeakyReLU(0.2) matches the slope used throughout the critic.
            self.aggregator = nn.Sequential(
                nn.Flatten(1),
                nn.utils.spectral_norm(nn.Linear(num_tiles, num_tiles)),
                nn.LeakyReLU(0.2),
                nn.utils.spectral_norm(nn.Linear(num_tiles, 1)),
            )

        self.apply(glorot_init)

    def tile_scores(self, inputs: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
        """Return the per-tile scores, before any aggregation.

        Kept separate from ``forward`` so the per-cell diagnostics can read the grid
        under either aggregation without disturbing the critic's own forward pass.

        Args:
        ----
            inputs: Input tensor of shape (batch_size, channels, height, width).
            weight: Coverage weights of the tile grid, shape (batch_size, 1, h, w), or None.

        Returns:
        -------
            A tensor of shape (batch_size, 1, h, w).

        """
        mask = None if weight is None else binary(weight)
        return self.final_conv(self.conv(inputs, mask))

    def forward(
        self, inputs: torch.Tensor, weight: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass.

        Args:
        ----
            inputs: Input tensor of shape (batch_size, channels, height, width).
            weight: Coverage weights of the tile grid (the last level of ``mask_pyramid``),
                shape (batch_size, 1, h, w). None means every tile has data under it.

        Returns:
        -------
            The critic score and the intermediate features. The score is the raw
            (batch_size, 1, h, w) grid under "mean" aggregation -- padding tiles included,
            the loss weights them out -- or the aggregated (batch_size, 1) scalar under
            "learned", where padding tiles are zeroed before the aggregator.

        """
        features = []
        mask = None if weight is None else binary(weight)
        out = self.conv(inputs, mask)
        features.append(out)
        tiles = self.final_conv(out)  # (B, 1, h, w)
        if self.aggregation == "mean":
            return tiles, features
        if mask is not None:
            tiles = tiles * mask
        return self.aggregator(tiles), features
