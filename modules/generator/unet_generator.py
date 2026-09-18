"""The UNET generator module."""

from __future__ import annotations

import torch
from torch import nn

from modules.generator.unet_frame_decoder import UNetFrameDecoder
from modules.generator.unet_frame_encoder import UNetFrameEncoder
from modules.masking import binary


class UNetGenerator(nn.Module):
    """The UNet applied to each map individually."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        noise_dim: int,
        encoder_channels: list[int],
        decoder_channels: list[int],
        dropout: float,
        normalization: str | None = "batchnorm",
        apply_center_block: bool = False,
        noise_weight_init: str = "randn",
        coherent_noise_init: str | float | None = None,
        encoder_late_dropout: float = 0.0,
    ) -> None:
        """Initialize arguments."""
        super().__init__()
        self.encoder = UNetFrameEncoder(
            in_channels=input_dim,
            out_channels=encoder_channels,
            dropout=dropout,
            normalization=normalization,
            apply_center_block=apply_center_block,
            noise_weight_init=noise_weight_init,
            coherent_noise_init=coherent_noise_init,
            encoder_late_dropout=encoder_late_dropout,
        )

        self.decoder = UNetFrameDecoder(
            output_dim=output_dim,
            head_channels=encoder_channels[-1] + noise_dim,
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            dropout=dropout,
            normalization=normalization,
            noise_weight_init=noise_weight_init,
            coherent_noise_init=coherent_noise_init,
        )
        self.noise_dim = noise_dim

    def forward(
        self,
        inputs: torch.Tensor,
        noise_vector: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
        ----
            inputs: The stacked input channels of shape (batch_size, C, H, W), already
                padded to the canvas the generator works on.
            noise_vector: The latent vector, of shape (batch_size, noise_dim). Drawn from
                the standard normal when None.
            mask: Region mask of shape (batch_size, 1, H, W) with values in {0, 1}, on the
                same canvas as ``inputs``. Given one, every normalization takes its
                statistics over the region's cells, every scSE channel gate pools over them,
                and every block re-zeroes its padding. None reproduces rounds 1-12 exactly.

        Returns:
        -------
            A Tensor of shape (batch_size, output_dim, H, W).

        """
        if mask is not None:
            # Level-0 re-zeroing. The maps arrive zero outside the region already, but the
            # timestamp embedding is a constant expanded over the whole canvas, so without
            # this the very first 3x3 kernel reads that constant on every coastal cell.
            inputs = inputs * mask.to(inputs.dtype)
        features, levels = self.encoder(inputs, mask)
        if noise_vector is None:
            noise_vector = torch.randn((inputs.shape[0], self.noise_dim), device=inputs.device)
        noise_vector = noise_vector[..., None, None].expand(
            -1, -1, features[-1].shape[-2], features[-1].shape[-1]
        )
        features[-1] = torch.concat([features[-1], noise_vector], dim=1)
        if levels is not None:
            # The latent is spatially constant, so it also fills the padding of the coarsest
            # level; zero it there to keep "the decoder is only ever fed the region" exact.
            features[-1] = features[-1] * binary(levels[-1])
        return self.decoder(features, levels)
