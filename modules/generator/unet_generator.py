"""The UNET generator module."""

import torch
from torch import nn

from modules.generator.unet_frame_decoder import UNetFrameDecoder
from modules.generator.unet_frame_encoder import UNetFrameEncoder


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
        )
        self.noise_dim = noise_dim

    def forward(
        self, inputs: torch.Tensor, noise_vector: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass."""
        features = self.encoder(inputs)
        if noise_vector is None:
            noise_vector = torch.randn((inputs.shape[0], self.noise_dim), device=inputs.device)
        noise_vector = noise_vector[..., None, None].expand(
            -1, -1, features[-1].shape[-2], features[-1].shape[-1]
        )
        features[-1] = torch.concat([features[-1], noise_vector], dim=1)
        return self.decoder(features)
