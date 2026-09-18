"""Module containing a minibatch discrimination module."""

from __future__ import annotations

import torch
from torch import nn


class MiniBatchDiscrimination(nn.Module):
    """Minibatch discrimination module."""

    def __init__(self, in_features: int, out_features: int, kernel_dims: int) -> None:
        """Minibatch discrimination module.

        See https://arxiv.org/pdf/1606.03498.

        Args:
        ----
            in_features: number of input features
            out_features: number of kernels (output features)
            kernel_dims: dimension for each kernel

        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_dims = kernel_dims

        self.T = nn.Parameter(torch.randn(in_features, out_features, kernel_dims) * 0.05)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass.

        The similarity statistic is spatial: at every cell, each sample's projected
        feature is compared to the other samples' at the same cell. In a padding cell every
        sample carries the same constant, so the statistic is ``batch_size - 1`` there
        regardless of the data; ``mask`` zeroes those cells so the padding geometry is not
        handed to the heads as a feature. Within a single-region batch the mask is the same
        for every sample, which is what keeps the comparison between like and like.

        Args:
        ----
            x: Input tensor of shape (batch_size, channels, H, W).
            mask: Binary mask of shape (batch_size, 1, H, W), or None.

        Returns:
        -------
            ``x`` with ``out_features`` similarity channels appended.

        """
        batch_size = x.size(0)
        H, W = x.size(2), x.size(3)  # noqa
        x_in = x.contiguous().transpose(3, 1)

        M = x_in @ self.T.view(self.in_features, -1)  # noqa
        M = M.view(  # noqa
            batch_size, W, H, self.out_features, self.kernel_dims
        )  # [batch_size, w, h, out_features, kernel_dims]

        # Compute L1 distance between each pair of samples
        M_expanded_1 = M.unsqueeze(0)  # [1, batch, w, h, out_features, kernel] # noqa
        M_expanded_2 = M.unsqueeze(1)  # [batch, 1, w, h, out_features, kernel] # noqa
        diff = torch.abs(M_expanded_1 - M_expanded_2).sum(5)  # [batch, batch, w, h, out_features]

        exp_neg_diff = torch.exp(-diff)

        others = 1 - torch.eye(batch_size, device=x.device)
        out = (
            (exp_neg_diff * others[..., None, None, None]).sum(1).contiguous()
        )  # [batch, w, h, out_features]
        out = out.transpose(3, 1)  # [batch, out_features, h, w]
        if mask is not None:
            out = out * mask.to(out.dtype)

        # Concatenate with original features
        return torch.cat([x, out], dim=1)
