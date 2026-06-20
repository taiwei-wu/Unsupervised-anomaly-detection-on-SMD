from __future__ import annotations

import torch
from torch import nn


class ConvWindowAutoencoder(nn.Module):
    """Convolutional autoencoder for fixed-length multivariate windows."""

    def __init__(self, n_channels: int, hidden_channels: int = 128) -> None:
        super().__init__()
        mid = max(hidden_channels // 2, 32)
        self.encoder = nn.Sequential(
            nn.Conv1d(n_channels, mid, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(mid, hidden_channels, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(hidden_channels, mid, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(mid, n_channels, kernel_size=5, padding=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input and output use (batch, window, channel) for consistency with numpy windows.
        z = self.encoder(x.transpose(1, 2))
        y = self.decoder(z).transpose(1, 2)
        if y.shape[1] != x.shape[1]:
            y = y[:, : x.shape[1], :]
        return y

