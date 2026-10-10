"""Autoencoder architecture for anomaly detection."""

from __future__ import annotations

import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    """Symmetric MLP autoencoder with a tight bottleneck.

    The bottleneck forces the network to learn the normal-data manifold;
    anomalies then produce large reconstruction errors.
    """

    def __init__(self, input_dim: int = 12, latent_dim: int = 3,
                 hidden: tuple[int, ...] = (64, 32), dropout: float = 0.1):
        super().__init__()
        enc_layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden:
            enc_layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        enc_layers.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers: list[nn.Module] = []
        prev = latent_dim
        for h in reversed(hidden):
            dec_layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    @torch.no_grad()
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample mean squared reconstruction error."""
        return ((self(x) - x) ** 2).mean(dim=1)
