"""
Denoising Autoencoder for PRI Reconstruction
==============================================
Based on: Tang & Wei, "Signal Separation Method for Radiation Sources
Based on a Parallel Denoising Autoencoder", Electronics, 2024.

Adapted for the PRI sequence reconstruction task:
  - Input:  normalised degraded PRI values (0-padded to max_len)
  - Output: reconstructed clean PRI values (0-padded to max_len)
  - Architecture: fully-connected encoder → bottleneck → decoder
    (no recurrence, no attention — a distinct paradigm from RNN/Diffusion)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DenoisingAutoencoder(nn.Module):
    """FC-only Denoising Autoencoder for PRI sequence reconstruction.

    The encoder compresses the padded PRI input into a low-dimensional
    latent code; the decoder expands it back to reconstruct the clean
    PRI sequence.  A separate head predicts the output sequence length.

    Parameters
    ----------
    max_len : int
        Fixed input / output dimension (max sequence length).
    hidden_dims : list[int]
        Sizes of the encoder hidden layers (decoder mirrors them).
    latent_dim : int
        Bottleneck dimension.
    dropout : float
        Dropout rate applied after each hidden layer.
    scale : float
        Normalisation constant (typically max PRI in µs, e.g. 1500).
    """

    def __init__(
        self,
        max_len: int,
        hidden_dims: list[int] | None = None,
        latent_dim: int = 128,
        dropout: float = 0.1,
        scale: float = 1500.0,
    ):
        super().__init__()
        self.max_len = max_len
        self.scale = scale

        if hidden_dims is None:
            hidden_dims = [512, 256]

        # ---------- Encoder ----------
        enc_layers: list[nn.Module] = []
        in_dim = max_len
        for h in hidden_dims:
            enc_layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = h
        enc_layers.extend([nn.Linear(in_dim, latent_dim), nn.ReLU()])
        self.encoder = nn.Sequential(*enc_layers)

        # ---------- Decoder ----------
        dec_layers: list[nn.Module] = []
        in_dim = latent_dim
        for h in reversed(hidden_dims):
            dec_layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = h
        # Final layer — no activation (regression output)
        dec_layers.append(nn.Linear(in_dim, max_len))
        self.decoder = nn.Sequential(*dec_layers)

        # ---------- Length prediction head ----------
        self.len_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : Tensor [B, max_len]
            Normalised degraded PRI values (divided by *scale*), 0-padded.

        Returns
        -------
        recon : Tensor [B, max_len]
            Reconstructed PRI values (normalised).
        pred_len : Tensor [B]
            Predicted output sequence length (continuous, clipped at eval).
        """
        z = self.encoder(x)           # [B, latent_dim]
        recon = self.decoder(z)        # [B, max_len]
        pred_len = self.len_head(z).squeeze(-1)  # [B]
        return recon, pred_len

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Inference helper.

        Returns
        -------
        values : Tensor [B, max_len]   — reconstructed PRI values in µs
        lengths : Tensor [B]           — predicted integer lengths
        """
        self.eval()
        recon, pred_len = self.forward(x)
        values = recon * self.scale             # de-normalise
        lengths = pred_len.round().clamp(min=1, max=self.max_len).long()
        return values, lengths


def build_model(
    max_len: int = 120,
    hidden_dims: list[int] | None = None,
    latent_dim: int = 128,
    dropout: float = 0.1,
    scale: float = 1500.0,
    device: str = 'cpu',
) -> DenoisingAutoencoder:
    model = DenoisingAutoencoder(
        max_len=max_len,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        dropout=dropout,
        scale=scale,
    )
    return model.to(device)
