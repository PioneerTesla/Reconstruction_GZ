"""
Radar Working Mode Recognition — Model
========================================
Multi-Scale 1D-CNN + Transformer Encoder classifier.

Architecture (inspired by recent radar emitter recognition literature):
  Input PRI [B, 1, L]
       │
  ┌────┴────┐
  │ MS-CNN  │  Multi-scale 1D convolutions (kernel 3, 5, 7)
  └────┬────┘
       │ concat → [B, C, L']
  ┌────┴────────────┐
  │ Transformer Enc │  2-layer self-attention on temporal positions
  └────┬────────────┘
       │ global avg pool → [B, C]
  ┌────┴────┐
  │   MLP   │  → num_classes
  └─────────┘

Reference style: Multi-scale CNN + self-attention for radar signal
recognition (IEEE JSTSP / Radar Conference 2023-2024 pattern).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleCNN(nn.Module):
    """Parallel 1-D conv branches with different kernel sizes."""

    def __init__(self, in_channels: int = 1, branch_channels: int = 64,
                 kernel_sizes=(3, 5, 7)):
        super().__init__()
        self.branches = nn.ModuleList()
        for k in kernel_sizes:
            self.branches.append(nn.Sequential(
                nn.Conv1d(in_channels, branch_channels, k, padding=k // 2),
                nn.BatchNorm1d(branch_channels),
                nn.GELU(),
                nn.Conv1d(branch_channels, branch_channels, k, padding=k // 2),
                nn.BatchNorm1d(branch_channels),
                nn.GELU(),
            ))
        self.out_channels = branch_channels * len(kernel_sizes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [branch(x) for branch in self.branches]
        return torch.cat(outs, dim=1)  # [B, C_total, L]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class PRIModeClassifier(nn.Module):
    """1D Multi-Scale CNN + Transformer Encoder for PRI mode recognition."""

    def __init__(
        self,
        num_classes: int = 4,
        cnn_channels: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ms_cnn = MultiScaleCNN(in_channels=1, branch_channels=cnn_channels)
        d_model = self.ms_cnn.out_channels  # 192 by default

        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout, activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, L] raw PRI values
            lengths: [B] actual lengths (for masking padding)
        Returns:
            logits: [B, num_classes]
        """
        # [B, 1, L]
        h = x.unsqueeze(1)
        h = self.ms_cnn(h)          # [B, C, L]
        h = h.transpose(1, 2)       # [B, L, C]
        h = self.pos_enc(h)

        # Build padding mask
        mask = None
        if lengths is not None:
            B, L, _ = h.shape
            mask = torch.arange(L, device=h.device).unsqueeze(0) >= lengths.unsqueeze(1)

        h = self.transformer(h, src_key_padding_mask=mask)  # [B, L, C]

        # Masked global average pooling
        if lengths is not None:
            len_clamp = lengths.clamp(min=1).unsqueeze(1).unsqueeze(2).float()
            mask_expand = (~mask).unsqueeze(2).float()
            h = (h * mask_expand).sum(dim=1) / len_clamp.squeeze(2)
        else:
            h = h.mean(dim=1)

        return self.classifier(h)
