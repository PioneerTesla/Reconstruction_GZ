"""
Seq2Seq RNN with Attention for PRI Reconstruction (Continuous Regression)
==========================================================================
Based on: Yuan & Liu, "Temporal Feature Learning and Pulse Prediction
for Radars with Variable Parameters", Remote Sensing, 2022.

Adapted as an encoder-decoder seq2seq model that maps observed (degraded)
PRI value sequences to clean PRI value sequences in **continuous space**.
Quantisation is applied only at evaluation time, not inside the model.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Bidirectional LSTM encoder for continuous PRI value sequences."""

    def __init__(self, embed_dim: int, hidden_dim: int,
                 n_layers: int, dropout: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.input_proj = nn.Linear(1, embed_dim)
        self.rnn = nn.LSTM(
            embed_dim, hidden_dim, n_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.fc_h = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_c = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor):
        """
        src      : [B, src_len]  continuous PRI values (normalised)
        src_mask : [B, src_len]  1=valid, 0=padding
        Returns:
            outputs : [B, src_len, hidden_dim*2]
            hidden  : [n_layers, B, hidden_dim]
            cell    : [n_layers, B, hidden_dim]
        """
        x = src.unsqueeze(-1)  # [B, L, 1]
        embedded = self.dropout(self.input_proj(x))  # [B, L, E]

        # pack padded sequences for efficiency
        lengths = src_mask.sum(dim=1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False,
        )
        packed_out, (hidden, cell) = self.rnn(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=src.size(1),
        )

        # Merge bidirectional states per layer
        hidden = hidden.view(self.n_layers, 2, -1, self.hidden_dim)
        hidden = torch.tanh(self.fc_h(
            torch.cat([hidden[:, 0], hidden[:, 1]], dim=2)
        ))
        cell = cell.view(self.n_layers, 2, -1, self.hidden_dim)
        cell = torch.tanh(self.fc_c(
            torch.cat([cell[:, 0], cell[:, 1]], dim=2)
        ))
        return outputs, hidden, cell


class BahdanauAttention(nn.Module):
    """Additive (Bahdanau) attention."""

    def __init__(self, enc_dim: int, dec_dim: int):
        super().__init__()
        self.W_enc = nn.Linear(enc_dim, dec_dim, bias=False)
        self.W_dec = nn.Linear(dec_dim, dec_dim, bias=False)
        self.v = nn.Linear(dec_dim, 1, bias=False)

    def forward(self, dec_hidden: torch.Tensor, enc_outputs: torch.Tensor,
                mask: torch.Tensor | None = None):
        """
        dec_hidden  : [B, dec_dim]
        enc_outputs : [B, src_len, enc_dim]
        mask        : [B, src_len]  1=valid, 0=padding
        Returns:
            context : [B, enc_dim]
            weights : [B, src_len]
        """
        src_len = enc_outputs.size(1)
        dh = self.W_dec(dec_hidden).unsqueeze(1).expand(-1, src_len, -1)
        eh = self.W_enc(enc_outputs)
        energy = torch.tanh(dh + eh)
        scores = self.v(energy).squeeze(2)  # [B, src_len]
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = F.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), enc_outputs).squeeze(1)
        return context, weights


class Decoder(nn.Module):
    """LSTM decoder with attention — outputs continuous PRI value + stop logit."""

    def __init__(self, embed_dim: int, hidden_dim: int,
                 enc_hidden_dim: int, n_layers: int, dropout: float):
        super().__init__()
        self.input_proj = nn.Linear(1, embed_dim)
        self.attention = BahdanauAttention(enc_hidden_dim, hidden_dim)
        self.rnn = nn.LSTM(
            embed_dim + enc_hidden_dim, hidden_dim, n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        feat_dim = hidden_dim + enc_hidden_dim + embed_dim
        self.value_head = nn.Linear(feat_dim, 1)     # regression → 1 scalar
        self.stop_head = nn.Linear(feat_dim, 1)      # stop probability logit
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_val: torch.Tensor, hidden: torch.Tensor,
                cell: torch.Tensor, enc_outputs: torch.Tensor,
                src_mask: torch.Tensor | None = None):
        """
        input_val   : [B]  previous step PRI value (normalised scalar)
        hidden/cell : [n_layers, B, H]
        enc_outputs : [B, src_len, enc_dim]
        src_mask    : [B, src_len]
        Returns:
            pred_val    : [B]   predicted PRI value
            stop_logit  : [B]   logit for stop prediction
            hidden, cell
        """
        embedded = self.dropout(
            self.input_proj(input_val.unsqueeze(-1))  # [B, E]
        ).unsqueeze(1)  # [B, 1, E]

        context, _ = self.attention(hidden[-1], enc_outputs, src_mask)  # [B, enc_dim]
        rnn_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)  # [B,1,E+enc]
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))   # [B,1,H]

        feat = torch.cat([output.squeeze(1), context, embedded.squeeze(1)], dim=1)
        pred_val = self.value_head(feat).squeeze(-1)    # [B]
        stop_logit = self.stop_head(feat).squeeze(-1)   # [B]
        return pred_val, stop_logit, hidden, cell


class Seq2SeqPRI(nn.Module):
    """Seq2Seq model: observed PRI → clean PRI (continuous regression)."""

    def __init__(self, encoder: Encoder, decoder: Decoder,
                 scale: float, device: str):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.scale = scale    # normalisation denominator (e.g. 1500)
        self.device = device

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor,
                trg: torch.Tensor, trg_mask: torch.Tensor,
                teacher_forcing_ratio: float = 0.5):
        """
        src      : [B, src_len]  raw PRI values (µs)
        src_mask : [B, src_len]
        trg      : [B, trg_len]  raw PRI values (µs), position 0 = start sentinel (0)
        trg_mask : [B, trg_len]
        Returns:
            pred_values : [B, trg_len-1]  predicted PRI (µs)
            stop_logits : [B, trg_len-1]
        """
        B = src.size(0)
        trg_len = trg.size(1)

        src_norm = src / self.scale
        enc_outputs, hidden, cell = self.encoder(src_norm, src_mask)

        pred_values = torch.zeros(B, trg_len - 1, device=src.device)
        stop_logits = torch.zeros(B, trg_len - 1, device=src.device)

        dec_input = trg[:, 0] / self.scale  # start sentinel = 0

        for t in range(1, trg_len):
            pv, sl, hidden, cell = self.decoder(
                dec_input, hidden, cell, enc_outputs, src_mask,
            )
            pred_values[:, t - 1] = pv * self.scale
            stop_logits[:, t - 1] = sl

            if torch.rand(1).item() < teacher_forcing_ratio:
                dec_input = trg[:, t] / self.scale
            else:
                dec_input = pv.detach()

        return pred_values, stop_logits

    @torch.no_grad()
    def predict(self, src: torch.Tensor, src_mask: torch.Tensor,
                max_len: int = 120):
        """Auto-regressive decoding in continuous space.

        Returns:
            all_values : [B, max_len]  predicted PRI values (µs), 0-padded
            all_lens   : [B]  predicted sequence lengths
        """
        self.eval()
        B = src.size(0)
        src_norm = src / self.scale
        enc_outputs, hidden, cell = self.encoder(src_norm, src_mask)

        dec_input = torch.zeros(B, device=src.device)  # start sentinel
        all_values = torch.zeros(B, max_len, device=src.device)
        finished = torch.zeros(B, dtype=torch.bool, device=src.device)
        all_lens = torch.full((B,), max_len, dtype=torch.long, device=src.device)

        for t in range(max_len):
            pv, sl, hidden, cell = self.decoder(
                dec_input, hidden, cell, enc_outputs, src_mask,
            )
            val = pv * self.scale
            stop_prob = torch.sigmoid(sl)

            val = val.clamp(min=0.0)  # PRI must be non-negative
            all_values[:, t] = val

            # mark newly stopped sequences
            just_stopped = (~finished) & (stop_prob > 0.5)
            all_lens[just_stopped] = t + 1
            finished = finished | just_stopped

            if finished.all():
                break

            dec_input = pv.detach()

        return all_values, all_lens


def build_model(
    embed_dim: int = 128,
    hidden_dim: int = 256,
    n_layers: int = 2,
    dropout: float = 0.1,
    scale: float = 1500.0,
    device: str = 'cuda',
) -> Seq2SeqPRI:
    enc = Encoder(embed_dim, hidden_dim, n_layers, dropout)
    dec = Decoder(embed_dim, hidden_dim, hidden_dim * 2, n_layers, dropout)
    model = Seq2SeqPRI(enc, dec, scale, device)
    return model.to(device)
