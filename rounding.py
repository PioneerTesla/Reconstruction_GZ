from __future__ import annotations

import torch


def nearest_token_ids(embedding: torch.nn.Embedding, hidden_states: torch.Tensor) -> torch.Tensor:
    """
    hidden_states: [B, L, D]
    return: [B, L]
    """
    weight = embedding.weight
    flat = hidden_states.reshape(-1, hidden_states.size(-1))
    h2 = (flat ** 2).sum(dim=-1, keepdim=True)
    w2 = (weight ** 2).sum(dim=-1).unsqueeze(0)
    dist = h2 + w2 - 2 * flat @ weight.t()
    ids = dist.argmin(dim=-1)
    return ids.view(hidden_states.size(0), hidden_states.size(1))


def round_hidden_states(embedding: torch.nn.Embedding, hidden_states: torch.Tensor) -> torch.Tensor:
    ids = nearest_token_ids(embedding, hidden_states)
    return embedding(ids)
