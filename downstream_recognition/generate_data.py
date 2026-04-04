"""
Radar Working Mode Recognition — Data Generation
==================================================
Use the existing Radar_words functions to generate labeled PRI sequences.

Modulation labels (4-class):
  0  sliding        (word_num 12–15)
  1  stagger         (word_num 16–21)
  2  dwell_switch    (word_num 22–28)
  3  complex_dwell   (word_num 29)
"""
from __future__ import annotations

import os
import sys
import random
from pathlib import Path
from typing import List, Tuple

import torch
import numpy as np

# Allow importing from parent + dataset folder (Radar_words.py uses 'from Modulation_types import ...')
_PARENT = str(Path(__file__).resolve().parent.parent)
_DATASET = str(Path(__file__).resolve().parent.parent / 'dataset')
for _p in (_PARENT, _DATASET):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dataset.Radar_words import (
    word_1_15, word_16, word_17, word_18, word_19, word_20, word_21,
    word_22_28, word_29,
)

# ── label mapping ────────────────────────────────────────────────────
LABEL_NAMES = ['sliding', 'stagger', 'dwell_switch', 'complex_dwell']

_WORD_NUM_TO_LABEL = {}
for wn in range(12, 16):
    _WORD_NUM_TO_LABEL[wn] = 0   # sliding
for wn in range(16, 22):
    _WORD_NUM_TO_LABEL[wn] = 1   # stagger
for wn in range(22, 29):
    _WORD_NUM_TO_LABEL[wn] = 2   # dwell_switch
_WORD_NUM_TO_LABEL[29] = 3       # complex_dwell

_WORD_FUNCS = {
    12: word_1_15, 13: word_1_15, 14: word_1_15, 15: word_1_15,
    16: word_16, 17: word_17, 18: word_18, 19: word_19,
    20: word_20, 21: word_21,
    22: word_22_28, 23: word_22_28, 24: word_22_28, 25: word_22_28,
    26: word_22_28, 27: word_22_28, 28: word_22_28,
    29: word_29,
}


def  generate_samples(
    n_per_class: int = 2000,
    seq_len: int = 100,
    seed: int = 42,
) -> Tuple[List[torch.Tensor], List[int]]:
    """Return (sequences, labels).  Each sequence is 1-D PRI float tensor."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    sequences: List[torch.Tensor] = []
    labels: List[int] = []

    # Distribute n_per_class equally across word_nums of each class
    for label_id in range(len(LABEL_NAMES)):
        word_nums = [wn for wn, lb in _WORD_NUM_TO_LABEL.items() if lb == label_id]
        per_word = max(1, n_per_class // len(word_nums))
        for wn in word_nums:
            func = _WORD_FUNCS[wn]
            for _ in range(per_word):
                mfr, _, _ = func(if_add_noise=False, seq_len=seq_len)
                pri = mfr[:, 0]  # PRI channel
                sequences.append(pri)
                labels.append(label_id)

    # Shuffle
    idx = list(range(len(sequences)))
    random.shuffle(idx)
    sequences = [sequences[i] for i in idx]
    labels = [labels[i] for i in idx]

    return sequences, labels


def collate_sequences(
    sequences: List[torch.Tensor],
    labels: List[int],
    max_len: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad sequences to max_len, return (X [N, L], lengths [N], y [N])."""
    if max_len is None:
        max_len = max(len(s) for s in sequences)
    N = len(sequences)
    X = torch.zeros(N, max_len)
    lengths = torch.zeros(N, dtype=torch.long)
    for i, s in enumerate(sequences):
        L = min(len(s), max_len)
        X[i, :L] = s[:L]
        lengths[i] = L
    y = torch.tensor(labels, dtype=torch.long)
    return X, lengths, y


if __name__ == '__main__':
    seqs, labs = generate_samples(n_per_class=500, seq_len=100)
    X, lengths, y = collate_sequences(seqs, labs)
    print(f'Generated {X.shape[0]} samples, max_len={X.shape[1]}')
    for i, name in enumerate(LABEL_NAMES):
        print(f'  class {i} ({name}): {(y == i).sum().item()} samples')
