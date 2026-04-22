"""Dataset and DataLoader construction for PRI reconstruction."""
from __future__ import annotations
import math
import os
import random
import re
from typing import List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from pri_tokenizer import PRIQuantizer, QuantizerConfig
from pri_dataset import PRISample, PRIDiffuSeqDataset, PRICollator
from utils import get_clean_pri_range, get_pri_range


_WORD_TYPE_RE = re.compile(r'(word\d+_\d+)')


def extract_word_type(filename: str) -> str:
    """'word1_12_1016.pt' -> 'word1_12'."""
    m = _WORD_TYPE_RE.match(filename)
    return m.group(1) if m else filename


def build_quantizer(
    quantizer_mode: str = 'uniform',
    quantizer_bins: int = 1000,
    quantizer_min: Optional[float] = None,
    quantizer_max: Optional[float] = None,
    clean_path: str = 'dataset/Ground_Truth',
    obs_path: str = 'dataset/Miss',
) -> Tuple[PRIQuantizer, float, float, int]:
    """Build a `PRIQuantizer`; bin count is at least (max-min+1) so every
    integer µs maps to its own bin."""
    mode = quantizer_mode.lower().strip()
    if mode != 'uniform':
        raise ValueError(f"Unsupported quantizer_mode: {quantizer_mode}")

    clean_min, clean_max = get_clean_pri_range(clean_path)
    obs_min, obs_max = get_pri_range(obs_path)
    data_min = min(clean_min, obs_min)
    data_max = max(clean_max, obs_max)

    min_value = data_min if quantizer_min is None else float(quantizer_min)
    max_value = data_max if quantizer_max is None else float(quantizer_max)

    requested_bins = int(quantizer_bins)
    if requested_bins <= 1:
        requested_bins = max(2, int(math.ceil(max_value - min_value)) + 1)
    unit_bins = int(math.ceil(max_value - min_value)) + 1
    bins = max(requested_bins, unit_bins)

    quantizer = PRIQuantizer(QuantizerConfig(
        mode='uniform',
        min_value=min_value,
        max_value=max_value,
        num_bins=bins,
        add_special_tokens=True,
        snap_tolerance=None,
        key_start=1,
    ))
    return quantizer, min_value, max_value, bins


def build_demo_loader(
    scene: str = 'Miss',
    seq_len: int = 250,
    bs: int = 64,
    quantizer_mode: str = 'uniform',
    quantizer_bins: int = 1000,
    quantizer_min: Optional[float] = None,
    quantizer_max: Optional[float] = None,
    max_train_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
    root: str = 'dataset',
):
    """Read paired clean/observed .pt files, split 90/10 train/test, and build
    DataLoaders + quantizer. Returns (train_loader, test_loader, quantizer,
    min_value, max_value, actual_bins, test_word_types)."""
    clean_path = os.path.join(root, 'Ground_Truth')
    obs_path = os.path.join(root, scene)

    samples: List[PRISample] = []
    filenames: List[str] = []
    for file in sorted(os.listdir(clean_path)):
        if not file.endswith('.pt'):
            continue
        clean = torch.load(os.path.join(clean_path, file), weights_only=False)['seq'].tolist()
        obs = torch.load(os.path.join(obs_path, file), weights_only=False)['seq'].tolist()
        samples.append(PRISample(observed_pri=obs, clean_pri=clean))
        filenames.append(file)

    n = len(samples)
    idx = list(range(n))
    random.shuffle(idx)
    split = int(0.9 * n)

    train_samples = [samples[i] for i in idx[:split]]
    test_samples = [samples[i] for i in idx[split:]]
    test_word_types = [extract_word_type(filenames[i]) for i in idx[split:]]

    if max_train_samples is not None:
        train_samples = train_samples[:max_train_samples]
    if max_test_samples is not None:
        test_samples = test_samples[:max_test_samples]
        test_word_types = test_word_types[:max_test_samples]

    quantizer, min_value, max_value, actual_bins = build_quantizer(
        quantizer_mode=quantizer_mode,
        quantizer_bins=quantizer_bins,
        quantizer_min=quantizer_min,
        quantizer_max=quantizer_max,
        clean_path=clean_path,
        obs_path=obs_path,
    )

    train_loader = DataLoader(
        PRIDiffuSeqDataset(train_samples, quantizer=quantizer, seq_len=seq_len),
        batch_size=bs, shuffle=True, collate_fn=PRICollator(),
    )
    test_loader = DataLoader(
        PRIDiffuSeqDataset(test_samples, quantizer=quantizer, seq_len=seq_len),
        batch_size=bs, shuffle=False, collate_fn=PRICollator(),
    )
    return train_loader, test_loader, quantizer, min_value, max_value, actual_bins, test_word_types
