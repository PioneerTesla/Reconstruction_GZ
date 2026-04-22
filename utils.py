"""Common utilities: config/argparse helpers, IO tee, seeding, PRI range lookup."""
from __future__ import annotations
import argparse
import json
import os
import random
import sys
from typing import List, Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Config / argparse
# ---------------------------------------------------------------------------

def load_defaults_config(path: str = 'dataset/dataset.json') -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("boolean value expected")


def add_dict_to_argparser(parser: argparse.ArgumentParser, default_dict: dict) -> None:
    for k, v in default_dict.items():
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        else:
            v_type = type(v)
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def create_argparser(path: str = 'dataset/dataset.json') -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, load_defaults_config(path))
    return parser


# ---------------------------------------------------------------------------
# Seeding / device
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> str:
    """Fall back to CPU when CUDA is requested but unavailable."""
    if device_name.startswith('cuda') and not torch.cuda.is_available():
        return 'cpu'
    return device_name


def normalize_optional_float(value):
    if value is None:
        return None
    if isinstance(value, str) and value.lower() == 'none':
        return None
    return float(value)


def normalize_optional_int(value):
    if value is None:
        return None
    if isinstance(value, str) and value.lower() == 'none':
        return None
    return int(value)


# ---------------------------------------------------------------------------
# stdout/stderr tee
# ---------------------------------------------------------------------------

class DualOutput:
    """Tee stdout/stderr to both console and a txt file."""

    def __init__(self, file_path: str, stream=None, mode: str = 'w'):
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self.console = stream if stream is not None else sys.stdout
        self.file = open(file_path, mode, encoding='utf-8', buffering=1)
        self.closed = False

    def write(self, data):
        if self.closed:
            return
        try:
            self.console.write(data)
        except UnicodeEncodeError:
            self.console.write(data.encode('utf-8', errors='replace').decode(
                self.console.encoding or 'utf-8', errors='replace'))
        self.file.write(data)

    def flush(self):
        if self.closed:
            return
        self.console.flush()
        self.file.flush()

    def close(self):
        if self.closed:
            return
        try:
            self.flush()
        finally:
            self.file.close()
            self.closed = True

    def isatty(self):
        return False


# ---------------------------------------------------------------------------
# Diffusion sampler helpers
# ---------------------------------------------------------------------------

def choose_trace_steps(diff_steps: int, ddim_steps: Optional[int], sampling_method: str) -> List[int]:
    """Return the timestep indices used by the sampler (for trace visualization)."""
    if (sampling_method.lower().strip() == 'ddim'
            and ddim_steps is not None and 0 < ddim_steps < diff_steps):
        steps = np.linspace(0, diff_steps - 1, ddim_steps, dtype=np.int64)
        return sorted(set(steps.tolist()))
    return list(range(diff_steps))


# ---------------------------------------------------------------------------
# PRI data-range helpers (scan *.pt files in a directory)
# ---------------------------------------------------------------------------

def get_pri_range(data_path: str) -> Tuple[float, float]:
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    for file in os.listdir(data_path):
        if not file.endswith('.pt'):
            continue
        seq = torch.load(os.path.join(data_path, file), weights_only=False)['seq'].float()
        seq_min, seq_max = float(seq.min().item()), float(seq.max().item())
        min_value = seq_min if min_value is None else min(min_value, seq_min)
        max_value = seq_max if max_value is None else max(max_value, seq_max)
    if min_value is None or max_value is None:
        raise ValueError(f'No valid .pt PRI files found in {data_path}')
    return min_value, max_value


def get_clean_pri_range(clean_path: str = 'dataset/Ground_Truth') -> Tuple[float, float]:
    return get_pri_range(clean_path)
