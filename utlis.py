import json
import argparse
import os
import torch
from typing import List, Tuple


def load_defaults_config(path='dataset/dataset.json'):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}



def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


def create_argparser(path='dataset/dataset.json'):
    defaults = dict()
    defaults.update(load_defaults_config(path))
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def get_ideal_pri_prototypes(clean_path: str = 'dataset/Ground_Truth'):
    prototypes: List[float] = []
    for file in os.listdir(clean_path):
        if file.endswith('.pt'):
            seq = torch.load(os.path.join(clean_path, file), weights_only=False)['seq']
            uniques = torch.unique(torch.round(seq)).tolist()
            prototypes.extend(uniques)
    return list(set(prototypes))


def get_pri_range(data_path: str) -> Tuple[float, float]:
    min_value = None
    max_value = None
    for file in os.listdir(data_path):
        if not file.endswith('.pt'):
            continue
        seq = torch.load(os.path.join(data_path, file), weights_only=False)['seq'].float()
        seq_min = float(seq.min().item())
        seq_max = float(seq.max().item())
        min_value = seq_min if min_value is None else min(min_value, seq_min)
        max_value = seq_max if max_value is None else max(max_value, seq_max)

    if min_value is None or max_value is None:
        raise ValueError(f'No valid .pt PRI files found in {data_path}')
    return float(min_value), float(max_value)


def get_clean_pri_range(clean_path: str = 'dataset/Ground_Truth') -> Tuple[float, float]:
    return get_pri_range(clean_path)


def choose_training_schedule(optimizer: torch.optim.Optimizer, scheduler_name: str = 'ReduceLROnPlateau'):
    if scheduler_name == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)
    elif scheduler_name == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=10)
    elif scheduler_name == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6, last_epoch=19)
    elif scheduler_name == 'Mix':
        warmup_scheduler = torch.optim.lr_scheduler.ConstantLR(
            optimizer=optimizer,
            factor=1.0,
            total_iters=20,
        )
        anneal_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=30,
            eta_min=1e-6,
        )
        final_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=5,
            gamma=0.95,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer=optimizer,
            schedulers=[warmup_scheduler, anneal_scheduler, final_scheduler],
            milestones=[20, 335],
        )
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")
    return scheduler
