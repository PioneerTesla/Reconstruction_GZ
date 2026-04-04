"""
Batch runner for train_miss_pri.py with multiple parameter configurations.

Usage examples:
    # Single run with defaults from config.json
    python run_file.py

    # Specify root and scene
    python run_file.py --root dataset --scene Miss

    # Batch: multiple roots × scenes
    python run_file.py --root dataset dataset_random_len --scene Miss Spurious Mix

    # Override any config.json parameter
    python run_file.py --root dataset --scene Miss --epochs 500 --learning_rate 0.0003
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import itertools
from typing import List


CONFIG_PATH = 'config.json'
TRAIN_SCRIPT = 'train_miss_pri.py'


def load_base_config() -> dict:
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_config(cfg: dict, path: str = CONFIG_PATH) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=4, ensure_ascii=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Batch runner: train_miss_pri.py across multiple (root, scene) combinations.',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('--root', nargs='+', default=None,
                        help='One or more dataset root directories. E.g. --root dataset dataset_random_len')
    parser.add_argument('--scene', nargs='+', default=None,
                        help='One or more scene types. E.g. --scene Miss Spurious Mix')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--diff_steps', type=int, default=None)
    parser.add_argument('--noise_schedule', type=str, default=None)
    parser.add_argument('--sampling_method', type=str, default=None)
    parser.add_argument('--ddim_steps', type=int, default=None)
    parser.add_argument('--ddim_eta', type=float, default=None)
    parser.add_argument('--ce_weight', type=float, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--spliced_seq_length', type=int, default=None)
    parser.add_argument('--quantizer_bins', type=int, default=None)
    parser.add_argument('--quantizer_min', type=float, default=None)
    parser.add_argument('--quantizer_max', type=float, default=None)
    parser.add_argument('--max_train_samples', type=int, default=None)
    parser.add_argument('--max_test_samples', type=int, default=None)
    parser.add_argument('--model_name', type=str, default=None,
                        help='Override model_name. If not set, auto-generated as DiffSeqPRI_{scene}_{root}')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print configurations without running training.')
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    base_cfg = load_base_config()

    roots: List[str] = args.root if args.root else [base_cfg.get('root', 'dataset')]
    scenes: List[str] = args.scene if args.scene else [base_cfg.get('scene', 'Miss')]

    # Collect overrides (non-batch, non-list params)
    override_keys = [
        'epochs', 'batch_size', 'learning_rate', 'diff_steps', 'noise_schedule',
        'sampling_method', 'ddim_steps', 'ddim_eta', 'ce_weight', 'device', 'seed',
        'spliced_seq_length', 'quantizer_bins', 'quantizer_min', 'quantizer_max',
        'max_train_samples', 'max_test_samples',
    ]
    overrides = {}
    for key in override_keys:
        val = getattr(args, key, None)
        if val is not None:
            overrides[key] = val

    combos = list(itertools.product(roots, scenes))
    print(f'=== Batch Runner: {len(combos)} configuration(s) ===')
    for i, (root, scene) in enumerate(combos):
        print(f'  [{i+1}/{len(combos)}] root={root}, scene={scene}')
    print()

    # Save original config to restore later
    original_cfg = base_cfg.copy()

    for i, (root, scene) in enumerate(combos):
        run_cfg = base_cfg.copy()
        run_cfg['root'] = root
        run_cfg['scene'] = scene
        run_cfg.update(overrides)

        if args.model_name:
            run_cfg['model_name'] = args.model_name
        else:
            safe_root = os.path.basename(root.rstrip('/\\')) or root
            run_cfg['model_name'] = f'DiffSeqPRI_{scene}_{safe_root}'

        print(f'{"="*60}')
        print(f'[{i+1}/{len(combos)}] Starting: root={root}, scene={scene}, model_name={run_cfg["model_name"]}')
        print(f'{"="*60}')

        if args.dry_run:
            print(json.dumps(run_cfg, indent=4, ensure_ascii=False))
            print()
            continue

        # Write config and run training
        save_config(run_cfg)
        ret = subprocess.run(
            [sys.executable, TRAIN_SCRIPT],
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        if ret.returncode != 0:
            print(f'[WARNING] Run [{i+1}/{len(combos)}] exited with code {ret.returncode}')
        else:
            print(f'[OK] Run [{i+1}/{len(combos)}] completed successfully.')
        print()

    # Restore original config
    save_config(original_cfg)
    print('=== All runs finished. Config restored to original. ===')


if __name__ == '__main__':
    main()
