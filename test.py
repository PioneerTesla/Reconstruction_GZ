"""
Test script: load a trained model, run inference on the test split, and save
per-sample reconstruction results to a folder for downstream analysis.

Each result file is named with the word type for easy identification, e.g.:
    word1_12_idx0042.pt

Saved dict per sample:
    {
        'word_type': str,           # e.g. 'word1_12'
        'filename': str,            # original filename, e.g. 'word1_12_1016.pt'
        'clean_pri': list[float],   # ground-truth target PRI values
        'pred_pri': list[float],    # reconstructed PRI values
        'src_pri': list[float],     # observed (corrupted) PRI values
        'clean_ids': list[int],     # ground-truth target token ids
        'pred_ids': list[int],      # predicted target token ids
        'exact_acc': float,         # per-sample exact accuracy
        'mae': float,               # per-sample MAE
    }

Usage:
    python test.py                           # use config.json defaults
    python test.py --model_path CheckPoint/DiffSeqPRI_Miss_dataset/best_model.pth
    python test.py --root dataset --scene Miss --output_dir test_results
"""
from __future__ import annotations

import os
import re
import json
import random
import argparse
from typing import List, Optional, Tuple

import torch
import numpy as np

from pri_tokenizer import PRIQuantizer
from pri_dataset import PRISample, PRIDiffuSeqDataset, PRICollator
from model import PRIDiffuSeq, PRIDiffuSeqConfig
from utils import (
    load_defaults_config,
    set_seed,
    resolve_device,
    normalize_optional_float,
    normalize_optional_int,
)
from data_loader import build_quantizer, extract_word_type
from evaluation import parse_target_tokens, parse_gt_target_tokens, extract_target_pri


def build_test_loader(
    scene: str,
    seq_len: int,
    bs: int,
    quantizer: PRIQuantizer,
    root: str = 'dataset',
    seed: int = 42,
    max_test_samples: Optional[int] = None,
):
    """Build the test DataLoader and return (loader, test_filenames, test_word_types).

    Uses the same 90/10 split as training to ensure test set consistency.
    """
    samples = []
    filenames: List[str] = []
    clean_data_path = os.path.join(root, 'Ground_Truth')
    obs_data_path = os.path.join(root, scene)

    for file in sorted(os.listdir(clean_data_path)):
        if not file.endswith('.pt'):
            continue
        clean = torch.load(os.path.join(clean_data_path, file), weights_only=False)['seq'].tolist()
        obs = torch.load(os.path.join(obs_data_path, file), weights_only=False)['seq'].tolist()
        samples.append(PRISample(observed_pri=obs, clean_pri=clean))
        filenames.append(file)

    n_samples = len(samples)
    all_nums = list(range(n_samples))
    random.seed(seed)
    random.shuffle(all_nums)

    test_indices = all_nums[int(0.9 * n_samples):]
    test_samples = [samples[i] for i in test_indices]
    test_filenames = [filenames[i] for i in test_indices]
    test_word_types = [extract_word_type(filenames[i]) for i in test_indices]

    if max_test_samples is not None:
        test_samples = test_samples[:max_test_samples]
        test_filenames = test_filenames[:max_test_samples]
        test_word_types = test_word_types[:max_test_samples]

    test_ds = PRIDiffuSeqDataset(test_samples, quantizer=quantizer, seq_len=seq_len)
    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, collate_fn=PRICollator())

    return test_loader, test_filenames, test_word_types


def main():
    parser = argparse.ArgumentParser(description='Test a trained PRI reconstruction model.')
    parser.add_argument('--config', type=str, default='config.json', help='Path to config.json')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to trained model checkpoint. If not set, uses best_model.pth from config.')
    parser.add_argument('--ema_path', type=str, default=None,
                        help='Path to EMA checkpoint. If set, use EMA weights for inference.')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save results. Default: test_results/<model_name>')
    parser.add_argument('--root', type=str, default=None)
    parser.add_argument('--scene', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--max_test_samples', type=int, default=None)
    args = parser.parse_args()

    cfg_dict = load_defaults_config(args.config)
    root = args.root or cfg_dict.get('root', 'dataset')
    scene = args.scene or cfg_dict.get('scene', 'Miss')
    batch_size = args.batch_size or cfg_dict.get('batch_size', 64)
    seq_len = cfg_dict.get('spliced_seq_length', 220)
    seed = cfg_dict.get('seed', 42)
    sampling_method = cfg_dict.get('sampling_method', 'ddpm')
    ddim_steps = int(cfg_dict.get('ddim_steps', 5))
    ddim_eta = float(cfg_dict.get('ddim_eta', 0.0))
    diff_steps = cfg_dict.get('diff_steps', 50)
    noise_schedule = cfg_dict.get('noise_schedule', 'cosine')
    ce_weight = float(cfg_dict.get('ce_weight', 0.5))
    device_name = resolve_device(cfg_dict.get('device', 'cpu'))
    max_test_samples = args.max_test_samples or normalize_optional_int(cfg_dict.get('max_test_samples', None))

    quantizer_mode = cfg_dict.get('quantizer_mode', 'uniform')
    quantizer_bins = int(cfg_dict.get('quantizer_bins', 1000))
    quantizer_min_v = normalize_optional_float(cfg_dict.get('quantizer_min', None))
    quantizer_max_v = normalize_optional_float(cfg_dict.get('quantizer_max', None))

    set_seed(seed)

    # Build quantizer
    clean_path = os.path.join(root, 'Ground_Truth')
    obs_path = os.path.join(root, scene)
    quantizer, q_min, q_max, actual_bins = build_quantizer(
        quantizer_mode=quantizer_mode,
        quantizer_bins=quantizer_bins,
        quantizer_min=quantizer_min_v,
        quantizer_max=quantizer_max_v,
        clean_path=clean_path,
        obs_path=obs_path,
    )

    # Build test loader
    test_loader, test_filenames, test_word_types = build_test_loader(
        scene=scene,
        seq_len=seq_len,
        bs=batch_size,
        quantizer=quantizer,
        root=root,
        seed=seed,
        max_test_samples=max_test_samples,
    )

    # Build model
    model_cfg = PRIDiffuSeqConfig(
        vocab_size=quantizer.vocab_size,
        seq_len=seq_len,
        hidden_dim=128,
        model_dim=256,
        diffusion_steps=diff_steps,
        beta_schedule=noise_schedule,
        ce_weight=ce_weight,
        device=device_name,
    )
    model = PRIDiffuSeq(model_cfg)

    # Determine model path
    model_name = cfg_dict.get('model_name', 'DiffSeqPRI')
    if args.model_path:
        model_path = args.model_path
    else:
        model_path = os.path.join('CheckPoint', model_name, 'best_model.pth')

    print(f'Loading model from: {model_path}')
    state_dict = torch.load(model_path, map_location=model.device, weights_only=False)
    model.load_state_dict(state_dict)

    # Optionally load EMA weights
    if args.ema_path and os.path.exists(args.ema_path):
        print(f'Loading EMA weights from: {args.ema_path}')
        from model import EMAModel
        ema = EMAModel(model)
        ema.load_state_dict(torch.load(args.ema_path, map_location=model.device, weights_only=False))
        ema.apply_shadow(model)

    # Output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join('test_results', f'{model_name}_{scene}_{os.path.basename(root)}')
    os.makedirs(output_dir, exist_ok=True)

    print(f'Scene: {scene}, Root: {root}')
    print(f'Test samples: {len(test_filenames)}')
    print(f'Output dir: {output_dir}')
    print(f'Sampling: {sampling_method}, diff_steps={diff_steps}, ddim_steps={ddim_steps}')
    print()

    # Run inference
    model.eval()
    global_idx = 0
    all_results = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(model.device)
            input_mask = batch['input_mask'].to(model.device)

            out = model.reconstruct(
                input_ids, input_mask,
                use_rounding=True,
                sampling_method=sampling_method,
                ddim_steps=ddim_steps,
                ddim_eta=ddim_eta,
                test_mode=True,
            )
            pred_ids = out['pred_ids']
            B = input_ids.size(0)

            for i in range(B):
                if global_idx >= len(test_filenames):
                    break

                fname = test_filenames[global_idx]
                wt = test_word_types[global_idx]

                # Extract PRIs
                clean_pri, pred_pri, src_pri = extract_target_pri(
                    input_ids[i], input_mask[i], pred_ids[i], quantizer,
                    test_mode=True,
                )

                # Compute per-sample metrics
                gt_tok = parse_gt_target_tokens(input_ids[i], input_mask[i], quantizer)
                pred_tok = parse_target_tokens(pred_ids[i], quantizer)

                min_l = min(len(gt_tok), len(pred_tok))
                max_l = max(len(gt_tok), len(pred_tok))
                exact_match = sum(1 for g, p in zip(gt_tok[:min_l], pred_tok[:min_l]) if g == p)
                exact_acc = exact_match / max_l if max_l > 0 else 0.0

                mv = min(len(clean_pri), len(pred_pri))
                mae = sum(abs(g - p) for g, p in zip(clean_pri[:mv], pred_pri[:mv])) / mv if mv > 0 else 0.0

                result = {
                    'word_type': wt,
                    'filename': fname,
                    'clean_pri': clean_pri,
                    'pred_pri': pred_pri,
                    'src_pri': src_pri,
                    'clean_ids': gt_tok,
                    'pred_ids': pred_tok,
                    'exact_acc': exact_acc,
                    'mae': mae,
                }

                # Save individual result file: <word_type>_idx<NNNN>.pt
                save_name = f'{wt}_idx{global_idx:04d}.pt'
                torch.save(result, os.path.join(output_dir, save_name))
                all_results.append(result)

                global_idx += 1

    # Print summary
    print(f'\n=== Test Results Summary ({len(all_results)} samples) ===')
    if all_results:
        avg_acc = np.mean([r['exact_acc'] for r in all_results])
        avg_mae = np.mean([r['mae'] for r in all_results])
        print(f'  Average exact_acc = {avg_acc:.4%}')
        print(f'  Average MAE       = {avg_mae:.4f}')

        # Per word-type summary
        from collections import defaultdict
        wt_stats = defaultdict(lambda: {'accs': [], 'maes': []})
        for r in all_results:
            wt_stats[r['word_type']]['accs'].append(r['exact_acc'])
            wt_stats[r['word_type']]['maes'].append(r['mae'])

        print(f'\n  Per word-type breakdown:')
        print(f'  {"Word Type":<15} {"Count":>6} {"Avg Acc":>10} {"Avg MAE":>10}')
        print(f'  {"-"*43}')
        for wt in sorted(wt_stats.keys()):
            s = wt_stats[wt]
            print(f'  {wt:<15} {len(s["accs"]):>6} {np.mean(s["accs"]):>10.4%} {np.mean(s["maes"]):>10.4f}')

    # Save summary JSON
    summary = {
        'model_path': model_path,
        'scene': scene,
        'root': root,
        'num_samples': len(all_results),
        'avg_exact_acc': float(np.mean([r['exact_acc'] for r in all_results])) if all_results else 0.0,
        'avg_mae': float(np.mean([r['mae'] for r in all_results])) if all_results else 0.0,
    }
    summary_path = os.path.join(output_dir, 'summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    print(f'\nSummary saved to: {summary_path}')
    print(f'Individual results saved to: {output_dir}/')


if __name__ == '__main__':
    main()
