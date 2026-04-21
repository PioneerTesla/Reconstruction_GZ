"""
Run Semantic Coding baseline on the PRI reconstruction task.
=============================================================
Usage:
    cd BaseLine/SemanticCoding
    python run.py                               # uses config.json defaults
    python run.py --scene Miss --root dataset   # explicit scene & root
"""
from __future__ import annotations

import os
import sys
import json
import random
import argparse
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------- path setup (import from project root) ----------
ROOT = str(Path(__file__).resolve().parent.parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pri_tokenizer import PRIQuantizer, QuantizerConfig
from utlis import get_clean_pri_range, get_pri_range
from algorithm import SemanticCodingReconstructor

import math


# ============================================================
# Helper: build quantizer (same logic as main method)
# ============================================================
def build_quantizer(
    quantizer_bins: int = 1500,
    quantizer_min=None,
    quantizer_max=None,
    clean_path: str = 'dataset/Ground_Truth',
    obs_path: str = 'dataset/Miss',
) -> Tuple[PRIQuantizer, float, float, int]:
    clean_min, clean_max = get_clean_pri_range(clean_path)
    obs_min, obs_max = get_pri_range(obs_path)
    data_min = min(clean_min, obs_min)
    data_max = max(clean_max, obs_max)

    base_min = data_min if quantizer_min is None else float(quantizer_min)
    base_max = data_max if quantizer_max is None else float(quantizer_max)
    requested_bins = int(quantizer_bins)
    if requested_bins <= 1:
        requested_bins = max(2, int(math.ceil(base_max - base_min)) + 1)
    unit_bins = int(math.ceil(base_max - base_min)) + 1
    bins = max(requested_bins, unit_bins)

    quantizer = PRIQuantizer(QuantizerConfig(
        mode='uniform',
        min_value=base_min,
        max_value=base_max,
        num_bins=bins,
        add_special_tokens=True,
        snap_tolerance=None,
        key_start=1,
    ))
    return quantizer, base_min, base_max, bins


# ============================================================
# Data loading (same split as main method)
# ============================================================
def load_data(
    scene: str,
    root: str,
    seed: int = 42,
    max_test_samples=None,
):
    clean_dir = os.path.join(root, 'Ground_Truth')
    obs_dir = os.path.join(root, scene)

    files = sorted(f for f in os.listdir(clean_dir) if f.endswith('.pt'))
    n = len(files)
    all_idx = list(range(n))
    random.seed(seed)
    random.shuffle(all_idx)

    split = int(0.9 * n)
    test_idx = all_idx[split:]

    samples = []
    for i in test_idx:
        f = files[i]
        clean = torch.load(os.path.join(clean_dir, f), weights_only=False)['seq'].tolist()
        obs = torch.load(os.path.join(obs_dir, f), weights_only=False)['seq'].tolist()
        samples.append((obs, clean, f))

    if max_test_samples is not None:
        samples = samples[:max_test_samples]
    return samples


# ============================================================
# Metrics (same formulas as the main method's evaluation.py)
# ============================================================
def compute_metrics(
    all_gt_values: List[List[float]],
    all_pred_values: List[List[float]],
    quantizer: PRIQuantizer,
    tolerance: int = 1,
):
    total_exact = total_tol = total_tokens = 0
    total_mae = 0.0
    mae_count = 0
    len_matches = 0
    sample_count = len(all_gt_values)
    per_sample_exact: List[float] = []

    for gt_vals, pred_vals in zip(all_gt_values, all_pred_values):
        gt_tok = quantizer.encode_values(gt_vals, add_boundary_tokens=False)
        pred_tok = quantizer.encode_values(pred_vals, add_boundary_tokens=False)

        gt_len = len(gt_tok)
        pred_len = len(pred_tok)

        if gt_len == pred_len:
            len_matches += 1

        min_l = min(gt_len, pred_len)
        max_l = max(gt_len, pred_len)

        ec = sum(1 for g, p in zip(gt_tok[:min_l], pred_tok[:min_l]) if g == p)
        tc = sum(1 for g, p in zip(gt_tok[:min_l], pred_tok[:min_l]) if abs(g - p) <= tolerance)

        total_exact += ec
        total_tol += tc
        total_tokens += max_l
        per_sample_exact.append(ec / max_l if max_l > 0 else 0.0)

        mv = min(len(gt_vals), len(pred_vals))
        for g, p in zip(gt_vals[:mv], pred_vals[:mv]):
            total_mae += abs(g - p)
            mae_count += 1

    return {
        'exact_acc': total_exact / max(total_tokens, 1),
        'tol_acc': total_tol / max(total_tokens, 1),
        'mae': total_mae / max(mae_count, 1),
        'len_match_rate': len_matches / max(sample_count, 1),
        'per_sample_exact': per_sample_exact,
    }


# ============================================================
# Main
# ============================================================

# ---------- Publication-quality plot style ----------
_PUB_RC = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'lines.linewidth': 1.2,
    'lines.markersize': 4,
}

# Colour palette (colourblind-friendly)
_C_OBS  = '#7F7F7F'   # grey  – observed
_C_GT   = '#2CA02C'   # green – ground truth
_C_PRED = '#D62728'   # red   – predicted


def plot_reconstruction_samples(
    obs_list: List[List[float]],
    gt_list: List[List[float]],
    pred_list: List[List[float]],
    scene: str,
    method_name: str,
    save_dir: str,
    n_samples: int = 4,
):
    """Plot *n_samples* reconstruction examples in top-journal style.

    Each sample is a sub-figure with three sequences overlaid:
      Observed (grey dashed) / Ground Truth (green solid) / Predicted (red ∘).
    """
    plt.rcParams.update(_PUB_RC)
    os.makedirs(save_dir, exist_ok=True)

    n = min(n_samples, len(obs_list))
    fig, axes = plt.subplots(n, 1, figsize=(7.2, 2.2 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for idx, ax in enumerate(axes):
        obs  = obs_list[idx]
        gt   = gt_list[idx]
        pred = pred_list[idx]

        x_obs  = np.arange(len(obs))
        x_gt   = np.arange(len(gt))
        x_pred = np.arange(len(pred))

        ax.step(x_obs, obs, where='mid', color=_C_OBS, linewidth=0.9,
                linestyle='--', alpha=0.65, label='Observed', zorder=1)
        ax.step(x_gt, gt, where='mid', color=_C_GT, linewidth=1.2,
                label='Ground Truth', zorder=2)
        ax.plot(x_pred, pred, color=_C_PRED, linewidth=0, marker='o',
                markersize=3.5, markeredgewidth=0.4, markeredgecolor='white',
                alpha=0.85, label='Predicted', zorder=3)

        ax.set_ylabel('PRI (µs)')
        ax.set_title(f'Sample {idx + 1}', fontsize=10, fontweight='bold', loc='left')
        ax.grid(True, linewidth=0.3, alpha=0.35)

        if idx == 0:
            ax.legend(loc='upper right', framealpha=0.7, edgecolor='none',
                      ncol=3, columnspacing=1.0)

    axes[-1].set_xlabel('Pulse Index')
    fig.suptitle(f'{method_name}  —  {scene} Scene', fontsize=12,
                 fontweight='bold', y=1.01)
    fig.tight_layout()
    save_path = os.path.join(save_dir, f'{scene}_samples.png')
    fig.savefig(save_path)
    plt.close(fig)
    return save_path
def main():
    parser = argparse.ArgumentParser(description='Semantic Coding PRI Reconstruction Baseline')
    parser.add_argument('--scene', type=str, default=None, help='Miss / Spurious / Mix')
    parser.add_argument('--root', type=str, default=None, help='dataset root')
    parser.add_argument('--config', type=str, default=os.path.join(ROOT, 'config.json'))
    parser.add_argument('--cluster_tol', type=float, default=8.0)
    parser.add_argument('--max_period', type=int, default=40)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_test_samples', type=int, default=None)
    args = parser.parse_args()

    # Load defaults from config.json, then override with CLI
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    scene = args.scene or cfg.get('scene', 'Miss')
    root_rel = args.root or cfg.get('root', 'dataset')
    root = os.path.join(ROOT, root_rel) if not os.path.isabs(root_rel) else root_rel
    seed = args.seed or cfg.get('seed', 42)
    q_bins = cfg.get('quantizer_bins', 1500)
    q_min = cfg.get('quantizer_min', 1)
    q_max = cfg.get('quantizer_max', 1500)
    max_test = args.max_test_samples

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    clean_path = os.path.join(root, 'Ground_Truth')
    obs_path = os.path.join(root, scene)

    quantizer, _, _, actual_bins = build_quantizer(
        quantizer_bins=q_bins, quantizer_min=q_min, quantizer_max=q_max,
        clean_path=clean_path, obs_path=obs_path,
    )

    print(f'=== Semantic Coding Baseline ===')
    print(f'scene={scene}  root={root}  seed={seed}')
    print(f'quantizer: bins={actual_bins}, vocab_size={quantizer.vocab_size}')

    # Load test data
    test_data = load_data(scene, root, seed=seed, max_test_samples=max_test)
    print(f'Test samples: {len(test_data)}')

    # Reconstruct
    reconstructor = SemanticCodingReconstructor(
        cluster_tol=args.cluster_tol,
        max_period=args.max_period,
    )

    all_gt, all_pred = [], []
    t0 = time.time()
    for i, (obs, clean, fname) in enumerate(test_data):
        pred = reconstructor.reconstruct(obs)
        all_gt.append(clean)
        all_pred.append(pred)
        if (i + 1) % 100 == 0:
            print(f'  [{i + 1}/{len(test_data)}] processed')
    elapsed = time.time() - t0
    print(f'Reconstruction done in {elapsed:.1f}s ({elapsed / len(test_data):.4f}s/sample)')

    # Evaluate
    metrics = compute_metrics(all_gt, all_pred, quantizer, tolerance=1)
    print(f'\n=== Results ({scene}) ===')
    print(f'  exact_acc      = {metrics["exact_acc"]:.4%}')
    print(f'  tol1_acc       = {metrics["tol_acc"]:.4%}')
    print(f'  MAE (µs)       = {metrics["mae"]:.4f}')
    print(f'  len_match_rate = {metrics["len_match_rate"]:.4%}')

    # Per-sample summary
    pse = metrics['per_sample_exact']
    print(f'  per_sample_acc: mean={np.mean(pse):.4%}  std={np.std(pse):.4%}')

    # Save results
    out_dir = os.path.join(ROOT, 'BaseLine', 'SemanticCoding', 'results')
    os.makedirs(out_dir, exist_ok=True)
    result_file = os.path.join(out_dir, f'{scene}_{root_rel.replace("/", "_")}.json')
    result = {
        'scene': scene,
        'root': root_rel,
        'num_test_samples': len(test_data),
        'exact_acc': metrics['exact_acc'],
        'tol_acc': metrics['tol_acc'],
        'mae': metrics['mae'],
        'len_match_rate': metrics['len_match_rate'],
        'per_sample_mean': float(np.mean(pse)),
        'per_sample_std': float(np.std(pse)),
        'time_per_sample': elapsed / len(test_data),
    }
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f'\nResults saved to {result_file}')

    # ---- Visualisation ----
    all_obs = [obs for obs, _, _ in test_data]
    vis_dir = os.path.join(ROOT, 'BaseLine', 'SemanticCoding', 'results')
    fig_path = plot_reconstruction_samples(
        all_obs, all_gt, all_pred, scene,
        method_name='Semantic Coding', save_dir=vis_dir, n_samples=4,
    )
    print(f'Visualisation saved to {fig_path}')


if __name__ == '__main__':
    main()
