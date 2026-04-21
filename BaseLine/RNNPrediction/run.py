"""
Train & Evaluate the RNN Seq2Seq baseline for PRI reconstruction.
===================================================================
Continuous-regression version: the model operates entirely in continuous
PRI value space (µs).  Quantisation via the same PRIQuantizer is applied
**only at evaluation time** to compute token-level metrics (exact_acc,
tol_acc) for a fair comparison with the main diffusion-based method.

Usage:
    cd BaseLine/RNNPrediction
    python run.py                                         # defaults from config.json
    python run.py --scene Miss --root dataset --epochs 200
"""
from __future__ import annotations

import os
import sys
import json
import math
import random
import argparse
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------- path setup ----------
_THIS_DIR = str(Path(__file__).resolve().parent)
ROOT = str(Path(__file__).resolve().parent.parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pri_tokenizer import PRIQuantizer, QuantizerConfig
from utils import get_clean_pri_range, get_pri_range

# Explicitly load local model.py (avoid name clash with ROOT/model.py)
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("rnn_model", os.path.join(_THIS_DIR, "model.py"))
_rnn_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_rnn_mod)
build_model = _rnn_mod.build_model


# ============================================================
# Dataset — continuous values
# ============================================================
class PRISample:
    __slots__ = ('observed_pri', 'clean_pri')
    def __init__(self, observed_pri, clean_pri):
        self.observed_pri = observed_pri
        self.clean_pri = clean_pri


class PRIRegressionDataset(Dataset):
    """Returns (src, src_mask, trg, trg_mask) with raw PRI values (µs).

    src = [obs₁ obs₂ ... obsₙ 0 0 ...]       (0-padded)
    trg = [0  cln₁ cln₂ ... clnₘ 0 0 ...]    (leading 0 = start sentinel)
    stop = [0  0  ...  0  1  0 ... 0]          (1 at the position right after
                                                 the last clean value)
    """

    def __init__(self, samples: list, max_src_len: int, max_trg_len: int):
        self.samples = samples
        self.max_src_len = max_src_len
        self.max_trg_len = max_trg_len  # includes the leading sentinel

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        obs = s.observed_pri
        cln = s.clean_pri

        src_len = min(len(obs), self.max_src_len)
        cln_len = min(len(cln), self.max_trg_len - 1)  # -1 for sentinel

        # src
        src = torch.zeros(self.max_src_len)
        src[:src_len] = torch.tensor(obs[:src_len], dtype=torch.float32)
        src_mask = torch.zeros(self.max_src_len)
        src_mask[:src_len] = 1.0

        # trg: [0, cln₁, ..., clnₘ, 0, 0, ...]
        trg = torch.zeros(self.max_trg_len)
        trg[1:cln_len + 1] = torch.tensor(cln[:cln_len], dtype=torch.float32)
        trg_mask = torch.zeros(self.max_trg_len)
        trg_mask[1:cln_len + 1] = 1.0  # only clean value positions

        # stop target: 1 right after last clean value
        stop = torch.zeros(self.max_trg_len - 1)  # aligned with decoder output
        if cln_len < self.max_trg_len - 1:
            stop[cln_len] = 1.0  # position after last clean value

        return {
            'src': src,
            'src_mask': src_mask,
            'trg': trg,
            'trg_mask': trg_mask,
            'stop': stop,
            'cln_len': cln_len,
        }


# ============================================================
# Quantizer builder (same logic as main method)
# ============================================================
def build_quantizer(q_bins=1500, q_min=None, q_max=None,
                    clean_path='', obs_path=''):
    clean_min, clean_max = get_clean_pri_range(clean_path)
    obs_min, obs_max = get_pri_range(obs_path)
    base_min = min(clean_min, obs_min) if q_min is None else float(q_min)
    base_max = max(clean_max, obs_max) if q_max is None else float(q_max)
    bins = int(q_bins)
    if bins <= 1:
        bins = max(2, int(math.ceil(base_max - base_min)) + 1)
    unit_bins = int(math.ceil(base_max - base_min)) + 1
    bins = max(bins, unit_bins)

    quantizer = PRIQuantizer(QuantizerConfig(
        mode='uniform', min_value=base_min, max_value=base_max,
        num_bins=bins, add_special_tokens=True, snap_tolerance=None,
        key_start=1,
    ))
    return quantizer, base_min, base_max, bins


# ============================================================
# Data loading (same 90/10 split with same seed as main method)
# ============================================================
def load_samples(scene, root, seed=42, max_train=None, max_test=None):
    clean_dir = os.path.join(root, 'Ground_Truth')
    obs_dir = os.path.join(root, scene)
    files = sorted(f for f in os.listdir(clean_dir) if f.endswith('.pt'))
    n = len(files)
    idx = list(range(n))
    random.seed(seed)
    random.shuffle(idx)
    split = int(0.9 * n)

    def _load(indices):
        out = []
        for i in indices:
            f = files[i]
            c = torch.load(os.path.join(clean_dir, f), weights_only=False)['seq'].tolist()
            o = torch.load(os.path.join(obs_dir, f), weights_only=False)['seq'].tolist()
            out.append(PRISample(o, c))
        return out

    train = _load(idx[:split])
    test = _load(idx[split:])
    if max_train:
        train = train[:max_train]
    if max_test:
        test = test[:max_test]
    return train, test


# ============================================================
# Evaluation — quantise at metric time only
# ============================================================
def evaluate_model(model, test_loader, quantizer, device, max_decode_len=120):
    """Predict continuous values → quantise → compute token-level metrics."""
    model.eval()
    total_exact = total_tol = total_tokens = 0
    total_mae = 0.0
    mae_count = 0
    len_matches = sample_count = 0
    per_sample_exact: List[float] = []

    with torch.no_grad():
        for batch in test_loader:
            src = batch['src'].to(device)
            src_mask = batch['src_mask'].to(device)
            trg = batch['trg']        # cpu
            cln_len = batch['cln_len']  # [B]
            B = src.size(0)

            all_values, all_lens = model.predict(src, src_mask,
                                                 max_len=max_decode_len)

            for i in range(B):
                # GT continuous values (from trg, skip sentinel at pos 0)
                n_gt = int(cln_len[i].item())
                gt_vals = trg[i, 1:n_gt + 1].tolist()

                # Predicted continuous values
                n_pred = int(all_lens[i].item())
                pred_vals = all_values[i, :n_pred].cpu().tolist()

                if not gt_vals:
                    continue
                sample_count += 1

                # --- MAE in continuous space (µs) ---
                mv = min(len(gt_vals), len(pred_vals))
                for g, p in zip(gt_vals[:mv], pred_vals[:mv]):
                    total_mae += abs(g - p)
                    mae_count += 1

                # --- Quantise for token-level metrics ---
                gt_tokens = quantizer.encode_values(gt_vals,
                                                    add_boundary_tokens=False)
                pred_tokens = quantizer.encode_values(pred_vals,
                                                      add_boundary_tokens=False)

                gt_len = len(gt_tokens)
                pred_len = len(pred_tokens)
                if gt_len == pred_len:
                    len_matches += 1

                min_l = min(gt_len, pred_len)
                max_l = max(gt_len, pred_len)

                ec = sum(1 for g, p in zip(gt_tokens[:min_l], pred_tokens[:min_l])
                         if g == p)
                tc = sum(1 for g, p in zip(gt_tokens[:min_l], pred_tokens[:min_l])
                         if abs(g - p) <= 1)
                total_exact += ec
                total_tol += tc
                total_tokens += max_l
                per_sample_exact.append(ec / max_l if max_l > 0 else 0.0)

    return {
        'exact_acc': total_exact / max(total_tokens, 1),
        'tol_acc': total_tol / max(total_tokens, 1),
        'mae': total_mae / max(mae_count, 1),
        'len_match_rate': len_matches / max(sample_count, 1),
        'per_sample_exact': per_sample_exact,
    }


# ============================================================
# Collect samples for visualisation
# ============================================================
def collect_visual_samples(model, test_samples, device, scale, max_decode_len=120, n=4):
    """Run inference on the first *n* test samples and return (obs, gt, pred) lists."""
    model.eval()
    obs_list, gt_list, pred_list = [], [], []
    for s in test_samples[:n]:
        obs_vals = s.observed_pri
        gt_vals = s.clean_pri
        src = torch.tensor(obs_vals, dtype=torch.float32).unsqueeze(0).to(device)
        src_mask = torch.ones(1, len(obs_vals)).to(device)
        with torch.no_grad():
            values, lengths = model.predict(src, src_mask, max_len=max_decode_len)
        n_pred = int(lengths[0].item())
        pred_vals = values[0, :n_pred].cpu().tolist()
        obs_list.append(obs_vals)
        gt_list.append(gt_vals)
        pred_list.append(pred_vals)
    return obs_list, gt_list, pred_list


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

_C_OBS  = '#7F7F7F'
_C_GT   = '#2CA02C'
_C_PRED = '#D62728'


def plot_reconstruction_samples(
    obs_list, gt_list, pred_list,
    scene, method_name, save_dir, n_samples=4,
):
    plt.rcParams.update(_PUB_RC)
    os.makedirs(save_dir, exist_ok=True)
    n = min(n_samples, len(obs_list))
    fig, axes = plt.subplots(n, 1, figsize=(7.2, 2.2 * n), sharex=False)
    if n == 1:
        axes = [axes]
    for idx, ax in enumerate(axes):
        obs, gt, pred = obs_list[idx], gt_list[idx], pred_list[idx]
        ax.step(np.arange(len(obs)), obs, where='mid', color=_C_OBS,
                linewidth=0.9, linestyle='--', alpha=0.65, label='Observed', zorder=1)
        ax.step(np.arange(len(gt)), gt, where='mid', color=_C_GT,
                linewidth=1.2, label='Ground Truth', zorder=2)
        ax.plot(np.arange(len(pred)), pred, color=_C_PRED, linewidth=0,
                marker='o', markersize=3.5, markeredgewidth=0.4,
                markeredgecolor='white', alpha=0.85, label='Predicted', zorder=3)
        ax.set_ylabel('PRI (µs)')
        ax.set_title(f'Sample {idx+1}', fontsize=10, fontweight='bold', loc='left')
        ax.grid(True, linewidth=0.3, alpha=0.35)
        if idx == 0:
            ax.legend(loc='upper right', framealpha=0.7, edgecolor='none',
                      ncol=3, columnspacing=1.0)
    axes[-1].set_xlabel('Pulse Index')
    fig.suptitle(f'{method_name}  \u2014  {scene} Scene', fontsize=12,
                 fontweight='bold', y=1.01)
    fig.tight_layout()
    save_path = os.path.join(save_dir, f'{scene}_samples.png')
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


# ============================================================
# Training loop — MSE + BCE stop loss
# ============================================================
def train_one_epoch(model, loader, optimizer, device,
                    teacher_forcing: float, stop_weight: float = 1.0,
                    grad_clip: float = 1.0):
    model.train()
    total_loss = 0.0
    n_batches = 0
    mse_fn = nn.MSELoss(reduction='none')
    bce_fn = nn.BCEWithLogitsLoss(reduction='none')

    for batch in loader:
        src = batch['src'].to(device)
        src_mask = batch['src_mask'].to(device)
        trg = batch['trg'].to(device)
        trg_mask = batch['trg_mask'].to(device)
        stop_target = batch['stop'].to(device)  # [B, trg_len-1]

        pred_values, stop_logits = model(
            src, src_mask, trg, trg_mask,
            teacher_forcing_ratio=teacher_forcing,
        )

        # value_mask: positions where clean values exist (trg_mask shifted by 1)
        value_mask = trg_mask[:, 1:]  # [B, trg_len-1]

        # MSE loss on valid clean positions only
        mse = mse_fn(pred_values, trg[:, 1:])  # [B, trg_len-1]
        # Normalise by scale² so loss magnitude is reasonable
        mse = mse / (model.scale ** 2)
        mse_loss = (mse * value_mask).sum() / value_mask.sum().clamp(min=1)

        # BCE stop loss
        bce = bce_fn(stop_logits, stop_target)
        bce_loss = bce.mean()

        loss = mse_loss + stop_weight * bce_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='RNN Seq2Seq PRI Reconstruction Baseline (Regression)')
    parser.add_argument('--scene', type=str, default=None)
    parser.add_argument('--root', type=str, default=None)
    parser.add_argument('--config', type=str, default=os.path.join(ROOT, 'config.json'))
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--teacher_forcing', type=float, default=0.5)
    parser.add_argument('--stop_weight', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--max_train_samples', type=int, default=None)
    parser.add_argument('--max_test_samples', type=int, default=None)
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    scene = args.scene or cfg.get('scene', 'Miss')
    root_rel = args.root or cfg.get('root', 'dataset')
    root = os.path.join(ROOT, root_rel) if not os.path.isabs(root_rel) else root_rel
    seed = args.seed or cfg.get('seed', 42)
    device = args.device
    if device.startswith('cuda') and not torch.cuda.is_available():
        device = 'cpu'

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    q_bins = cfg.get('quantizer_bins', 1500)
    q_min = cfg.get('quantizer_min', 1)
    q_max = cfg.get('quantizer_max', 1500)
    seq_len = cfg.get('spliced_seq_length', 220)
    scale = float(q_max) if q_max else 1500.0
    clean_path = os.path.join(root, 'Ground_Truth')
    obs_path = os.path.join(root, scene)

    quantizer, _, _, actual_bins = build_quantizer(
        q_bins, q_min, q_max, clean_path, obs_path,
    )

    print('=== RNN Seq2Seq Baseline (Regression) ===')
    print(f'scene={scene}  root={root}  seed={seed}  device={device}')
    print(f'quantizer (eval only): bins={actual_bins}, vocab_size={quantizer.vocab_size}')
    print(f'model: embed={args.embed_dim} hidden={args.hidden_dim} '
          f'layers={args.n_layers} dropout={args.dropout} scale={scale}')

    # ---- Data ----
    train_samples, test_samples = load_samples(
        scene, root, seed,
        max_train=args.max_train_samples, max_test=args.max_test_samples,
    )
    max_src = max(len(s.observed_pri) for s in train_samples + test_samples)
    max_trg_clean = max(len(s.clean_pri) for s in train_samples + test_samples)
    max_src_len = min(max_src + 5, seq_len)
    max_trg_len = min(max_trg_clean + 1 + 5, seq_len)  # +1 for sentinel

    train_ds = PRIRegressionDataset(train_samples, max_src_len, max_trg_len)
    test_ds = PRIRegressionDataset(test_samples, max_src_len, max_trg_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    print(f'train={len(train_ds)}  test={len(test_ds)}  '
          f'max_src={max_src_len}  max_trg={max_trg_len}')

    # ---- Model ----
    model = build_model(
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        dropout=args.dropout,
        scale=scale,
        device=device,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable params: {n_params:,}')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # LR scheduler: cosine annealing with warmup
    warmup_epochs = 10
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, args.epochs - warmup_epochs)
        return max(1e-6 / args.lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Teacher forcing schedule: linear decay from initial to 0
    def get_tf_ratio(epoch):
        return max(0.0, args.teacher_forcing * (1.0 - epoch / args.epochs))

    # ---- Output dir ----
    out_dir = os.path.join(ROOT, 'BaseLine', 'RNNPrediction', 'CheckPoint',
                           f'{scene}_{root_rel.replace("/", "_")}')
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, 'log.txt')
    log_f = open(log_path, 'w', encoding='utf-8')

    def log(msg):
        print(msg)
        log_f.write(msg + '\n')
        log_f.flush()

    # ---- Training ----
    best_loss = float('inf')
    best_model_path = os.path.join(out_dir, 'best_model.pth')
    latest_model_path = os.path.join(out_dir, 'latest_model.pth')

    t0 = time.time()
    for epoch in range(args.epochs):
        tf = get_tf_ratio(epoch)
        train_loss = train_one_epoch(model, train_loader, optimizer, device,
                                     tf, stop_weight=args.stop_weight)
        scheduler.step()

        # Evaluate every 10 epochs (or last 5)
        if (epoch + 1) % 10 == 0 or epoch >= args.epochs - 5:
            metrics = evaluate_model(model, test_loader, quantizer, device,
                                     max_decode_len=max_trg_clean + 10)
            log(f'epoch={epoch:4d}  loss={train_loss:.6f}  '
                f'exact={metrics["exact_acc"]:.4%}  tol1={metrics["tol_acc"]:.4%}  '
                f'mae={metrics["mae"]:.2f}  len_match={metrics["len_match_rate"]:.4%}  '
                f'tf={tf:.3f}  lr={optimizer.param_groups[0]["lr"]:.2e}')
        else:
            log(f'epoch={epoch:4d}  loss={train_loss:.6f}  '
                f'tf={tf:.3f}  lr={optimizer.param_groups[0]["lr"]:.2e}')

        torch.save(model.state_dict(), latest_model_path)
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), best_model_path)

    train_time = time.time() - t0
    log(f'\nTraining done in {train_time:.0f}s')

    # ---- Final evaluation with best model ----
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device,
                                         weights_only=False))

    log('\n=== Final Evaluation (best model) ===')
    final = evaluate_model(model, test_loader, quantizer, device,
                           max_decode_len=max_trg_clean + 10)
    log(f'  exact_acc      = {final["exact_acc"]:.4%}')
    log(f'  tol1_acc       = {final["tol_acc"]:.4%}')
    log(f'  MAE (µs)       = {final["mae"]:.4f}')
    log(f'  len_match_rate = {final["len_match_rate"]:.4%}')

    pse = final['per_sample_exact']
    if pse:
        log(f'  per_sample_acc: mean={np.mean(pse):.4%}  std={np.std(pse):.4%}')

    # Save result json
    result = {
        'scene': scene, 'root': root_rel,
        'epochs': args.epochs,
        'num_train': len(train_ds), 'num_test': len(test_ds),
        'exact_acc': final['exact_acc'],
        'tol_acc': final['tol_acc'],
        'mae': final['mae'],
        'len_match_rate': final['len_match_rate'],
        'train_time_s': train_time,
    }
    result_path = os.path.join(out_dir, 'result.json')
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    log(f'\nResults saved to {result_path}')
    log(f'Model saved to {best_model_path}')

    # ---- Visualisation ----
    vis_dir = os.path.join(out_dir, 'visuals')
    obs_vis, gt_vis, pred_vis = collect_visual_samples(
        model, test_samples, device, scale,
        max_decode_len=max_trg_clean + 10, n=4,
    )
    fig_path = plot_reconstruction_samples(
        obs_vis, gt_vis, pred_vis, scene,
        method_name='RNN Seq2Seq', save_dir=vis_dir, n_samples=4,
    )
    log(f'Visualisation saved to {fig_path}')
    log_f.close()


if __name__ == '__main__':
    main()
