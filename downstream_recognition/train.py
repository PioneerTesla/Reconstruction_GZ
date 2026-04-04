"""
Radar Working Mode Recognition — Training & Evaluation
========================================================
Train the PRIModeClassifier on generated PRI data.
Produces:
  - Training curves (loss / accuracy)
  - Per-class metrics (precision, recall, F1)
  - Confusion matrix

Usage:
    python train.py                        # default parameters
    python train.py --epochs 80 --lr 1e-3  # custom
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Allow importing from parent
_PARENT = str(Path(__file__).resolve().parent.parent)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

# Ensure local directory is searched first
_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from generate_data import generate_samples, collate_sequences, LABEL_NAMES

# Import from the local model.py (not the parent model.py)
import importlib.util
_model_spec = importlib.util.spec_from_file_location(
    "downstream_model", os.path.join(_THIS_DIR, "model.py"))
_model_mod = importlib.util.module_from_spec(_model_spec)
_model_spec.loader.exec_module(_model_mod)
PRIModeClassifier = _model_mod.PRIModeClassifier

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Publication-quality plot style ─────────────────────────────────
_RC = {
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
    'lines.linewidth': 1.2,
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--n_per_class', type=int, default=2000)
    p.add_argument('--seq_len', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--out_dir', type=str, default='results')
    return p.parse_args()


# ── Metrics helpers ────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int):
    """Return per-class precision, recall, F1 and overall accuracy."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        precision[c] = tp / max(tp + fp, 1)
        recall[c] = tp / max(tp + fn, 1)
        f1[c] = 2 * precision[c] * recall[c] / max(precision[c] + recall[c], 1e-12)
    acc = np.trace(cm) / max(cm.sum(), 1)
    return cm, precision, recall, f1, acc


# ── Plotting functions ─────────────────────────────────────────────

def save_training_curves(history: dict, out_dir: str):
    plt.rcParams.update(_RC)
    epochs = np.arange(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(epochs, history['train_loss'], label='Train', color='#4C72B0')
    ax1.plot(epochs, history['val_loss'], label='Val', color='#DD8452')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.legend()
    ax1.grid(True, linewidth=0.3, alpha=0.4)

    ax2.plot(epochs, history['train_acc'], label='Train', color='#4C72B0')
    ax2.plot(epochs, history['val_acc'], label='Val', color='#DD8452')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Classification Accuracy')
    ax2.legend()
    ax2.grid(True, linewidth=0.3, alpha=0.4)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

    fig.tight_layout()
    path = os.path.join(out_dir, 'training_curves.png')
    fig.savefig(path)
    plt.close(fig)
    return path


def save_confusion_matrix(cm: np.ndarray, label_names: list, out_dir: str, normalize: bool = True):
    plt.rcParams.update(_RC)
    if normalize:
        row_sum = cm.sum(axis=1, keepdims=True).astype(float)
        row_sum[row_sum == 0] = 1.0
        cm_plot = cm.astype(float) / row_sum
    else:
        cm_plot = cm.astype(float)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm_plot, cmap='Blues', interpolation='nearest', vmin=0, vmax=1 if normalize else None)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label('Proportion' if normalize else 'Count')

    n = len(label_names)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(label_names, rotation=45, ha='right')
    ax.set_yticklabels(label_names)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = cm_plot[i, j]
            color = 'white' if val > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}' if normalize else f'{int(cm[i, j])}',
                    ha='center', va='center', color=color, fontsize=10)

    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix (row-normalised)' if normalize else 'Confusion Matrix')
    fig.tight_layout()
    path = os.path.join(out_dir, 'confusion_matrix.png')
    fig.savefig(path)
    plt.close(fig)
    return path


def save_per_class_bar(precision, recall, f1, label_names, out_dir):
    plt.rcParams.update(_RC)
    x = np.arange(len(label_names))
    w = 0.25

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w, precision, w, label='Precision', color='#4C72B0')
    ax.bar(x, recall, w, label='Recall', color='#55A868')
    ax.bar(x + w, f1, w, label='F1', color='#DD8452')
    ax.set_xticks(x)
    ax.set_xticklabels(label_names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score')
    ax.set_title('Per-class Metrics')
    ax.legend()
    ax.grid(axis='y', linewidth=0.3, alpha=0.4)
    fig.tight_layout()
    path = os.path.join(out_dir, 'per_class_metrics.png')
    fig.savefig(path)
    plt.close(fig)
    return path


# ── Training loop ──────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, L_batch, y_batch in loader:
        X_batch, L_batch, y_batch = X_batch.to(device), L_batch.to(device), y_batch.to(device)
        logits = model(X_batch, L_batch)
        loss = criterion(logits, y_batch)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * y_batch.size(0)
        correct += (logits.argmax(1) == y_batch).sum().item()
        total += y_batch.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_pred, all_true = [], []
    for X_batch, L_batch, y_batch in loader:
        X_batch, L_batch, y_batch = X_batch.to(device), L_batch.to(device), y_batch.to(device)
        logits = model(X_batch, L_batch)
        loss = criterion(logits, y_batch)
        total_loss += loss.item() * y_batch.size(0)
        pred = logits.argmax(1)
        correct += (pred == y_batch).sum().item()
        total += y_batch.size(0)
        all_pred.append(pred.cpu().numpy())
        all_true.append(y_batch.cpu().numpy())
    return total_loss / total, correct / total, np.concatenate(all_pred), np.concatenate(all_true)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = args.device if torch.cuda.is_available() else 'cpu'
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    print('=== Generating data ===')
    seqs, labs = generate_samples(n_per_class=args.n_per_class, seq_len=args.seq_len, seed=args.seed)
    X, lengths, y = collate_sequences(seqs, labs)
    print(f'Total samples: {len(y)}')
    for i, name in enumerate(LABEL_NAMES):
        print(f'  {name}: {(y == i).sum().item()}')

    # Train / val split (80/20)
    n = len(y)
    idx = torch.randperm(n)
    split = int(0.8 * n)
    train_idx, val_idx = idx[:split], idx[split:]
    train_ds = TensorDataset(X[train_idx], lengths[train_idx], y[train_idx])
    val_ds = TensorDataset(X[val_idx], lengths[val_idx], y[val_idx])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    num_classes = len(LABEL_NAMES)
    model = PRIModeClassifier(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0

    print('=== Training ===')
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(vl_acc)

        if epoch % 10 == 0 or epoch == 1:
            print(f'epoch={epoch:3d}  tr_loss={tr_loss:.4f}  vl_loss={vl_loss:.4f}  '
                  f'tr_acc={tr_acc:.4f}  vl_acc={vl_acc:.4f}')

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), os.path.join(out_dir, 'best_model.pth'))

    # Load best model for final eval
    model.load_state_dict(torch.load(os.path.join(out_dir, 'best_model.pth'), map_location=device))
    _, final_acc, y_pred, y_true = evaluate(model, val_loader, criterion, device)

    cm, precision, recall, f1, acc = compute_metrics(y_true, y_pred, num_classes)
    print(f'\n=== Final Evaluation (best model) ===')
    print(f'Overall accuracy: {acc:.4f}')
    for i, name in enumerate(LABEL_NAMES):
        print(f'  {name:15s}  P={precision[i]:.4f}  R={recall[i]:.4f}  F1={f1[i]:.4f}')

    # Save plots
    p1 = save_training_curves(history, out_dir)
    print(f'[plot] training curves -> {p1}')
    p2 = save_confusion_matrix(cm, LABEL_NAMES, out_dir)
    print(f'[plot] confusion matrix -> {p2}')
    p3 = save_per_class_bar(precision, recall, f1, LABEL_NAMES, out_dir)
    print(f'[plot] per-class metrics -> {p3}')

    print('\nDone.')


if __name__ == '__main__':
    main()
