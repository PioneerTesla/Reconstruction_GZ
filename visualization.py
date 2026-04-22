from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Sequence, Union
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Publication-quality style (Nature / IEEE)
# ---------------------------------------------------------------------------
_PUB_RC = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'lines.linewidth': 1.2,
    'lines.markersize': 3,
}


def _apply_style():
    plt.rcParams.update(_PUB_RC)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().float().cpu().numpy()


def _pca_2d(data: np.ndarray) -> np.ndarray:
    """Project [N, D] -> [N, 2] via PCA (numpy only)."""
    data = data.astype(np.float64)
    mean = data.mean(axis=0)
    centered = data - mean
    cov = centered.T @ centered / max(len(data) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1][:2]
    proj = centered @ eigvecs[:, idx]
    return proj.astype(np.float32)


# ===================================================================
# 1. Codebook embedding distribution (1500 codes in feature space)
# ===================================================================

def save_codebook_visualization(
    embedding_weight: torch.Tensor,
    offset: int,
    num_pri_tokens: int,
    epoch: int,
    out_dir: str,
) -> str:
    """
    Plot the full codebook (all 1500 PRI token embeddings) projected to 2-D
    via PCA at a given epoch. Colour encodes the PRI key value (1..1500).
    """
    _apply_style()
    ensure_dir(out_dir)

    w = _to_numpy(embedding_weight)          # [vocab_size, D]
    pri_w = w[offset: offset + num_pri_tokens]   # [1500, D]
    proj = _pca_2d(pri_w)                    # [1500, 2]

    keys = np.arange(1, num_pri_tokens + 1)

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(proj[:, 0], proj[:, 1], c=keys, cmap='viridis',
                    s=6, alpha=0.85, edgecolors='none')
    cbar = fig.colorbar(sc, ax=ax, shrink=0.82, pad=0.02)
    cbar.set_label('PRI Code (1\u20131500)')
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_title(f'Codebook Embedding Distribution  (epoch {epoch})')
    ax.grid(True, linewidth=0.3, alpha=0.4)
    fig.tight_layout()

    save_path = os.path.join(out_dir, f'epoch_{epoch:04d}_codebook.png')
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


# ===================================================================
# 2. Diffusion denoising process (clean vs target-position latent)
# ===================================================================

def _denoise_prepare(trace, clean_embeds, input_mask, sample_idx=0):
    """Shared helper: compute PCA projections for denoise plots.

    Returns (clean_proj, proj_per_step, steps_sorted, T) or None on failure.
    """
    mask_np = _to_numpy(input_mask[sample_idx]).astype(int)
    trg_idx = np.where(mask_np == 1)[0]
    if len(trg_idx) == 0:
        return None

    clean_np = _to_numpy(clean_embeds[sample_idx])
    clean_trg = clean_np[trg_idx]

    steps_sorted = sorted(trace.keys(), reverse=True)
    if not steps_sorted:
        return None

    all_vecs = [clean_trg]
    step_feats = {}
    for s in steps_sorted:
        feat = _to_numpy(trace[s][sample_idx])
        trg_feat = feat[trg_idx]
        step_feats[s] = trg_feat
        all_vecs.append(trg_feat)
    all_vecs_cat = np.concatenate(all_vecs, axis=0)
    all_proj = _pca_2d(all_vecs_cat)

    T = len(trg_idx)
    clean_proj = all_proj[:T]
    proj_per_step = {}
    for i, s in enumerate(steps_sorted):
        start = T + i * T
        proj_per_step[s] = all_proj[start: start + T]

    return clean_proj, proj_per_step, steps_sorted, T


def _plot_single_denoise_step(clean_proj, step_proj, step, T, epoch, ax=None):
    """Plot a single diffusion step scatter onto *ax*."""
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(5, 4.5))
    else:
        fig = None

    # Clean embedding – large red star markers for visibility
    ax.scatter(clean_proj[:, 0], clean_proj[:, 1],
               c='#E63946', marker='*', s=64, linewidths=0.4,
               edgecolors='black', label='Clean embedding', zorder=10)
    sc = ax.scatter(step_proj[:, 0], step_proj[:, 1],
                    c=np.arange(T), cmap='coolwarm',
                    s=16, alpha=0.85, edgecolors='white', linewidths=0.3,
                    zorder=2)
    ax.set_title(f't = {step}')
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.legend(loc='upper right', fontsize=14, framealpha=0.6)
    ax.grid(True, linewidth=0.3, alpha=0.4)
    return fig, sc


def save_diffusion_denoise_visualization(
    trace: Dict[int, torch.Tensor],
    clean_embeds: torch.Tensor,
    input_mask: torch.Tensor,
    epoch: int,
    out_dir: str,
    mode: str = 'combined',
    select_steps: Optional[List[int]] = None,
) -> Union[str, List[str]]:
    """Visualise the diffusion denoising trajectory in 2-D PCA space.

    Args:
        mode: 'combined'   – all steps side-by-side in one PNG (default).
              'individual' – one PNG per recorded timestep.
              'select'     – only render steps in *select_steps* (one PNG each
                             if len > 1, or combined if len ≤ 5).
        select_steps: used when mode='select'; list of timestep ints to show.

    Returns the saved path(s).
    """
    _apply_style()
    ensure_dir(out_dir)

    prep = _denoise_prepare(trace, clean_embeds, input_mask)
    if prep is None:
        return ''
    clean_proj, proj_per_step, steps_sorted, T = prep

    # Decide which steps to plot
    if mode == 'select' and select_steps is not None:
        steps_to_show = [s for s in steps_sorted if s in select_steps]
        if not steps_to_show:
            steps_to_show = steps_sorted
    else:
        steps_to_show = steps_sorted

    # --- individual mode: one PNG per step ---
    if mode == 'individual':
        paths: List[str] = []
        for step in steps_to_show:
            fig, sc = _plot_single_denoise_step(
                clean_proj, proj_per_step[step], step, T, epoch,
            )
            cbar = fig.colorbar(sc, ax=fig.axes[0], shrink=0.75, pad=0.02)
            cbar.set_label('Target position index')
            fig.suptitle(f'Diffusion Denoising  (epoch {epoch})', fontsize=12, y=1.01)
            fig.tight_layout()
            p = os.path.join(out_dir, f'epoch_{epoch:04d}_denoise_t{step}.png')
            fig.savefig(p)
            plt.close(fig)
            paths.append(p)
        return paths

    # --- select mode with few steps → combined into one figure ---
    # --- combined mode (default) ---
    n_steps = len(steps_to_show)
    fig, axes = plt.subplots(1, n_steps, figsize=(4.5 * n_steps, 4.2), squeeze=False)
    axes = axes[0]
    sc = None
    for ax, step in zip(axes, steps_to_show):
        _, sc = _plot_single_denoise_step(
            clean_proj, proj_per_step[step], step, T, epoch, ax=ax,
        )

    if sc is not None:
        cbar = fig.colorbar(sc, ax=axes.tolist(), shrink=0.75, pad=0.02)
        cbar.set_label('Target position index')
    fig.suptitle(f'Diffusion Denoising Process  (epoch {epoch})', fontsize=13, y=1.02)
    fig.tight_layout()

    suffix = '_select' if mode == 'select' else ''
    save_path = os.path.join(out_dir, f'epoch_{epoch:04d}_diffusion_denoise{suffix}.png')
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


# ===================================================================
# 3. Training curves
# ===================================================================

def save_training_curves(history: Dict[str, List[float]], out_dir: str) -> str:
    """
    Plot MSE loss, CE loss, total loss, AP-exact, AP-tol1, MAE, and len_match over epochs.
    """
    _apply_style()
    ensure_dir(out_dir)

    epochs = np.arange(len(history['train_loss']))
    has_len_match = 'len_match' in history and len(history['len_match']) > 0

    n_cols = 4 if has_len_match else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(5.5 * n_cols, 4.5))
    ax_loss, ax_acc, ax_mae = axes[0], axes[1], axes[2]

    # -- Loss subplot --
    ax_loss.plot(epochs, history['train_loss'], label='Train Loss', color='#4C72B0')
    ax_loss.plot(epochs, history['test_loss'], label='Test Loss', color='#DD8452')
    ax_loss.plot(epochs, history['train_mse'], label='MSE', color='#55A868', linestyle='--')
    ax_loss.plot(epochs, history['train_ce'], label='CE', color='#C44E52', linestyle='--')
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss')
    ax_loss.set_title('Training & Validation Loss')
    ax_loss.legend(loc='best', framealpha=0.7, fontsize=14)
    ax_loss.grid(True, linewidth=0.3, alpha=0.4)

    # -- Accuracy subplot --
    ax_acc.plot(epochs, history['ap_exact_acc'], label='AP-Exact', color='#4C72B0')
    ax_acc.plot(epochs, history['ap_tol1_acc'], label='AP-Tol@1', color='#55A868')
    ax_acc.set_xlabel('Epoch')
    ax_acc.set_ylabel('Accuracy')
    ax_acc.set_title('Auto-Parse Accuracy (test-mode)')
    ax_acc.legend(loc='best', framealpha=0.7, fontsize=14)
    ax_acc.grid(True, linewidth=0.3, alpha=0.4)
    ax_acc.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

    # -- MAE subplot --
    ax_mae.plot(epochs, history['ap_mae'], label='MAE (\u00b5s)', color='#C44E52')
    ax_mae.set_xlabel('Epoch')
    ax_mae.set_ylabel('MAE (\u00b5s)')
    ax_mae.set_title('Mean Absolute Error')
    ax_mae.legend(loc='best', framealpha=0.7, fontsize=14)
    ax_mae.grid(True, linewidth=0.3, alpha=0.4)

    # -- Len Match subplot --
    if has_len_match:
        ax_len = axes[3]
        ax_len.plot(epochs, history['len_match'], label='Len Match Rate', color='#8172B2')
        ax_len.set_xlabel('Epoch')
        ax_len.set_ylabel('Len Match Rate')
        ax_len.set_title('Length Match Rate vs Epoch')
        ax_len.legend(loc='best', framealpha=0.7, fontsize=14)
        ax_len.grid(True, linewidth=0.3, alpha=0.4)
        ax_len.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

    fig.tight_layout()
    save_path = os.path.join(out_dir, 'training_curves.png')
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


# ===================================================================
# 4. Reconstruction comparison (predicted target vs clean target)
# ===================================================================

def save_reconstruction_comparison(
    clean_pri: Sequence[float],
    pred_pri: Sequence[float],
    obs_pri: Sequence[float],
    sample_idx,
    out_dir: str,
) -> str:
    """
    Plot clean PRI, observed (corrupted) PRI, and predicted (reconstructed) PRI
    on the same axes for visual comparison.
    """
    _apply_style()
    ensure_dir(out_dir)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    x_clean = np.arange(len(clean_pri))
    x_obs = np.arange(len(obs_pri))
    x_pred = np.arange(len(pred_pri))

    ax.plot(x_clean, clean_pri, label='Clean PRI', color='#55A868',
            linewidth=1.0, marker='o', markersize=2.5)
    ax.plot(x_obs, obs_pri, label='Observed PRI (corrupted)', color='#C44E52',
            linewidth=0.8, marker='s', markersize=2, alpha=0.7)
    ax.plot(x_pred, pred_pri, label='Reconstructed PRI', color='#4C72B0',
            linewidth=1.0, marker='^', markersize=2.5, linestyle='--')
    ax.set_xlabel('Position in target segment')
    ax.set_ylabel('PRI Value')
    ax.set_title(f'Reconstruction Comparison  (sample {sample_idx})')
    ax.legend(loc='best', framealpha=0.7)
    ax.grid(True, linewidth=0.3, alpha=0.4)
    fig.tight_layout()

    save_path = os.path.join(out_dir, f'reconstruction_sample_{sample_idx}.png')
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


# ===================================================================
# 5. Confusion matrix
# ===================================================================

def compute_confusion_matrix_from_arrays(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            mat[t, p] += 1
    return mat


def save_confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor, labels: Optional[np.ndarray], save_path: str, normalize: bool = True) -> str:
    _apply_style()
    y_true_np = _to_numpy(y_true).astype(np.int64)
    y_pred_np = _to_numpy(y_pred).astype(np.int64)
    if labels is None: 
        labels = np.unique(np.concatenate([y_true_np, y_pred_np]))
    label_to_index = {int(label): idx for idx, label in enumerate(labels.tolist())}
    true_idx = np.asarray([label_to_index[int(v)] for v in y_true_np], dtype=np.int64)
    pred_idx = np.asarray([label_to_index[int(v)] for v in y_pred_np], dtype=np.int64)
    cm = compute_confusion_matrix_from_arrays(true_idx, pred_idx, len(labels)).astype(np.float32)
    if normalize:
        row_sum = cm.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        cm = cm / row_sum

    ensure_dir(os.path.dirname(save_path))
    fig_size = 7.0 if len(labels) > 100 else max(5.5, min(12.0, len(labels) * 0.12))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(cm, aspect='auto', interpolation='nearest', cmap='Blues')
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Normalised frequency' if normalize else 'Count')
    ax.set_title('Confusion Matrix (row-normalised)' if normalize else 'Confusion Matrix')
    tick_positions = np.arange(len(labels))
    if len(labels) <= 60:
        tick_labels = [str(int(v)) for v in labels]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=90, fontsize=6)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels, fontsize=6)
    elif len(labels) <= 150:
        sparse_ticks = np.linspace(0, len(labels) - 1, num=20, dtype=int)
        sparse_labels = [str(int(labels[i])) for i in sparse_ticks]
        ax.set_xticks(sparse_ticks)
        ax.set_xticklabels(sparse_labels, rotation=90, fontsize=7)
        ax.set_yticks(sparse_ticks)
        ax.set_yticklabels(sparse_labels, fontsize=7)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_xlabel('Predicted token ID')
    ax.set_ylabel('True token ID')
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


# ===================================================================
# 6. Diversity visualisation (multiple reconstruction overlays)
# ===================================================================

def save_diversity_visualization(
    all_predictions: list,
    quantizer,
    out_dir: str,
    max_samples: int = 4,
) -> str:
    """Overlay *N* reconstruction runs on top of the ground-truth PRI.

    *all_predictions* comes from ``evaluate_diversity``:
    each element is ``(gt_tokens, [pred_run1, …, pred_runN])``.
    """
    _apply_style()
    ensure_dir(out_dir)

    n_show = min(len(all_predictions), max_samples)
    fig, axes = plt.subplots(n_show, 1, figsize=(10, 3.2 * n_show), squeeze=False)

    for idx in range(n_show):
        gt_tok, pred_runs = all_predictions[idx]
        gt_vals = quantizer.decode_ids(gt_tok, remove_special=False)
        ax = axes[idx, 0]

        x_gt = np.arange(len(gt_vals))
        ax.plot(x_gt, gt_vals, label='Ground Truth', color='#55A868',
                linewidth=1.2, marker='o', markersize=2.5, zorder=10)

        cmap = plt.cm.get_cmap('tab10')
        for r, pred_tok in enumerate(pred_runs):
            pv = quantizer.decode_ids(pred_tok, remove_special=False)
            x_p = np.arange(len(pv))
            ax.plot(x_p, pv, color=cmap(r % 10), alpha=0.35, linewidth=0.7,
                    label=f'Run {r+1}' if r < 5 else None)

        ax.set_xlabel('Position')
        ax.set_ylabel('PRI Value')
        ax.set_title(f'Diversity – Sample {idx}')
        ax.legend(loc='best', framealpha=0.6, fontsize=14, ncol=3)
        ax.grid(True, linewidth=0.3, alpha=0.4)

    fig.tight_layout()
    save_path = os.path.join(out_dir, 'diversity_overlay.png')
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


# ===================================================================
# 6b. Diversity mean ± variance (shaded band)
# ===================================================================

def save_diversity_mean_variance(
    all_predictions: list,
    quantizer,
    out_dir: str,
    max_samples: int = 4,
) -> str:
    """Plot mean reconstruction with ±1 std shading overlay on GT.

    *all_predictions* comes from ``evaluate_diversity``.
    """
    _apply_style()
    ensure_dir(out_dir)

    n_show = min(len(all_predictions), max_samples)
    fig, axes = plt.subplots(n_show, 1, figsize=(10, 3.2 * n_show), squeeze=False)

    for idx in range(n_show):
        gt_tok, pred_runs = all_predictions[idx]
        gt_vals = np.asarray(quantizer.decode_ids(gt_tok, remove_special=False))
        ax = axes[idx, 0]

        # Decode all runs into a 2-D array (N_runs × max_len), pad with NaN
        decoded_runs = []
        for pred_tok in pred_runs:
            pv = quantizer.decode_ids(pred_tok, remove_special=False)
            decoded_runs.append(pv)
        max_len = max(len(r) for r in decoded_runs) if decoded_runs else 0
        if max_len == 0:
            continue
        mat = np.full((len(decoded_runs), max_len), np.nan)
        for r, pv in enumerate(decoded_runs):
            mat[r, :len(pv)] = pv

        x = np.arange(max_len)
        mean_vals = np.nanmean(mat, axis=0)
        std_vals = np.nanstd(mat, axis=0)

        x_gt = np.arange(len(gt_vals))
        ax.plot(x_gt, gt_vals, label='Ground Truth', color='#55A868',
                linewidth=1.2, marker='o', markersize=2.5, zorder=10)
        ax.plot(x, mean_vals, label='Mean Recon.', color='#4C72B0',
                linewidth=1.0, zorder=5)
        ax.fill_between(x, mean_vals - std_vals, mean_vals + std_vals,
                        color='#4C72B0', alpha=0.2, label='±1 Std', zorder=4)

        ax.set_xlabel('Position')
        ax.set_ylabel('PRI Value')
        ax.set_title(f'Mean ± Std Diversity – Sample {idx}')
        ax.legend(loc='best', framealpha=0.6, fontsize=14)
        ax.grid(True, linewidth=0.3, alpha=0.4)

    fig.tight_layout()
    save_path = os.path.join(out_dir, 'diversity_mean_variance.png')
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


# ===================================================================
# 7. Per-sample accuracy histogram
# ===================================================================

def save_per_sample_accuracy_histogram(
    per_sample_accs: Sequence[float],
    out_dir: str,
    label: str = '',
) -> str:
    """Histogram showing the distribution of per-sample exact accuracies."""
    _apply_style()
    ensure_dir(out_dir)

    accs = np.asarray(per_sample_accs)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(accs, bins=30, range=(0, 1), color='#4C72B0', edgecolor='white',
            linewidth=0.6, alpha=0.85)
    ax.axvline(accs.mean(), color='#C44E52', linestyle='--', linewidth=1.0,
               label=f'Mean = {accs.mean():.2%}')
    ax.set_xlabel('Per-sample Exact Accuracy')
    ax.set_ylabel('Count')
    title = 'Per-Sample Accuracy Distribution'
    if label:
        title += f'  ({label})'
    ax.set_title(title)
    ax.legend(loc='upper left', framealpha=0.7)
    ax.grid(True, linewidth=0.3, alpha=0.4)
    fig.tight_layout()

    suffix = f'_{label}' if label else ''
    save_path = os.path.join(out_dir, f'per_sample_accuracy{suffix}.png')
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


# ===================================================================
# 9. Per-word-type exact-accuracy bar chart
# ===================================================================

def save_per_word_type_accuracy(
    per_word_type_exact: Dict[str, List[float]],
    out_dir: str,
    label: str = '',
) -> Optional[str]:
    """Bar chart of mean per-sample exact accuracy grouped by word type.

    *per_word_type_exact* is a dict mapping word-type string (e.g. ``word1_12``)
    to a list of per-sample exact accuracies (as returned by
    :func:`evaluate_testmode_metrics` when ``word_types`` is supplied).
    """
    if not per_word_type_exact:
        return None
    _apply_style()
    ensure_dir(out_dir)

    def _sort_key(name: str):
        # Natural-sort: word<N>_<M> -> (N, M); fallback to string.
        import re
        m = re.match(r'word(\d+)_(\d+)', name)
        if m:
            return (int(m.group(1)), int(m.group(2)))
        return (float('inf'), name)

    names = sorted(per_word_type_exact.keys(), key=_sort_key)
    means = np.asarray([float(np.mean(per_word_type_exact[n])) for n in names])
    counts = [len(per_word_type_exact[n]) for n in names]
    overall = float(np.mean(np.concatenate([
        np.asarray(per_word_type_exact[n], dtype=np.float32) for n in names
    ])))

    fig_w = max(6.0, 0.28 * len(names) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, 4))
    x = np.arange(len(names))
    bars = ax.bar(x, means, color='#4C72B0', edgecolor='white', linewidth=0.6, alpha=0.88)
    ax.axhline(overall, color='#C44E52', linestyle='--', linewidth=1.0,
               label=f'Overall mean = {overall:.2%}')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=60, ha='right', fontsize=8)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel('Mean Exact Accuracy')
    title = 'Per-Word-Type Exact Accuracy'
    if label:
        title += f'  ({label})'
    ax.set_title(title)
    ax.grid(True, axis='y', linewidth=0.3, alpha=0.4)
    ax.legend(loc='upper right', framealpha=0.7)
    # annotate sample count above each bar
    for rect, c in zip(bars, counts):
        ax.text(rect.get_x() + rect.get_width() / 2.0, rect.get_height() + 0.01,
                f'n={c}', ha='center', va='bottom', fontsize=7, color='#333333')
    fig.tight_layout()

    suffix = f'_{label}' if label else ''
    save_path = os.path.join(out_dir, f'per_word_type_accuracy{suffix}.png')
    fig.savefig(save_path)
    plt.close(fig)
    return save_path
