"""Final-report generation: confusion matrix, test-mode & EMA evaluation,
per-sample / per-word-type plots, diversity, majority vote, reconstruction
demo. Kept separate from `trainer.py` so the training loop stays short."""
from __future__ import annotations
import os
import re
from typing import Dict, List, Optional

import numpy as np
import torch

from evaluation import (
    collect_confusion_data,
    evaluate_diversity,
    evaluate_majority_vote,
    evaluate_testmode_metrics,
    extract_target_pri,
)
from utils import choose_trace_steps
from visualization import (
    save_codebook_visualization,
    save_confusion_matrix,
    save_diffusion_denoise_visualization,
    save_diversity_mean_variance,
    save_diversity_visualization,
    save_per_sample_accuracy_histogram,
    save_per_word_type_accuracy,
    save_reconstruction_comparison,
    save_training_curves,
)


# ---------------------------------------------------------------------------
# Mid-training visuals
# ---------------------------------------------------------------------------

def save_epoch_visuals(
    model, batch, quantizer, epoch, out_dir,
    sampling_method, ddim_steps, ddim_eta,
    denoise_mode: str = 'combined', denoise_select_steps: Optional[List[int]] = None,
):
    """Codebook scatter + diffusion-denoising trace for a single epoch."""
    os.makedirs(out_dir, exist_ok=True)
    input_ids = batch['input_ids'].to(model.device)
    input_mask = batch['input_mask'].to(model.device)

    cb_path = save_codebook_visualization(
        model.model.word_embedding.weight,
        offset=quantizer.offset,
        num_pri_tokens=quantizer.num_pri_tokens,
        epoch=epoch, out_dir=out_dir,
    )
    print(f'[viz] epoch={epoch} codebook map saved -> {cb_path}')

    clean_embeds = model.model.get_embeds(input_ids)
    trace_steps = choose_trace_steps(model.cfg.diffusion_steps, ddim_steps, sampling_method)
    recon = model.reconstruct(
        input_ids, input_mask,
        sampling_method=sampling_method, ddim_steps=ddim_steps, ddim_eta=ddim_eta,
        return_trace=True, trace_steps=trace_steps,
    )
    paths = save_diffusion_denoise_visualization(
        recon['trace'], clean_embeds, input_mask, epoch, out_dir,
        mode=denoise_mode, select_steps=denoise_select_steps,
    )
    if isinstance(paths, list):
        for p in paths:
            print(f'[viz] epoch={epoch} diffusion denoise saved -> {p}')
    elif paths:
        print(f'[viz] epoch={epoch} diffusion denoise saved -> {paths}')


def save_final_confusion_matrix(model, dataloader, quantizer, out_dir,
                                sampling_method, ddim_steps, ddim_eta):
    y_true, y_pred = collect_confusion_data(model, dataloader, quantizer,
                                            sampling_method, ddim_steps, ddim_eta)
    if y_true is None or y_pred is None:
        print('[viz] confusion matrix skipped: no valid target tokens found.')
        return None
    labels = torch.unique(torch.cat([y_true, y_pred], dim=0)).cpu().numpy()
    save_path = os.path.join(out_dir, 'confusion_matrix.png')
    save_confusion_matrix(y_true, y_pred, labels=labels, save_path=save_path, normalize=True)
    print(f'[viz] confusion matrix saved -> {save_path}')
    return save_path


# ---------------------------------------------------------------------------
# Final-report sections (each prints + saves a figure)
# ---------------------------------------------------------------------------

def _print_metrics(label: str, tm: dict) -> None:
    print(f'\n=== {label} ===')
    print(f'  exact_acc  = {tm["exact_acc"]:.4%}')
    print(f'  tol1_acc   = {tm["tol_acc"]:.4%}')
    print(f'  MAE (us)   = {tm["mae"]:.4f}')
    print(f'  len_match  = {tm["len_match_rate"]:.4%}')


def _print_per_word_type(per_wt: Dict[str, List[float]]) -> None:
    def key(n):
        m = re.match(r'word(\d+)_(\d+)', n)
        return (int(m.group(1)), int(m.group(2))) if m else (float('inf'), n)
    print('\n=== Per-Word-Type Exact Accuracy ===')
    for wt in sorted(per_wt.keys(), key=key):
        accs = per_wt[wt]
        print(f'  {wt:>16s}  n={len(accs):3d}  mean_exact={float(np.mean(accs)):.4%}')


def run_final_eval(
    model, ema, trainer_best_metric_ema_path,
    test_loader, quantizer, vis_dir,
    sampling_method, ddim_steps, ddim_eta,
    test_word_types: List[str],
) -> dict:
    """Test-mode evaluation on raw + EMA weights, with per-sample and
    per-word-type visuals. Returns the raw-model `final_tm` dict."""
    save_final_confusion_matrix(model, test_loader, quantizer, vis_dir,
                                sampling_method, ddim_steps, ddim_eta)

    final_tm = evaluate_testmode_metrics(
        model, test_loader, quantizer,
        sampling_method=sampling_method, ddim_steps=ddim_steps, ddim_eta=ddim_eta,
        tolerance=1, word_types=test_word_types,
    )
    _print_metrics('Final Test-Mode Evaluation (Raw Model)', final_tm)

    if trainer_best_metric_ema_path and os.path.exists(trainer_best_metric_ema_path):
        ema.load_state_dict(torch.load(trainer_best_metric_ema_path, map_location=model.device))
        ema.apply_shadow(model)
        ema_tm = evaluate_testmode_metrics(
            model, test_loader, quantizer,
            sampling_method=sampling_method, ddim_steps=ddim_steps, ddim_eta=ddim_eta,
            tolerance=1,
        )
        _print_metrics('Final Test-Mode Evaluation (EMA Model)', ema_tm)
        ema.restore(model)

    if final_tm['per_sample_exact']:
        p = save_per_sample_accuracy_histogram(final_tm['per_sample_exact'],
                                               out_dir=vis_dir, label='single')
        print(f'[viz] per-sample accuracy histogram saved -> {p}')

    pwt = final_tm.get('per_word_type_exact')
    if pwt:
        p = save_per_word_type_accuracy(pwt, out_dir=vis_dir, label='single')
        if p:
            print(f'[viz] per-word-type accuracy saved -> {p}')
        _print_per_word_type(pwt)

    return final_tm


def run_diversity_and_majority_vote(
    model, test_loader, quantizer, vis_dir,
    sampling_method, ddim_steps, ddim_eta,
    num_samples: int = 10, num_votes: int = 10,
) -> None:
    print('\n=== Diversity Evaluation (10 runs) ===')
    div = evaluate_diversity(
        model, test_loader, quantizer, num_samples=num_samples,
        sampling_method=sampling_method, ddim_steps=ddim_steps, ddim_eta=ddim_eta,
    )
    print(f'  avg_diversity  = {div["avg_diversity"]:.4f}')
    print(f'  avg_agreement  = {div["avg_agreement"]:.4%}')
    if div['all_predictions']:
        p = save_diversity_visualization(div['all_predictions'], quantizer, out_dir=vis_dir)
        print(f'[viz] diversity visualization saved -> {p}')
        p = save_diversity_mean_variance(div['all_predictions'], quantizer, out_dir=vis_dir)
        print(f'[viz] diversity mean±variance saved -> {p}')

    vote_tm = evaluate_majority_vote(
        model, test_loader, quantizer, num_votes=num_votes,
        sampling_method=sampling_method, ddim_steps=ddim_steps, ddim_eta=ddim_eta,
        tolerance=1,
    )
    _print_metrics(f'Majority Vote Evaluation ({num_votes} votes)', vote_tm)
    if vote_tm['per_sample_exact']:
        p = save_per_sample_accuracy_histogram(vote_tm['per_sample_exact'],
                                               out_dir=vis_dir, label='majority_vote')
        print(f'[viz] majority-vote accuracy histogram saved -> {p}')


def save_per_word_type_reconstructions(
    model, test_loader, quantizer, vis_dir, test_word_types,
    sampling_method, ddim_steps, ddim_eta,
) -> None:
    """Save one reconstruction figure per unique word type encountered."""
    model.eval()
    seen: set = set()
    idx = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(model.device)
            input_mask = batch['input_mask'].to(model.device)
            out = model.reconstruct(
                input_ids, input_mask, use_rounding=True,
                sampling_method=sampling_method, ddim_steps=ddim_steps, ddim_eta=ddim_eta,
                test_mode=True,
            )
            for i in range(input_ids.size(0)):
                if idx >= len(test_word_types):
                    return
                wt = test_word_types[idx]
                idx += 1
                if wt in seen:
                    continue
                seen.add(wt)
                clean_pri, pred_pri, src_pri = extract_target_pri(
                    input_ids[i], input_mask[i], out['pred_ids'][i], quantizer,
                    test_mode=True,
                )
                if clean_pri and pred_pri:
                    p = save_reconstruction_comparison(
                        clean_pri, pred_pri, src_pri, sample_idx=wt, out_dir=vis_dir)
                    print(f'[viz] reconstruction comparison ({wt}) saved -> {p}')


def print_reconstruction_demo(model, test_loader, quantizer, sampling_method, ddim_steps, ddim_eta):
    batch = next(iter(test_loader))
    input_ids = batch['input_ids'].to(model.device)
    input_mask = batch['input_mask'].to(model.device)
    out = model.reconstruct(
        input_ids, input_mask,
        sampling_method=sampling_method, ddim_steps=ddim_steps, ddim_eta=ddim_eta,
        test_mode=True,
    )
    clean_pri, pred_pri, src_pri = extract_target_pri(
        input_ids[0], input_mask[0], out['pred_ids'][0], quantizer, test_mode=True)
    print('\n=== Reconstruction Demo ===')
    print(f'source PRI (observed, decoded): {src_pri[:20]}...')
    print(f'target PRI (clean, decoded):    {clean_pri[:20]}...')
    print(f'predicted PRI (reconstructed):  {pred_pri[:20]}...')


# ---------------------------------------------------------------------------
# Top-level: run all sections
# ---------------------------------------------------------------------------

def run_full_report(
    model, ema, trainer,
    test_loader, quantizer, test_word_types,
    history: dict, vis_dir: str,
    sampling_method: str, ddim_steps: int, ddim_eta: float,
    *,
    final_eval_use_best_metric: bool = True,
) -> None:
    """Everything that happens after the training loop finishes."""
    curves_path = save_training_curves(history, vis_dir)
    print(f'[viz] training curves saved -> {curves_path}')

    final_ckpt = trainer.load_final_checkpoint(prefer_metric=final_eval_use_best_metric)
    ema_path = final_ckpt.replace('.pth', '_ema.pth') if final_ckpt else None

    run_final_eval(
        model, ema, ema_path, test_loader, quantizer, vis_dir,
        sampling_method, ddim_steps, ddim_eta, test_word_types,
    )
    run_diversity_and_majority_vote(
        model, test_loader, quantizer, vis_dir,
        sampling_method, ddim_steps, ddim_eta,
    )
    save_per_word_type_reconstructions(
        model, test_loader, quantizer, vis_dir, test_word_types,
        sampling_method, ddim_steps, ddim_eta,
    )
    print_reconstruction_demo(model, test_loader, quantizer,
                              sampling_method, ddim_steps, ddim_eta)
