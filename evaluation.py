"""Evaluation utilities for PRI reconstruction.

Contains token parsing, test-mode metrics, diversity analysis, majority-vote
reconstruction, and helper functions for final-report generation.
"""
from __future__ import annotations

import os
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Target-token parsing (variable-length PRI support)
# ---------------------------------------------------------------------------

def parse_target_tokens(pred_ids_1d, quantizer) -> List[int]:
    """Parse predicted target PRI token IDs from a single sequence.

    Locates the first SEP, then collects data tokens after SEP until the
    first END token.  This is the core routine for *test-time* evaluation
    where the target/pad boundary is unknown and must be inferred from the
    generated END marker – enabling variable-length PRI reconstruction.
    """
    ids = pred_ids_1d.cpu().tolist() if isinstance(pred_ids_1d, torch.Tensor) else list(pred_ids_1d)
    sep_id = quantizer.sep_token_id
    end_id = quantizer.end_token_id
    start_id = quantizer.start_token_id

    try:
        sep_pos = ids.index(sep_id)
    except ValueError:
        return []

    target_start = sep_pos + 1
    if target_start < len(ids) and ids[target_start] == start_id:
        target_start += 1

    result: List[int] = []
    for i in range(target_start, len(ids)):
        if ids[i] == end_id:
            break
        if ids[i] >= quantizer.offset:
            result.append(ids[i])
    return result


def parse_gt_target_tokens(input_ids_1d, input_mask_1d, quantizer) -> List[int]:
    """Extract ground-truth target PRI token IDs via the known mask."""
    ids = input_ids_1d.cpu().numpy()
    mask = input_mask_1d.cpu().numpy()
    target_idx = np.where(mask == 1)[0]
    return [int(ids[i]) for i in target_idx if ids[i] >= quantizer.offset]


# ---------------------------------------------------------------------------
# Extract PRI values (source / GT target / predicted target)
# ---------------------------------------------------------------------------

def extract_target_pri(input_ids, input_mask, pred_ids, quantizer, test_mode=False):
    """Extract target-segment PRI values from token id tensors.

    When *test_mode=True* the target boundary is parsed from the predicted
    END token (variable-length support) instead of using the known mask.
    Returns (clean_pri, pred_pri, src_pri) as lists of floats.
    """
    ids_np = input_ids.cpu().numpy()

    sep_id = quantizer.sep_token_id

    # ---- Source segment (always known) ----
    sep_positions = np.where(ids_np == sep_id)[0]
    if len(sep_positions) > 0:
        src_end = int(sep_positions[0])
        src_range = np.arange(1, src_end)
        src_ids = [int(ids_np[i]) for i in src_range if ids_np[i] >= quantizer.offset]
    else:
        src_ids = []

    # ---- Ground-truth target (from known mask) ----
    gt_target_ids = parse_gt_target_tokens(input_ids, input_mask, quantizer)

    # ---- Predicted target (always parsed via END token) ----
    pred_target_ids = parse_target_tokens(pred_ids, quantizer)

    clean_pri = quantizer.decode_ids(gt_target_ids, remove_special=False)
    pred_pri = quantizer.decode_ids(pred_target_ids, remove_special=False)
    src_pri = quantizer.decode_ids(src_ids, remove_special=False)
    return clean_pri, pred_pri, src_pri


# ---------------------------------------------------------------------------
# Test-mode evaluation metrics
# ---------------------------------------------------------------------------

def evaluate_testmode_metrics(
    model, dataloader, quantizer,
    sampling_method='ddpm', ddim_steps=None, ddim_eta=0.0,
    tolerance=1,
) -> Dict:
    """Evaluate using *test-mode* reconstruction (no GT target/pad mask).

    Returns dict:
      exact_acc      – exact token-match accuracy (auto-parsed)
      tol_acc        – token-match within ±*tolerance* bins
      mae            – mean absolute error on decoded PRI values (µs)
      len_match_rate – fraction of samples whose predicted length equals GT
      per_sample_exact – per-sample exact accuracy list
    """
    model.eval()
    total_exact = total_tol = total_tokens = 0
    total_mae = 0.0
    mae_count = 0
    len_matches = sample_count = 0
    per_sample_exact: List[float] = []

    with torch.no_grad():
        for batch in dataloader:
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
                gt_tok = parse_gt_target_tokens(input_ids[i], input_mask[i], quantizer)
                pred_tok = parse_target_tokens(pred_ids[i], quantizer)
                if not gt_tok:
                    continue
                sample_count += 1

                gt_len, pred_len = len(gt_tok), len(pred_tok)
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

                gv = quantizer.decode_ids(gt_tok, remove_special=False)
                pv = quantizer.decode_ids(pred_tok, remove_special=False)
                mv = min(len(gv), len(pv))
                if mv > 0:
                    for g, p in zip(gv[:mv], pv[:mv]):
                        total_mae += abs(g - p)
                        mae_count += 1

    return {
        'exact_acc': total_exact / max(total_tokens, 1),
        'tol_acc': total_tol / max(total_tokens, 1),
        'mae': total_mae / max(mae_count, 1),
        'len_match_rate': len_matches / max(sample_count, 1),
        'per_sample_exact': per_sample_exact,
    }


# ---------------------------------------------------------------------------
# Diffusion diversity evaluation
# ---------------------------------------------------------------------------

def evaluate_diversity(
    model, dataloader, quantizer, num_samples=10,
    sampling_method='ddpm', ddim_steps=None, ddim_eta=0.0,
) -> Dict:
    """Run *num_samples* independent reconstructions and quantify diversity.

    Returns dict:
      avg_diversity        – mean # unique predictions per target position
      avg_agreement        – fraction of positions where all runs agree
      per_sample_diversity – list
      per_sample_agreement – list
      all_predictions      – [(gt_tokens, [pred_run1, …, pred_runN]), …]
    """
    model.eval()
    diversities: List[float] = []
    agreements: List[float] = []
    all_predictions: list = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(model.device)
            input_mask = batch['input_mask'].to(model.device)

            multi_pred = []
            for _ in range(num_samples):
                out = model.reconstruct(
                    input_ids, input_mask,
                    use_rounding=True,
                    sampling_method=sampling_method,
                    ddim_steps=ddim_steps,
                    ddim_eta=ddim_eta,
                    test_mode=True,
                )
                multi_pred.append(out['pred_ids'])

            B = input_ids.size(0)
            for i in range(B):
                gt_tok = parse_gt_target_tokens(input_ids[i], input_mask[i], quantizer)
                if not gt_tok:
                    continue

                sample_preds = [parse_target_tokens(multi_pred[r][i], quantizer)
                                for r in range(num_samples)]
                T = len(gt_tok)
                unique_counts = []
                agree_count = 0
                for j in range(T):
                    vals = {pred[j] if j < len(pred) else -1 for pred in sample_preds}
                    unique_counts.append(len(vals))
                    if len(vals) == 1:
                        agree_count += 1

                diversities.append(sum(unique_counts) / T if T else 0.0)
                agreements.append(agree_count / T if T else 0.0)
                all_predictions.append((gt_tok, sample_preds))

    return {
        'avg_diversity': sum(diversities) / max(len(diversities), 1),
        'avg_agreement': sum(agreements) / max(len(agreements), 1),
        'per_sample_diversity': diversities,
        'per_sample_agreement': agreements,
        'all_predictions': all_predictions,
    }


# ---------------------------------------------------------------------------
# Majority-vote reconstruction (self-consistency)
# ---------------------------------------------------------------------------

def evaluate_majority_vote(
    model, dataloader, quantizer, num_votes=10,
    sampling_method='ddpm', ddim_steps=None, ddim_eta=0.0,
    tolerance=1,
) -> Dict:
    """Reconstruct via majority voting across *num_votes* runs.

    Returns the same metric dict as *evaluate_testmode_metrics*.
    """
    model.eval()
    total_exact = total_tol = total_tokens = 0
    total_mae = 0.0
    mae_count = 0
    len_matches = sample_count = 0
    per_sample_exact: List[float] = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(model.device)
            input_mask = batch['input_mask'].to(model.device)

            multi_pred = []
            for _ in range(num_votes):
                out = model.reconstruct(
                    input_ids, input_mask,
                    use_rounding=True,
                    sampling_method=sampling_method,
                    ddim_steps=ddim_steps,
                    ddim_eta=ddim_eta,
                    test_mode=True,
                )
                multi_pred.append(out['pred_ids'])

            B = input_ids.size(0)
            for i in range(B):
                gt_tok = parse_gt_target_tokens(input_ids[i], input_mask[i], quantizer)
                if not gt_tok:
                    continue
                sample_count += 1

                all_preds = [parse_target_tokens(multi_pred[r][i], quantizer)
                             for r in range(num_votes)]

                # Majority vote: median length then per-position mode
                pred_lens = sorted(len(p) for p in all_preds)
                voted_len = pred_lens[len(pred_lens) // 2]
                voted_tokens: List[int] = []
                for j in range(voted_len):
                    cands = [p[j] for p in all_preds if j < len(p)]
                    if cands:
                        voted_tokens.append(Counter(cands).most_common(1)[0][0])

                gt_len, pred_len = len(gt_tok), len(voted_tokens)
                if gt_len == pred_len:
                    len_matches += 1

                min_l = min(gt_len, pred_len)
                max_l = max(gt_len, pred_len)
                ec = sum(1 for g, p in zip(gt_tok[:min_l], voted_tokens[:min_l]) if g == p)
                tc = sum(1 for g, p in zip(gt_tok[:min_l], voted_tokens[:min_l]) if abs(g - p) <= tolerance)

                total_exact += ec
                total_tol += tc
                total_tokens += max_l
                per_sample_exact.append(ec / max_l if max_l > 0 else 0.0)

                gv = quantizer.decode_ids(gt_tok, remove_special=False)
                pv = quantizer.decode_ids(voted_tokens, remove_special=False)
                mv = min(len(gv), len(pv))
                if mv > 0:
                    for g, p in zip(gv[:mv], pv[:mv]):
                        total_mae += abs(g - p)
                        mae_count += 1

    return {
        'exact_acc': total_exact / max(total_tokens, 1),
        'tol_acc': total_tol / max(total_tokens, 1),
        'mae': total_mae / max(mae_count, 1),
        'len_match_rate': len_matches / max(sample_count, 1),
        'per_sample_exact': per_sample_exact,
    }


# ---------------------------------------------------------------------------
# Confusion matrix data collection
# ---------------------------------------------------------------------------

def collect_confusion_data(model, dataloader, quantizer, sampling_method, ddim_steps, ddim_eta):
    """Collect (y_true, y_pred) tensors for confusion matrix plotting.

    Returns (y_true, y_pred) as concatenated 1-D tensors, or (None, None)
    if no valid tokens were found.
    """
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in dataloader:
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
                gt_tok = parse_gt_target_tokens(input_ids[i], input_mask[i], quantizer)
                pred_tok = parse_target_tokens(pred_ids[i], quantizer)
                min_l = min(len(gt_tok), len(pred_tok))
                if min_l > 0:
                    y_true.append(torch.tensor(gt_tok[:min_l]))
                    y_pred.append(torch.tensor(pred_tok[:min_l]))

    if not y_true:
        return None, None
    return torch.cat(y_true, dim=0), torch.cat(y_pred, dim=0)
