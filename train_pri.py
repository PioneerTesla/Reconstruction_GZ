from __future__ import annotations
import os
import re
import random
import json
import sys
import math
from typing import Optional, Tuple, List, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from pri_tokenizer import PRIQuantizer, QuantizerConfig
from pri_dataset import PRISample, PRIDiffuSeqDataset, PRICollator
from model import PRIDiffuSeq, PRIDiffuSeqConfig, EMAModel
from utlis import get_clean_pri_range, get_pri_range, create_argparser, load_defaults_config, choose_training_schedule
from evaluation import (
    parse_target_tokens,
    parse_gt_target_tokens,
    extract_target_pri,
    evaluate_testmode_metrics,
    evaluate_diversity,
    evaluate_majority_vote,
    collect_confusion_data,
)
from visualization import (
    save_codebook_visualization,
    save_diffusion_denoise_visualization,
    save_training_curves,
    save_reconstruction_comparison,
    save_confusion_matrix,
    save_diversity_visualization,
    save_diversity_mean_variance,
    save_per_sample_accuracy_histogram,
)


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
        self.console.write(data)
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


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> str:
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


def choose_trace_steps(diff_steps: int, ddim_steps: Optional[int], sampling_method: str) -> List[int]:
    """Return the actual timestep indices used by the sampler."""
    if sampling_method.lower().strip() == 'ddim' and ddim_steps is not None and ddim_steps > 0 and ddim_steps < diff_steps:
        steps = np.linspace(0, diff_steps - 1, ddim_steps, dtype=np.int64)
        return sorted(set(steps.tolist()))
    return list(range(diff_steps))


def build_quantizer(
    quantizer_mode: str = 'uniform',
    quantizer_bins: int = 1000,
    quantizer_min: Optional[float] = None,
    quantizer_max: Optional[float] = None,
    clean_path: str = 'dataset/Ground_Truth',
    obs_path: str = 'dataset/Miss',
) -> Tuple[PRIQuantizer, float, float, int]:
    quantizer_mode = quantizer_mode.lower().strip()

    if quantizer_mode == 'uniform':
        clean_min, clean_max = get_clean_pri_range(clean_path)
        obs_min, obs_max = get_pri_range(obs_path)
        data_min = min(clean_min, obs_min)
        data_max = max(clean_max, obs_max)

        base_min = data_min if quantizer_min is None else float(quantizer_min)
        base_max = data_max if quantizer_max is None else float(quantizer_max)
        min_value = base_min
        max_value = base_max

        requested_bins = int(quantizer_bins)
        if requested_bins <= 1:
            requested_bins = max(2, int(math.ceil(max_value - min_value)) + 1)
        unit_resolution_bins = int(math.ceil(max_value - min_value)) + 1
        bins = max(requested_bins, unit_resolution_bins)

        quantizer = PRIQuantizer(
            QuantizerConfig(
                mode='uniform',
                min_value=min_value,
                max_value=max_value,
                num_bins=bins,
                add_special_tokens=True,
                snap_tolerance=None,
                key_start=1,
            )
        )
        return quantizer, min_value, max_value, bins

    raise ValueError(f"Unsupported quantizer_mode: {quantizer_mode}")


def _extract_word_type(filename: str) -> str:
    """Extract word type from filename, e.g. 'word1_12_1016.pt' -> 'word1_12'."""
    m = re.match(r'(word\d+_\d+)', filename)
    return m.group(1) if m else filename


def build_demo_loader(
    scene: str = 'Miss',
    seq_len: int = 250,
    bs: int = 64,
    quantizer_mode: str = 'uniform',
    quantizer_bins: int = 1000,
    quantizer_min: Optional[float] = None,
    quantizer_max: Optional[float] = None,
    max_train_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
    root: str = 'dataset',
):
    samples = []
    sample_filenames: List[str] = []
    clean_data_path = os.path.join(root, 'Ground_Truth')
    obs_data_path = os.path.join(root, scene)

    for file in sorted(os.listdir(clean_data_path)):
        if not file.endswith('.pt'):
            continue
        clean = torch.load(os.path.join(clean_data_path, file), weights_only=False)['seq'].tolist()
        obs = torch.load(os.path.join(obs_data_path, file), weights_only=False)['seq'].tolist()
        samples.append(PRISample(observed_pri=obs, clean_pri=clean))
        sample_filenames.append(file)

    n_samples = len(samples)
    all_nums = list(range(0, n_samples))
    random.shuffle(all_nums)

    train_samples = [samples[i] for i in all_nums[: int(0.9 * n_samples)]]
    test_samples = [samples[i] for i in all_nums[int(0.9 * n_samples):]]

    if max_train_samples is not None:
        train_samples = train_samples[:max_train_samples]
    if max_test_samples is not None:
        test_samples = test_samples[:max_test_samples]

    quantizer, min_value, max_value, actual_bins = build_quantizer(
        quantizer_mode=quantizer_mode,
        quantizer_bins=quantizer_bins,
        quantizer_min=quantizer_min,
        quantizer_max=quantizer_max,
        clean_path=clean_data_path,
        obs_path=obs_data_path,
    )

    train_ds = PRIDiffuSeqDataset(train_samples, quantizer=quantizer, seq_len=seq_len)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=PRICollator())

    test_ds = PRIDiffuSeqDataset(test_samples, quantizer=quantizer, seq_len=seq_len)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, collate_fn=PRICollator())

    # Build word-type map for test set (index in test_ds → word type)
    test_word_types = [_extract_word_type(sample_filenames[all_nums[int(0.9 * n_samples) + i]])
                       for i in range(len(test_samples))]

    return train_loader, test_loader, quantizer, min_value, max_value, actual_bins, test_word_types


def save_epoch_visuals(model, batch, quantizer, epoch, out_dir, sampling_method, ddim_steps, ddim_eta,
                       denoise_mode='combined', denoise_select_steps=None):
    """Save codebook distribution and diffusion denoising visualizations.

    Args:
        denoise_mode: 'combined' – all steps in one image (default);
                      'individual' – one PNG per step;
                      'select' – only show steps listed in *denoise_select_steps*.
        denoise_select_steps: list of timestep indices to show (used when
                              denoise_mode='select').
    """
    os.makedirs(out_dir, exist_ok=True)
    input_ids = batch['input_ids'].to(model.device)
    input_mask = batch['input_mask'].to(model.device)

    # 1. Codebook embedding distribution
    cb_path = save_codebook_visualization(
        model.model.word_embedding.weight,
        offset=quantizer.offset,
        num_pri_tokens=quantizer.num_pri_tokens,
        epoch=epoch,
        out_dir=out_dir,
    )
    print(f'[viz] epoch={epoch} codebook map saved -> {cb_path}')

    # 2. Diffusion denoising process
    clean_embeds = model.model.get_embeds(input_ids)
    trace_steps = choose_trace_steps(model.cfg.diffusion_steps, ddim_steps, sampling_method)
    recon_out = model.reconstruct(
        input_ids, input_mask,
        sampling_method=sampling_method,
        ddim_steps=ddim_steps,
        ddim_eta=ddim_eta,
        return_trace=True,
        trace_steps=trace_steps,
    )
    paths = save_diffusion_denoise_visualization(
        recon_out['trace'], clean_embeds, input_mask, epoch, out_dir,
        mode=denoise_mode, select_steps=denoise_select_steps,
    )
    if isinstance(paths, list):
        for p in paths:
            print(f'[viz] epoch={epoch} diffusion denoise saved -> {p}')
    elif paths:
        print(f'[viz] epoch={epoch} diffusion denoise saved -> {paths}')


def save_final_confusion_matrix(model, dataloader, quantizer, out_dir, sampling_method, ddim_steps, ddim_eta):
    """Collect confusion data via evaluation and plot."""
    y_true, y_pred = collect_confusion_data(model, dataloader, quantizer, sampling_method, ddim_steps, ddim_eta)
    if y_true is None or y_pred is None:
        print('[viz] confusion matrix skipped because no valid target tokens were found.')
        return None
    labels = torch.unique(torch.cat([y_true, y_pred], dim=0)).cpu().numpy()
    save_path = os.path.join(out_dir, 'confusion_matrix.png')
    save_confusion_matrix(y_true, y_pred, labels=labels, save_path=save_path, normalize=True)
    print(f'[viz] confusion matrix saved -> {save_path}')
    return save_path


def main():
    config_path = 'config.json'
    args = create_argparser(config_path).parse_args()
    model_name = (getattr(args, 'model_name', 'DiffSeqPRI') or 'DiffSeqPRI') + '_' + args.scene + '_' + args.root
    set_seed(int(getattr(args, 'seed', 42)))

    log_path = os.path.join('CheckPoint', model_name, 'log.txt')
    best_model_path = os.path.join('CheckPoint', model_name, 'best_model.pth')
    latest_model_path = os.path.join('CheckPoint', model_name, 'latest_model.pth')
    vis_dir = os.path.join('CheckPoint', model_name, 'visuals')
    os.makedirs(os.path.dirname(latest_model_path), exist_ok=True)

    dual_output = DualOutput(log_path)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = dual_output
    sys.stderr = dual_output

    try:
        quantizer_min = normalize_optional_float(getattr(args, 'quantizer_min', None))
        quantizer_max = normalize_optional_float(getattr(args, 'quantizer_max', None))
        quantizer_mode = getattr(args, 'quantizer_mode', 'uniform')
        quantizer_bins = int(getattr(args, 'quantizer_bins', 1000))
        sampling_method = getattr(args, 'sampling_method', 'ddim')
        ddim_steps = int(getattr(args, 'ddim_steps', 5))
        ddim_eta = float(getattr(args, 'ddim_eta', 0.0))
        device_name = resolve_device(getattr(args, 'device', 'cpu'))
        max_train_samples = normalize_optional_int(getattr(args, 'max_train_samples', None))
        max_test_samples = normalize_optional_int(getattr(args, 'max_test_samples', None))

        print('=== 训练配置 ===')
        print(json.dumps(load_defaults_config(config_path), indent=4, ensure_ascii=False))

        train_loader, test_loader, quantizer, q_min, q_max, actual_bins, test_word_types = build_demo_loader(
            scene=args.scene,
            seq_len=args.spliced_seq_length,
            bs=args.batch_size,
            quantizer_mode=quantizer_mode,
            quantizer_bins=quantizer_bins,
            quantizer_min=quantizer_min,
            quantizer_max=quantizer_max,
            max_train_samples=max_train_samples,
            max_test_samples=max_test_samples,
            root=args.root
        )

        print(f'quantizer_mode={quantizer_mode}')
        print(f'quantizer_bins(requested)={quantizer_bins}')
        print(f'quantizer_bins(actual)={actual_bins}')
        print(f'quantizer_value_range=[{q_min:.6f}, {q_max:.6f}]')
        print(f'quantizer_key_range=[{quantizer.key_start}, {quantizer.key_end}]')
        print(f'vocab_size(with special tokens)={quantizer.vocab_size}')
        print(f'special_tokens: start={quantizer.start_token_id}, end={quantizer.end_token_id}, unk={quantizer.unk_token_id}, pad={quantizer.pad_token_id}, sep={quantizer.sep_token_id}')
        print(f'sampling_method={sampling_method}')
        if sampling_method.lower() == 'ddim':
            print(f'ddim_steps={ddim_steps}, ddim_eta={ddim_eta}')

        p_full_target = float(getattr(args, 'p_full_target', 0.0))
        loss_mode = str(getattr(args, 'loss_mode', 'mse+ce'))
        print(f'p_full_target={p_full_target}')
        print(f'loss_mode={loss_mode}')

        cfg = PRIDiffuSeqConfig(
            vocab_size=quantizer.vocab_size,
            seq_len=args.spliced_seq_length,
            hidden_dim=128,
            model_dim=256,
            diffusion_steps=args.diff_steps,
            beta_schedule=args.noise_schedule,
            ce_weight=float(getattr(args, 'ce_weight', 0.5)),
            device=device_name,
            p_full_target=p_full_target,
            loss_mode=loss_mode,
        )
        model = PRIDiffuSeq(cfg)
        ema = EMAModel(model, decay=0.9999)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=50, T_mult=2, eta_min=1e-6
        )

        vis_batch = next(iter(test_loader))
        viz_epochs = {0, max(0, args.epochs // 2), max(0, args.epochs - 1)}

        # Training history for plotting
        history: Dict[str, List[float]] = {
            'train_loss': [], 'test_loss': [],
            'train_mse': [], 'train_ce': [],
            'ap_exact_acc': [], 'ap_tol1_acc': [], 'ap_mae': [],
            'len_match': [],
        }

        best_test_loss = float('inf')
        for epoch in range(args.epochs):
            train_stats = model.fit_epoch(train_loader, optimizer, schedual=scheduler)
            ema.update(model)
            test_stats = model.evaluate(test_loader)

            test_loss_now = test_stats['loss']

            # Use EMA model for evaluation metrics
            ema.apply_shadow(model)
            tm = evaluate_testmode_metrics(
                model, test_loader, quantizer,
                sampling_method=sampling_method,
                ddim_steps=ddim_steps,
                ddim_eta=ddim_eta,
                tolerance=1,
            )
            ema.restore(model)

            # Record history
            history['train_loss'].append(train_stats['loss'])
            history['test_loss'].append(test_loss_now)
            history['train_mse'].append(train_stats['mse'])
            history['train_ce'].append(train_stats['ce'])
            history['ap_exact_acc'].append(tm['exact_acc'])
            history['ap_tol1_acc'].append(tm['tol_acc'])
            history['ap_mae'].append(tm['mae'])
            history['len_match'].append(tm['len_match_rate'])

            print(
                f'epoch={epoch} train_loss={train_stats["loss"]:.6f} test_loss={test_loss_now:.6f}',
                f'train_mse={train_stats["mse"]:.6f} train_ce={train_stats["ce"]:.6f}',
                f'ap_exact={tm["exact_acc"]:.4%} ap_tol1={tm["tol_acc"]:.4%} mae={tm["mae"]:.2f}',
                f'len_match={tm["len_match_rate"]:.4%} lr={optimizer.param_groups[0]["lr"]:.6e}',
            )

            torch.save(model.state_dict(), latest_model_path)

            if epoch in viz_epochs:
                save_epoch_visuals(model, vis_batch, quantizer, epoch, vis_dir, sampling_method, ddim_steps, ddim_eta)

            if test_loss_now < best_test_loss:
                best_test_loss = test_loss_now
                torch.save(model.state_dict(), best_model_path)
                torch.save(ema.state_dict(), best_model_path.replace('.pth', '_ema.pth'))
                print(f'New best model saved at epoch {epoch} with test_loss={best_test_loss:.6f}')

        # Save training curves
        curves_path = save_training_curves(history, vis_dir)
        print(f'[viz] training curves saved -> {curves_path}')

        # Load best model for final evaluation
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path, map_location=model.device))

        # Save confusion matrix
        save_final_confusion_matrix(model, test_loader, quantizer, vis_dir, sampling_method, ddim_steps, ddim_eta)

        # ----- Final test-mode evaluation -----
        print('\n=== Final Test-Mode Evaluation ===')
        final_tm = evaluate_testmode_metrics(
            model, test_loader, quantizer,
            sampling_method=sampling_method,
            ddim_steps=ddim_steps,
            ddim_eta=ddim_eta,
            tolerance=1,
        )
        print(f'  exact_acc  = {final_tm["exact_acc"]:.4%}')
        print(f'  tol1_acc   = {final_tm["tol_acc"]:.4%}')
        print(f'  MAE (\u00b5s)  = {final_tm["mae"]:.4f}')
        print(f'  len_match  = {final_tm["len_match_rate"]:.4%}')

        # Per-sample accuracy histogram
        if final_tm['per_sample_exact']:
            hist_path = save_per_sample_accuracy_histogram(
                final_tm['per_sample_exact'], out_dir=vis_dir, label='single',
            )
            print(f'[viz] per-sample accuracy histogram saved -> {hist_path}')

        # ----- Diversity evaluation -----
        print('\n=== Diversity Evaluation (10 runs) ===')
        div_results = evaluate_diversity(
            model, test_loader, quantizer, num_samples=10,
            sampling_method=sampling_method,
            ddim_steps=ddim_steps,
            ddim_eta=ddim_eta,
        )
        print(f'  avg_diversity  = {div_results["avg_diversity"]:.4f}')
        print(f'  avg_agreement  = {div_results["avg_agreement"]:.4%}')

        if div_results['all_predictions']:
            div_path = save_diversity_visualization(
                div_results['all_predictions'], quantizer, out_dir=vis_dir,
            )
            print(f'[viz] diversity visualization saved -> {div_path}')
            mv_path = save_diversity_mean_variance(
                div_results['all_predictions'], quantizer, out_dir=vis_dir,
            )
            print(f'[viz] diversity mean±variance saved -> {mv_path}')

        # ----- Majority-vote evaluation -----
        print('\n=== Majority Vote Evaluation (10 votes) ===')
        vote_tm = evaluate_majority_vote(
            model, test_loader, quantizer, num_votes=10,
            sampling_method=sampling_method,
            ddim_steps=ddim_steps,
            ddim_eta=ddim_eta,
            tolerance=1,
        )
        print(f'  exact_acc  = {vote_tm["exact_acc"]:.4%}')
        print(f'  tol1_acc   = {vote_tm["tol_acc"]:.4%}')
        print(f'  MAE (\u00b5s)  = {vote_tm["mae"]:.4f}')
        print(f'  len_match  = {vote_tm["len_match_rate"]:.4%}')

        if vote_tm['per_sample_exact']:
            hist_path_v = save_per_sample_accuracy_histogram(
                vote_tm['per_sample_exact'], out_dir=vis_dir, label='majority_vote',
            )
            print(f'[viz] majority-vote accuracy histogram saved -> {hist_path_v}')

        # Save reconstruction comparison: one sample per word type
        model.eval()
        seen_word_types: set = set()
        global_sample_idx = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
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
                B = input_ids.size(0)
                for i in range(B):
                    if global_sample_idx >= len(test_word_types):
                        break
                    wt = test_word_types[global_sample_idx]
                    global_sample_idx += 1
                    if wt in seen_word_types:
                        continue
                    seen_word_types.add(wt)
                    clean_pri, pred_pri, src_pri = extract_target_pri(
                        input_ids[i], input_mask[i], out['pred_ids'][i], quantizer,
                        test_mode=True,
                    )
                    if clean_pri and pred_pri:
                        p = save_reconstruction_comparison(
                            clean_pri, pred_pri, src_pri,
                            sample_idx=wt, out_dir=vis_dir,
                        )
                        print(f'[viz] reconstruction comparison ({wt}) saved -> {p}')

        # Print a demo reconstruction
        batch = next(iter(test_loader))
        input_ids = batch['input_ids'].to(model.device)
        input_mask = batch['input_mask'].to(model.device)
        out = model.reconstruct(
            input_ids, input_mask,
            sampling_method=sampling_method,
            ddim_steps=ddim_steps,
            ddim_eta=ddim_eta,
            test_mode=True,
        )
        clean_pri, pred_pri, src_pri = extract_target_pri(
            input_ids[0], input_mask[0], out['pred_ids'][0], quantizer,
            test_mode=True,
        )

        print('\n=== Reconstruction Demo ===')
        print(f'source PRI (observed, decoded): {src_pri[:20]}...')
        print(f'target PRI (clean, decoded):    {clean_pri[:20]}...')
        print(f'predicted PRI (reconstructed):  {pred_pri[:20]}...')

        print(f'All console outputs have been saved to: {log_path}')
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        dual_output.close()


if __name__ == '__main__':
    main()
