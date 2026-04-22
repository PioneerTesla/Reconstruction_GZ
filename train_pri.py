"""Entry point for training PRIDiffuSeq on a single scene.

The file is intentionally short: heavy lifting lives in
  * data_loader.py — dataset / DataLoader / quantizer construction
  * trainer.py     — epoch loop, checkpointing, LR schedule, early stopping
  * reporting.py   — post-training evaluation and figure generation
"""
from __future__ import annotations
import json
import os
import sys

import torch

from data_loader import build_demo_loader
from model import EMAModel, PRIDiffuSeq, PRIDiffuSeqConfig
from reporting import run_full_report, save_epoch_visuals
from trainer import Trainer, TrainerPaths
from utils import (
    DualOutput,
    create_argparser,
    load_defaults_config,
    normalize_optional_float,
    normalize_optional_int,
    resolve_device,
    set_seed,
)


SAVE_ROOT = 'Checkpoint'


def _build_model_config(args, vocab_size: int, device: str) -> PRIDiffuSeqConfig:
    return PRIDiffuSeqConfig(
        vocab_size=vocab_size,
        seq_len=args.spliced_seq_length,
        hidden_dim=128,
        model_dim=256,
        diffusion_steps=args.diff_steps,
        beta_schedule=args.noise_schedule,
        ce_weight=float(getattr(args, 'ce_weight', 0.5)),
        device=device,
        p_full_target=float(getattr(args, 'p_full_target', 0.0)),
        loss_mode=str(getattr(args, 'loss_mode', 'mse+ce')),
        label_smoothing=float(getattr(args, 'label_smoothing', 0.0)),
        end_loss_weight=float(getattr(args, 'end_loss_weight', 0.0)),
        p_full_target_curriculum=bool(getattr(args, 'p_full_target_curriculum', False)),
        curriculum_warmup_epochs=int(getattr(args, 'curriculum_warmup_epochs', 200)),
    )


def main() -> None:
    config_path = 'config.json'
    args = create_argparser(config_path).parse_args()
    set_seed(int(getattr(args, 'seed', 42)))

    model_name = f'{getattr(args, "model_name", "DiffSeqPRI") or "DiffSeqPRI"}_{args.scene}_{args.root}'
    run_dir = os.path.join(SAVE_ROOT, model_name)
    paths = TrainerPaths.from_root(run_dir)

    dual = DualOutput(os.path.join(run_dir, 'log.txt'))
    orig_out, orig_err = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = dual

    try:
        print('=== 训练配置 ===')
        print(json.dumps(load_defaults_config(config_path), indent=4, ensure_ascii=False))

        device = resolve_device(getattr(args, 'device', 'cpu'))
        sampling_method = getattr(args, 'sampling_method', 'ddpm')
        ddim_steps = int(getattr(args, 'ddim_steps', 5))
        ddim_eta = float(getattr(args, 'ddim_eta', 0.0))

        # ---- data / quantizer ---------------------------------------------
        train_loader, test_loader, quantizer, q_min, q_max, actual_bins, test_word_types = build_demo_loader(
            scene=args.scene,
            seq_len=args.spliced_seq_length,
            bs=args.batch_size,
            quantizer_mode=getattr(args, 'quantizer_mode', 'uniform'),
            quantizer_bins=int(getattr(args, 'quantizer_bins', 1000)),
            quantizer_min=normalize_optional_float(getattr(args, 'quantizer_min', None)),
            quantizer_max=normalize_optional_float(getattr(args, 'quantizer_max', None)),
            max_train_samples=normalize_optional_int(getattr(args, 'max_train_samples', None)),
            max_test_samples=normalize_optional_int(getattr(args, 'max_test_samples', None)),
            root=args.root,
        )
        print(f'quantizer_bins(actual)={actual_bins}  '
              f'range=[{q_min:.6f}, {q_max:.6f}]  vocab_size={quantizer.vocab_size}')
        print(f'special_tokens: start={quantizer.start_token_id}, end={quantizer.end_token_id}, '
              f'unk={quantizer.unk_token_id}, pad={quantizer.pad_token_id}, sep={quantizer.sep_token_id}')
        print(f'sampling_method={sampling_method}  ddim_steps={ddim_steps}  ddim_eta={ddim_eta}')

        # ---- model / EMA / optimizer --------------------------------------
        model = PRIDiffuSeq(_build_model_config(args, quantizer.vocab_size, device))
        # EMA decay=0.999 converges after ~5000 steps (~360 epochs with 14 batches/epoch);
        # decay=0.9999 would retain ~25% random init even after 14000 steps.
        ema = EMAModel(model, decay=0.999)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)

        # ---- train --------------------------------------------------------
        vis_batch = next(iter(test_loader))
        viz_epochs = {0, max(0, args.epochs // 2), max(0, args.epochs - 1)}

        def _epoch_callback(model_, epoch_):
            if epoch_ in viz_epochs:
                save_epoch_visuals(model_, vis_batch, quantizer, epoch_, paths.vis_dir,
                                   sampling_method, ddim_steps, ddim_eta)

        trainer = Trainer(
            model, ema, optimizer, quantizer, train_loader, test_loader, paths,
            epochs=args.epochs,
            sampling_method=sampling_method, ddim_steps=ddim_steps, ddim_eta=ddim_eta,
            lr_warmup_epochs=int(getattr(args, 'lr_warmup_epochs', 200)),
            lr_plateau_patience=int(getattr(args, 'lr_plateau_patience', 10)),
            lr_plateau_factor=float(getattr(args, 'lr_plateau_factor', 0.5)),
            lr_plateau_min=float(getattr(args, 'lr_plateau_min', 1e-6)),
            use_cosine_lr=bool(getattr(args, 'use_cosine_lr', False)),
            cosine_lr_min=float(getattr(args, 'cosine_lr_min', 1e-6)),
            early_stop_patience=int(getattr(args, 'early_stop_patience', 0)),
            early_stop_metric=str(getattr(args, 'early_stop_metric', 'ap_exact_acc')),
            best_metric_name=str(getattr(args, 'best_metric_name', 'ap_exact_acc_plus_half_len')),
            on_epoch_end=_epoch_callback,
        )
        history = trainer.fit()

        # ---- report -------------------------------------------------------
        run_full_report(
            model, ema, trainer, test_loader, quantizer, test_word_types,
            history, paths.vis_dir,
            sampling_method, ddim_steps, ddim_eta,
            final_eval_use_best_metric=bool(getattr(args, 'final_eval_use_best_metric', True)),
        )
        print(f'\nAll console outputs have been saved to: {os.path.join(run_dir, "log.txt")}')
    finally:
        sys.stdout = orig_out
        sys.stderr = orig_err
        dual.close()


if __name__ == '__main__':
    main()
