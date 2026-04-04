from __future__ import annotations
import os
import random
import torch
from torch.utils.data import DataLoader

from pri_tokenizer import PRIQuantizer, QuantizerConfig
from pri_dataset import PRISample, PRIDiffuSeqDataset, PRICollator
from train_miss_pri import evaluate_reconstruction_accuracy, save_epoch_visuals, save_final_confusion_matrix, set_seed
from model import PRIDiffuSeq, PRIDiffuSeqConfig


def main():
    set_seed(42)
    out_root = os.path.join('CheckPoint', 'SmokeValidate')
    vis_dir = os.path.join(out_root, 'visuals')
    os.makedirs(vis_dir, exist_ok=True)

    clean_root = os.path.join('dataset', 'Ground_Truth')
    obs_root = os.path.join('dataset', 'Miss')
    files = [f for f in os.listdir(clean_root) if f.endswith('.pt')]
    random.shuffle(files)
    files = files[:6]

    samples = []
    for f in files:
        clean = torch.load(os.path.join(clean_root, f), weights_only=False)['seq'].tolist()
        obs = torch.load(os.path.join(obs_root, f), weights_only=False)['seq'].tolist()
        samples.append(PRISample(observed_pri=obs, clean_pri=clean))

    train_samples = samples[:4]
    test_samples = samples[4:]

    quantizer = PRIQuantizer(
        QuantizerConfig(
            mode='uniform',
            min_value=420.0,
            max_value=710.0,
            num_bins=291,
            add_special_tokens=True,
            key_start=1,
        )
    )

    train_loader = DataLoader(PRIDiffuSeqDataset(train_samples, quantizer, seq_len=220), batch_size=1, shuffle=True, collate_fn=PRICollator())
    test_loader = DataLoader(PRIDiffuSeqDataset(test_samples, quantizer, seq_len=220), batch_size=1, shuffle=False, collate_fn=PRICollator())

    cfg = PRIDiffuSeqConfig(
        vocab_size=quantizer.vocab_size,
        seq_len=220,
        hidden_dim=8,
        model_dim=8,
        time_dim=8,
        num_layers=1,
        num_heads=1,
        diffusion_steps=4,
        beta_schedule='cosine',
        ce_weight=0.3,
        device='cpu',
    )
    model = PRIDiffuSeq(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

    vis_batch = next(iter(test_loader))
    for epoch in range(2):
        train_stats = model.fit_epoch(train_loader, optimizer)
        test_stats = model.evaluate(test_loader)
        pred_acc, greedy_acc = evaluate_reconstruction_accuracy(
            model,
            test_loader,
            quantizer,
            sampling_method='ddim',
            ddim_steps=2,
            ddim_eta=0.0,
        )
        print(
            f'[smoke] epoch={epoch} train_loss={train_stats["loss"]:.4f} '
            f'test_loss={test_stats["loss"]:.4f} pred_acc={pred_acc:.4%} greedy_acc={greedy_acc:.4%}',
            flush=True,
        )
        save_epoch_visuals(model, vis_batch, quantizer, epoch, vis_dir, 'ddim', 2, 0.0)

    save_final_confusion_matrix(model, test_loader, quantizer, vis_dir, 'ddim', 2, 0.0)
    print(f'[smoke] outputs saved to: {out_root}', flush=True)


if __name__ == '__main__':
    main()
