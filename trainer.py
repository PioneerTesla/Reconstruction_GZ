"""Training loop for PRIDiffuSeq.

`Trainer` owns:
  - the epoch loop and history dict
  - dual best-checkpoint selection (by test_loss and by composite metric
    `exact_acc + 0.5 * len_match_rate`, with EMA mirrors)
  - LR schedule: constant `lr_warmup_epochs` followed by either
    ReduceLROnPlateau (on train loss) or CosineAnnealingLR
  - optional early stopping on the composite metric

`train_pri.py` only calls `Trainer(...).fit()`.
"""
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from evaluation import evaluate_testmode_metrics


@dataclass
class TrainerPaths:
    best_loss: str            # best by test_loss
    best_metric: str          # best by composite metric
    latest: str
    vis_dir: str

    @classmethod
    def from_root(cls, root: str) -> 'TrainerPaths':
        os.makedirs(root, exist_ok=True)
        return cls(
            best_loss=os.path.join(root, 'best_model.pth'),
            best_metric=os.path.join(root, 'best_model_by_metric.pth'),
            latest=os.path.join(root, 'latest_model.pth'),
            vis_dir=os.path.join(root, 'visuals'),
        )


class Trainer:
    """Encapsulates the DiffuSeq-PRI training loop.

    Composite metric used for `best_model_by_metric.pth`:
        metric = exact_acc + 0.5 * len_match_rate
    exact_acc dominates reconstruction fidelity; len_match_rate rewards
    correct END-token prediction (critical for Miss / Mix scenes).
    """

    def __init__(
        self,
        model,
        ema,
        optimizer: torch.optim.Optimizer,
        quantizer,
        train_loader,
        test_loader,
        paths: TrainerPaths,
        *,
        epochs: int,
        sampling_method: str,
        ddim_steps: int,
        ddim_eta: float,
        lr_warmup_epochs: int = 200,
        lr_plateau_patience: int = 10,
        lr_plateau_factor: float = 0.5,
        lr_plateau_min: float = 1e-6,
        use_cosine_lr: bool = False,
        cosine_lr_min: float = 1e-6,
        early_stop_patience: int = 0,
        early_stop_metric: str = 'ap_exact_acc',
        best_metric_name: str = 'ap_exact_acc_plus_half_len',
        on_epoch_end=None,   # optional callback(model, epoch) for mid-training visuals
    ):
        self.model = model
        self.ema = ema
        self.optimizer = optimizer
        self.quantizer = quantizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.paths = paths
        self.epochs = int(epochs)
        self.sampling_method = sampling_method
        self.ddim_steps = ddim_steps
        self.ddim_eta = ddim_eta
        self.lr_warmup_epochs = int(lr_warmup_epochs)
        self.use_cosine_lr = bool(use_cosine_lr)
        self.early_stop_patience = int(early_stop_patience)
        self.early_stop_metric = early_stop_metric
        self.best_metric_name = best_metric_name
        self.on_epoch_end = on_epoch_end

        if self.use_cosine_lr:
            T = max(1, self.epochs - self.lr_warmup_epochs)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=T, eta_min=cosine_lr_min)
            print(f'lr_schedule=CosineAnnealingLR (active after epoch {self.lr_warmup_epochs})  '
                  f'T_max={T}  eta_min={cosine_lr_min}')
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=lr_plateau_factor,
                patience=lr_plateau_patience, min_lr=lr_plateau_min)
            print(f'lr_schedule=ReduceLROnPlateau (active after epoch {self.lr_warmup_epochs})  '
                  f'patience={lr_plateau_patience}  factor={lr_plateau_factor}  min_lr={lr_plateau_min}')
        if self.early_stop_patience > 0:
            print(f'early_stop_metric={self.early_stop_metric}  patience={self.early_stop_patience}')
        print(f'best_metric_name={self.best_metric_name}')

        self.history: Dict[str, List[float]] = {
            'train_loss': [], 'test_loss': [],
            'train_mse': [], 'train_ce': [],
            'ap_exact_acc': [], 'ap_tol1_acc': [], 'ap_mae': [],
            'len_match': [],
        }
        self.best_test_loss = float('inf')
        self.best_metric_value = -float('inf')
        self.best_metric_epoch = -1

    # ---- internal helpers --------------------------------------------------

    def _step_scheduler(self, epoch: int, train_loss: float) -> None:
        if epoch < self.lr_warmup_epochs:
            return
        if self.use_cosine_lr:
            self.scheduler.step()
        else:
            # ReduceLROnPlateau — pass the monitored scalar (train loss).
            self.scheduler.step(train_loss)

    def _save_ckpt_with_ema(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)
        torch.save(self.ema.state_dict(), path.replace('.pth', '_ema.pth'))

    def _compose_metric(self, tm: dict) -> float:
        return tm['exact_acc'] + 0.5 * tm['len_match_rate']

    # ---- public API --------------------------------------------------------

    def evaluate(self) -> dict:
        return evaluate_testmode_metrics(
            self.model, self.test_loader, self.quantizer,
            sampling_method=self.sampling_method,
            ddim_steps=self.ddim_steps, ddim_eta=self.ddim_eta,
            tolerance=1,
        )

    def fit(self) -> Dict[str, List[float]]:
        epochs_since_improved = 0
        for epoch in range(self.epochs):
            # Train one epoch. We step the LR schedule ourselves after
            # fit_epoch so CosineAnnealingLR (no arg) and ReduceLROnPlateau
            # (needs loss) are handled uniformly.
            train_stats = self.model.fit_epoch(
                self.train_loader, self.optimizer, schedual=None,
                current_epoch=epoch, total_epochs=self.epochs,
            )
            self._step_scheduler(epoch, train_stats['loss'])

            self.ema.update(self.model)
            test_stats = self.model.evaluate(self.test_loader)
            tm = self.evaluate()
            metric = self._compose_metric(tm)

            # Record history
            self.history['train_loss'].append(train_stats['loss'])
            self.history['test_loss'].append(test_stats['loss'])
            self.history['train_mse'].append(train_stats['mse'])
            self.history['train_ce'].append(train_stats['ce'])
            self.history['ap_exact_acc'].append(tm['exact_acc'])
            self.history['ap_tol1_acc'].append(tm['tol_acc'])
            self.history['ap_mae'].append(tm['mae'])
            self.history['len_match'].append(tm['len_match_rate'])

            lr = self.optimizer.param_groups[0]['lr']
            print(
                f'epoch={epoch} train_loss={train_stats["loss"]:.6f} '
                f'test_loss={test_stats["loss"]:.6f} '
                f'train_mse={train_stats["mse"]:.6f} train_ce={train_stats["ce"]:.6f} '
                f'ap_exact={tm["exact_acc"]:.4%} ap_tol1={tm["tol_acc"]:.4%} mae={tm["mae"]:.2f} '
                f'len_match={tm["len_match_rate"]:.4%} metric={metric:.4f} lr={lr:.6e}'
            )

            torch.save(self.model.state_dict(), self.paths.latest)

            if self.on_epoch_end is not None:
                self.on_epoch_end(self.model, epoch)

            if test_stats['loss'] < self.best_test_loss:
                self.best_test_loss = test_stats['loss']
                self._save_ckpt_with_ema(self.paths.best_loss)
                print(f'New best (by test_loss) model saved at epoch {epoch} '
                      f'with test_loss={self.best_test_loss:.6f}')

            if metric > self.best_metric_value:
                self.best_metric_value = metric
                self.best_metric_epoch = epoch
                epochs_since_improved = 0
                self._save_ckpt_with_ema(self.paths.best_metric)
                print(f'New best (by {self.best_metric_name}) model saved at epoch {epoch} '
                      f'with metric={self.best_metric_value:.4f} '
                      f'(exact={tm["exact_acc"]:.4%}, len_match={tm["len_match_rate"]:.4%})')
            else:
                epochs_since_improved += 1

            if self.early_stop_patience > 0 and epochs_since_improved >= self.early_stop_patience:
                print(f'Early stopping at epoch {epoch}: no improvement in '
                      f'{self.early_stop_metric}-based metric for '
                      f'{self.early_stop_patience} epochs (best epoch={self.best_metric_epoch}, '
                      f'best metric={self.best_metric_value:.4f}).')
                break

        return self.history

    def load_final_checkpoint(self, prefer_metric: bool = True) -> Optional[str]:
        """Load the preferred final checkpoint into `self.model`. Returns the
        path loaded (without the EMA suffix) or None if no checkpoint exists."""
        if prefer_metric and os.path.exists(self.paths.best_metric):
            path = self.paths.best_metric
            label = (f'best_by_metric ({self.best_metric_name}, '
                     f'epoch={self.best_metric_epoch}, metric={self.best_metric_value:.4f})')
        elif os.path.exists(self.paths.best_loss):
            path = self.paths.best_loss
            label = 'best_by_test_loss'
        else:
            print('[trainer] no best checkpoint found; using in-memory weights.')
            return None
        print(f'\n=== Final checkpoint for visuals/report: {label} ===')
        self.model.load_state_dict(torch.load(path, map_location=self.model.device))
        return path
