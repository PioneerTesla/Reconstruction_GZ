from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rounding import nearest_token_ids, round_hidden_states


@dataclass
class PRIDiffuSeqConfig:
    vocab_size: int
    seq_len: int = 256
    hidden_dim: int = 128
    model_dim: int = 256
    time_dim: int = 128
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    diffusion_steps: int = 40
    beta_schedule: str = 'cosine'
    ce_weight: float = 0.5
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    # --- variable-length & ablation options ---
    p_full_target: float = 0.0      # 0.0 = original behaviour; >0 extends target mask to full SEP-onwards region
    loss_mode: str = 'mse+ce'       # 'mse+ce' (default) | 'mse_only' | 'mse+ce_no_tw' (CE without time weighting)
    # --- stabilization options (values injected from config.json via train_pri.py) ---
    label_smoothing: float = 0.0
    end_loss_weight: float = 0.0
    p_full_target_curriculum: bool = False
    curriculum_warmup_epochs: int = 200



def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half)
    args = timesteps[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def get_named_beta_schedule(schedule_name: str, num_diffusion_timesteps: int) -> np.ndarray:
    if schedule_name == 'linear':
        scale = 1000 / num_diffusion_timesteps
        return np.linspace(scale * 0.0001, scale * 0.02, num_diffusion_timesteps, dtype=np.float64)
    if schedule_name == 'cosine':
        def alpha_bar(t: float) -> float:
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), 0.999))
        return np.array(betas, dtype=np.float64)
    raise ValueError(f'unknown schedule: {schedule_name}')


class DenoiseTransformer(nn.Module):
    def __init__(self, cfg: PRIDiffuSeqConfig):
        super().__init__()
        self.cfg = cfg
        self.word_embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
        self.input_up = nn.Linear(cfg.hidden_dim, cfg.model_dim) if cfg.hidden_dim != cfg.model_dim else nn.Identity()
        self.output_down = nn.Linear(cfg.model_dim, cfg.hidden_dim) if cfg.hidden_dim != cfg.model_dim else nn.Identity()
        self.pos_embedding = nn.Embedding(cfg.seq_len, cfg.model_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.time_dim, cfg.model_dim),
            nn.SiLU(),
            nn.Linear(cfg.model_dim, cfg.model_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.model_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.model_dim * 4,
            dropout=cfg.dropout,
            batch_first=True,
            activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)
        self.lm_head = nn.Linear(cfg.hidden_dim, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.word_embedding.weight

    def get_embeds(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.word_embedding(input_ids)

    def get_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        pos_ids = torch.arange(seqlen, device=x.device).unsqueeze(0)
        time_emb = self.time_mlp(timestep_embedding(timesteps, self.cfg.time_dim)).unsqueeze(1)
        h = self.input_up(x)
        h = h + self.pos_embedding(pos_ids) + time_emb
        key_padding_mask = None
        if padding_mask is not None:
            key_padding_mask = padding_mask.bool()
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        h = self.output_down(h)
        return h


class GaussianDiffusion1D:
    def __init__(self, cfg: PRIDiffuSeqConfig):
        self.cfg = cfg
        betas = get_named_beta_schedule(cfg.beta_schedule, cfg.diffusion_steps)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        self.num_timesteps = int(cfg.diffusion_steps)
        self.betas = torch.tensor(betas, dtype=torch.float32)
        self.alphas_cumprod = torch.tensor(alphas_cumprod, dtype=torch.float32)
        self.alphas_cumprod_prev = torch.tensor(alphas_cumprod_prev, dtype=torch.float32)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_variance = torch.tensor(posterior_variance, dtype=torch.float32)
        self.posterior_log_variance_clipped = torch.log(torch.clamp(self.posterior_variance, min=1e-20))
        self.posterior_mean_coef1 = torch.tensor(
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod), dtype=torch.float32
        )
        self.posterior_mean_coef2 = torch.tensor(
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod), dtype=torch.float32
        )

    def to(self, device: torch.device) -> 'GaussianDiffusion1D':
        for name, value in list(self.__dict__.items()):
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(device))
        return self

    def _extract(self, arr: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        out = arr.gather(0, t).float()
        while len(out.shape) < len(x_shape):
            out = out.unsqueeze(-1)
        return out.expand(x_shape)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        x_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        if mask is not None:
            x_t = torch.where(mask == 0, x_start, x_t)
        return x_t

    def predict_xstart_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        return self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps

    def q_posterior_mean_variance(self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        log_var = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return mean, log_var

    @torch.no_grad()
    def p_sample(self, model: nn.Module, x: torch.Tensor, t: torch.Tensor, mask: Optional[torch.Tensor] = None, x_start: Optional[torch.Tensor] = None, round_fn=None, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x_0 parameterization: model directly predicts x_0
        pred_xstart = model(x, t, padding_mask=padding_mask)
        if round_fn is not None:
            pred_xstart = round_fn(pred_xstart)
        mean, log_var = self.q_posterior_mean_variance(pred_xstart, x, t)
        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().view(-1, 1, 1)
        sample = mean + nonzero_mask * torch.exp(0.5 * log_var) * noise
        if mask is not None and x_start is not None:
            sample = torch.where(mask == 0, x_start, sample)
        return sample

    @torch.no_grad()
    def p_sample_loop(self, model: nn.Module, shape: Tuple[int, int, int], noise: torch.Tensor, mask: Optional[torch.Tensor] = None, x_start: Optional[torch.Tensor] = None, round_fn=None, padding_mask: Optional[torch.Tensor] = None, collect_steps=None):
        x = noise
        trace = {}
        collect_steps = set([] if collect_steps is None else collect_steps)
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((shape[0],), i, device=x.device, dtype=torch.long)
            x = self.p_sample(model, x, t, mask=mask, x_start=x_start, round_fn=round_fn, padding_mask=padding_mask)
            if i in collect_steps:
                trace[i] = x.detach().cpu()
        return x, trace

    def _make_ddim_timesteps(self, ddim_steps: int) -> np.ndarray:
        ddim_steps = int(ddim_steps)
        if ddim_steps <= 0:
            raise ValueError('ddim_steps must be > 0')
        if ddim_steps >= self.num_timesteps:
            return np.arange(self.num_timesteps, dtype=np.int64)
        steps = np.linspace(0, self.num_timesteps - 1, ddim_steps, dtype=np.int64)
        return np.unique(steps)

    @torch.no_grad()
    def ddim_sample_loop(
        self,
        model: nn.Module,
        shape: Tuple[int, int, int],
        noise: torch.Tensor,
        ddim_steps: int,
        eta: float = 0.0,
        mask: Optional[torch.Tensor] = None,
        x_start: Optional[torch.Tensor] = None,
        round_fn=None,
        padding_mask: Optional[torch.Tensor] = None,
        collect_steps=None,
    ):
        x = noise
        trace = {}
        collect_steps = set([] if collect_steps is None else collect_steps)
        step_seq = self._make_ddim_timesteps(ddim_steps)
        step_seq = list(step_seq.tolist())

        for idx in reversed(range(len(step_seq))):
            t_value = step_seq[idx]
            prev_t_value = step_seq[idx - 1] if idx > 0 else -1
            t = torch.full((shape[0],), t_value, device=x.device, dtype=torch.long)

            # x_0 parameterization: model directly predicts x_0
            pred_xstart = model(x, t, padding_mask=padding_mask)
            if round_fn is not None:
                pred_xstart = round_fn(pred_xstart)

            if prev_t_value < 0:
                x = pred_xstart
            else:
                alpha_bar_t = self.alphas_cumprod[t_value]
                alpha_bar_prev = self.alphas_cumprod[prev_t_value]

                # Derive eps from pred_xstart for DDIM update
                pred_eps = (x - torch.sqrt(alpha_bar_t) * pred_xstart) / torch.sqrt(1.0 - alpha_bar_t).clamp(min=1e-8)

                sigma = eta * torch.sqrt((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)) * torch.sqrt(1.0 - alpha_bar_t / alpha_bar_prev)
                sigma = torch.clamp(sigma, min=0.0)
                c = torch.sqrt(torch.clamp(1.0 - alpha_bar_prev - sigma ** 2, min=0.0))
                noise_term = sigma * torch.randn_like(x)
                x = torch.sqrt(alpha_bar_prev) * pred_xstart + c * pred_eps + noise_term

            if mask is not None and x_start is not None:
                x = torch.where(mask == 0, x_start, x)
            if t_value in collect_steps:
                trace[t_value] = x.detach().cpu()
        return x, trace


class EMAModel:
    """Exponential Moving Average of model parameters for stable inference."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {name: p.clone().detach() for name, p in model.named_parameters()}
        self.backup = {}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, p in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    def apply_shadow(self, model: nn.Module):
        """Replace model params with EMA params (for evaluation)."""
        self.backup = {name: p.data.clone() for name, p in model.named_parameters()}
        for name, p in model.named_parameters():
            if name in self.shadow:
                p.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        """Restore original model params after evaluation."""
        for name, p in model.named_parameters():
            if name in self.backup:
                p.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state_dict):
        self.shadow = {k: v.clone() for k, v in state_dict.items()}


class PRIDiffuSeq(nn.Module):
    """
    Task-specific wrapper for PRI reconstruction.

    Training view:
      source  = observed / corrupted PRI sequence
      target  = clean PRI sequence
      diffusion only denoises target segment, while source / separator / pad stay fixed.
    """

    def __init__(self, cfg: PRIDiffuSeqConfig):
        super().__init__()
        self.cfg = cfg
        self.model = DenoiseTransformer(cfg)
        self.diffusion = GaussianDiffusion1D(cfg)
        self.device_name = cfg.device
        self.to(cfg.device)
        self.diffusion.to(torch.device(cfg.device))

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.model(x_t, t, padding_mask=padding_mask)

    def compute_loss(self, input_ids: torch.Tensor, input_mask: torch.Tensor, category_ids=None, current_epoch: int = 0, total_epochs: int = 1) -> Dict[str, torch.Tensor]:
        del category_ids
        input_ids = input_ids.to(self.device)
        input_mask = input_mask.to(self.device)
        target_mask = input_mask.float()

        # ── Variable-length alignment ──────────────────────────────────
        # With p_full_target > 0, extend target_mask to cover everything
        # after SEP (including PAD region), so the model also learns to
        # predict END + PAD from pure noise — matching test_mode.
        SEP_ID, PAD_ID, END_ID = 4, 3, 1
        p_ft = self.cfg.p_full_target

        # Curriculum: gradually ramp p_full_target from 0 to its max value
        if self.cfg.p_full_target_curriculum and self.training:
            warmup = max(1, self.cfg.curriculum_warmup_epochs)
            p_ft = p_ft * min(1.0, current_epoch / warmup)

        if self.training and p_ft > 0.0:
            full_mask = torch.zeros_like(target_mask)
            for i in range(input_ids.size(0)):
                sep_pos = (input_ids[i] == SEP_ID).nonzero(as_tuple=True)[0]
                if len(sep_pos) > 0:
                    full_mask[i, sep_pos[0].item() + 1:] = 1.0
            use_full = (torch.rand(input_ids.size(0), device=self.device) < p_ft).float().unsqueeze(1)
            target_mask = use_full * full_mask + (1.0 - use_full) * target_mask

        padding_mask = (input_ids == PAD_ID).to(self.device)

        x_start = self.model.get_embeds(input_ids)
        noise = torch.randn_like(x_start)
        t = torch.randint(0, self.cfg.diffusion_steps, (input_ids.size(0),), device=input_ids.device)
        mask3 = target_mask.unsqueeze(-1).expand_as(x_start)
        x_t = self.diffusion.q_sample(x_start, t, noise=noise, mask=mask3)

        # x_0 parameterization: model directly predicts x_0
        pred_x0 = self.forward(x_t, t, padding_mask=padding_mask)

        # MSE loss on x_0 prediction (target positions only)
        mse = ((pred_x0 - x_start) ** 2).mean(dim=-1)
        mse = (mse * target_mask).sum(dim=-1) / target_mask.sum(dim=-1).clamp_min(1.0)

        # ── Loss mode selection (ablation interface) ──────────────────
        loss_mode = self.cfg.loss_mode
        label_smoothing = self.cfg.label_smoothing

        if loss_mode == 'mse_only':
            # Ablation: pure MSE, no CE
            ce = torch.zeros_like(mse)
            loss = mse.mean()
        else:
            # CE loss from predicted x_0 (with optional label smoothing)
            logits = self.model.get_logits(pred_x0)
            ce = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                input_ids.view(-1),
                reduction='none',
                label_smoothing=label_smoothing,
            ).view_as(input_ids)
            ce = (ce * target_mask).sum(dim=-1) / target_mask.sum(dim=-1).clamp_min(1.0)

            if loss_mode == 'mse+ce':
                # Default: time-dependent CE weighting
                t_weight = 1.0 - t.float() / self.cfg.diffusion_steps
                ce = ce * t_weight
            # else loss_mode == 'mse+ce_no_tw': CE without time weighting (keep ce as-is)

            loss = mse.mean() + self.cfg.ce_weight * ce.mean()

        # ── Extra END-token prediction loss ───────────────────────────
        end_loss_w = self.cfg.end_loss_weight
        if end_loss_w > 0.0 and loss_mode != 'mse_only':
            # Locate positions where GT is END token within target region
            end_positions = ((input_ids == END_ID) & (target_mask > 0.5)).float()
            if end_positions.sum() > 0:
                end_logits = self.model.get_logits(pred_x0)
                end_ce = F.cross_entropy(
                    end_logits.view(-1, end_logits.size(-1)),
                    input_ids.view(-1),
                    reduction='none',
                ).view_as(input_ids)
                end_loss = (end_ce * end_positions).sum() / end_positions.sum().clamp_min(1.0)
                loss = loss + end_loss_w * end_loss

        return {'loss': loss, 'mse': mse.mean(), 'ce': ce.mean()}

    def fit_epoch(self, dataloader, optimizer: torch.optim.Optimizer, grad_clip: float = 1.0, schedual=None, category_ids=None, current_epoch: int = 0, total_epochs: int = 1) -> Dict[str, float]:
        self.train()
        meters = {'loss': 0.0, 'mse': 0.0, 'ce': 0.0}
        count = 0
        for batch in dataloader:
            stats = self.compute_loss(batch['input_ids'], batch['input_mask'], category_ids, current_epoch=current_epoch, total_epochs=total_epochs)
            optimizer.zero_grad(set_to_none=True)
            stats['loss'].backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
            optimizer.step()

            for k in meters:
                meters[k] += float(stats[k].item())
            count += 1

        if schedual is not None:
            try:
                avg_loss = meters['loss'] / max(count, 1)
                schedual.step(avg_loss)
            except TypeError:
                schedual.step()
        return {k: v / max(count, 1) for k, v in meters.items()}

    @torch.no_grad()
    def evaluate(self, dataloader, category_ids=None) -> Dict[str, float]:
        self.eval()
        meters = {'loss': 0.0, 'mse': 0.0, 'ce': 0.0}
        count = 0
        for batch in dataloader:
            stats = self.compute_loss(batch['input_ids'], batch['input_mask'], category_ids)
            for k in meters:
                meters[k] += float(stats[k].item())
            count += 1
        return {k: v / max(count, 1) for k, v in meters.items()}

    @torch.no_grad()
    def reconstruct(
        self,
        input_ids: torch.Tensor,
        input_mask: torch.Tensor,
        use_rounding: bool = True,
        sampling_method: str = 'ddpm',
        ddim_steps: Optional[int] = None,
        ddim_eta: float = 0.0,
        return_trace: bool = False,
        trace_steps=None,
        test_mode: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Reconstruct target PRI from noisy diffusion.

        When *test_mode=True* the target/pad boundary is **unknown**: the mask
        is derived solely from the SEP token position, and ground-truth target
        embeddings are replaced with PAD embeddings to prevent information
        leakage.  This mirrors real deployment where only observed PRI (source)
        is available.
        """
        self.eval()
        input_ids = input_ids.to(self.device)

        SEP_ID, PAD_ID = 4, 3

        if test_mode:
            B, L = input_ids.shape
            target_mask = torch.zeros(B, L, device=self.device, dtype=torch.float)
            for i in range(B):
                sep_pos = (input_ids[i] == SEP_ID).nonzero(as_tuple=True)[0]
                if len(sep_pos) > 0:
                    target_mask[i, sep_pos[0].item() + 1:] = 1.0
            # Replace target tokens with PAD so embeddings carry no GT info
            input_ids_for_embed = input_ids.clone()
            input_ids_for_embed[target_mask.bool()] = PAD_ID
            x_start = self.model.get_embeds(input_ids_for_embed)
            # No attention masking – all positions attend to each other
            padding_mask = torch.zeros(B, L, device=self.device)
        else:
            input_mask = input_mask.to(self.device)
            target_mask = input_mask.float()
            padding_mask = (input_ids == PAD_ID).float().to(self.device)
            x_start = self.model.get_embeds(input_ids)

        noise = torch.randn_like(x_start)
        mask3 = target_mask.unsqueeze(-1).expand_as(x_start)
        x_noised = torch.where(mask3 == 0, x_start, noise)
        round_fn = (lambda x: round_hidden_states(self.model.word_embedding, x)) if use_rounding else None

        sampling_method = sampling_method.lower().strip()
        if sampling_method == 'ddim':
            steps = ddim_steps if ddim_steps is not None else self.cfg.diffusion_steps
            sample, trace = self.diffusion.ddim_sample_loop(
                self,
                x_start.shape,
                noise=x_noised,
                ddim_steps=steps,
                eta=ddim_eta,
                mask=mask3,
                x_start=x_start,
                round_fn=round_fn,
                padding_mask=padding_mask,
                collect_steps=trace_steps if return_trace else None,
            )
        else:
            sample, trace = self.diffusion.p_sample_loop(
                self,
                x_start.shape,
                noise=x_noised,
                mask=mask3,
                x_start=x_start,
                round_fn=round_fn,
                padding_mask=padding_mask,
                collect_steps=trace_steps if return_trace else None,
            )

        pred_ids = nearest_token_ids(self.model.word_embedding, sample)
        logits = self.model.get_logits(sample)
        greedy_ids = logits.argmax(dim=-1)
        out = {'sample': sample, 'pred_ids': pred_ids, 'greedy_ids': greedy_ids}
        if return_trace:
            out['trace'] = trace
        return out

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
