import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


__all__ = [
    "set_torch",
    "sample_t_uniform",
    "make_warmup_cosine",
    "NoiseScheduler",
    "iter_laplace_bases",
    "log_pole_health",
    "compute_per_dim_stats",
    "normalize_and_check",
    "flatten_targets",
    "EMA",
    "ewma_std",
    "two_stage_norm",
    "normalize_cond_per_batch",
    "compute_latent_stats",
    "invert_two_stage_norm",
    "decode_latents_with_vae",
]


# ============================ LLapDiT utils ============================
def set_torch() -> torch.device:
    """
    Configure PyTorch for training/inference.

    - Enables TF32 for CUDA matmul (Ampere+ gives a nice speedup).
    - Sets higher float32 matmul precision on CUDA.
    - Returns the selected device.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # TF32 hints
    torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    return device


def sample_t_uniform(scheduler: "NoiseScheduler", n: int, device: torch.device) -> torch.Tensor:
    """
    Uniformly sample discrete timesteps in [0, T).

    Args:
        scheduler: The NoiseScheduler (provides .timesteps).
        n:         Number of samples.
        device:    Target device for the indices tensor.

    Returns:
        Tensor[int64]: shape [n]
    """
    return torch.randint(0, int(scheduler.timesteps), (n,), device=device)


def make_warmup_cosine(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_frac: float = 0.05,
    base_lr: float = 5e-4,
    min_lr: float = 1e-6,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Linear warmup -> cosine decay (multiplier schedule).

    The returned lambda produces a factor that multiplies the optimizer's base LR(s),
    bottoming out at min_lr / base_lr.

    Args:
        optimizer:    Optimizer.
        total_steps:  Total training steps (must be > 0).
        warmup_frac:  Fraction of steps to warm up (0..1).
        base_lr:      Reference base LR for computing the min multiplier.
        min_lr:       Minimum absolute LR after decay.

    Returns:
        torch.optim.lr_scheduler.LambdaLR
    """
    total_steps = max(1, int(total_steps))
    warmup_steps = max(1, int(total_steps * max(0.0, min(1.0, warmup_frac))))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1.0, total_steps - warmup_steps)
        # cosine in [0, 1] -> [1, 0]
        cos_term = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_mult = max(min_lr / max(base_lr, 1e-12), 0.0)
        return max(min_mult, cos_term)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _cosine_alpha_bar(t: torch.Tensor, s: float = 0.008) -> torch.Tensor:
    """
    Continuous-time ᾱ(t) from Nichol & Dhariwal (t ∈ [0, 1]).
    """
    return torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2


class NoiseScheduler(nn.Module):
    """
    Diffusion scheduler utilities with precomputed buffers and a DDIM sampler.

    Supports:
      - schedules: 'linear' or 'cosine'
      - parameterizations: 'eps' / 'v' / 'x0'
    """

    def __init__(
        self,
        timesteps: int = 1000,
        schedule: str = "cosine",
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ) -> None:
        super().__init__()
        self.timesteps = int(timesteps)
        if schedule not in {"linear", "cosine"}:
            raise ValueError(f"Unknown schedule: {schedule}")
        self.schedule = schedule

        if schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, self.timesteps, dtype=torch.float32)
            betas = betas.clamp(1e-8, 0.999)
        else:
            ts = torch.linspace(0, 1, self.timesteps + 1, dtype=torch.float32)
            abar = _cosine_alpha_bar(ts)
            abar = abar / abar[0]  # normalize so ᾱ(0)=1
            alphas = torch.ones(self.timesteps, dtype=torch.float32)
            alphas[1:] = abar[1:self.timesteps] / abar[0 : self.timesteps - 1]
            betas = (1.0 - alphas).clone()
            betas[1:] = betas[1:].clamp(min=1e-8, max=0.999)
            betas[0] = 0.0

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", 1.0 - betas)
        alpha_bars = torch.cumprod(1.0 - betas, dim=0)
        # Small clamp to keep sqrt arguments safe from tiny negative drift
        self.register_buffer("alpha_bars", alpha_bars.clamp(0.0, 1.0))
        self.register_buffer("sqrt_alphas", torch.sqrt(1.0 - betas))
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(self.alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - self.alpha_bars))

    @torch.no_grad()
    def timesteps_desc(self) -> torch.Tensor:
        """Return a [T] tensor with timesteps T-1..0 on the correct device."""
        return torch.arange(self.timesteps - 1, -1, -1, dtype=torch.long, device=self.alpha_bars.device)

    def _gather(self, buf: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Gather values for integer timesteps t along dim 0."""
        t = t.clamp(min=0, max=self.timesteps - 1).to(device=buf.device, dtype=torch.long)
        return buf.gather(0, t)

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion x₀ -> x_t.

        Args:
            x0:    Clean input, shape [B, ...]
            t:     Timesteps, shape [B]
            noise: Optional noise; if None, sampled ~ N(0, I).

        Returns:
            (x_t, noise), both shaped like x0.
        """
        if noise is None:
            noise = torch.randn_like(x0)
        view = (-1,) + (1,) * (x0.dim() - 1)
        sqrt_ab = self._gather(self.sqrt_alpha_bars, t).view(*view)
        sqrt_1_ab = self._gather(self.sqrt_one_minus_alpha_bars, t).view(*view)
        x_t = sqrt_ab * x0 + sqrt_1_ab * noise
        return x_t, noise

    # ----- conversions between parameterizations -----
    def pred_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        alpha = self._gather(self.sqrt_alpha_bars, t).view(-1, *([1] * (x_t.dim() - 1)))
        sigma = self._gather(self.sqrt_one_minus_alpha_bars, t).view(-1, *([1] * (x_t.dim() - 1)))
        return (x_t - sigma * eps) / (alpha + 1e-12)

    def pred_eps_from_x0(self, x_t: torch.Tensor, t: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        alpha = self._gather(self.sqrt_alpha_bars, t).view(-1, *([1] * (x_t.dim() - 1)))
        sigma = self._gather(self.sqrt_one_minus_alpha_bars, t).view(-1, *([1] * (x_t.dim() - 1)))
        return (x_t - alpha * x0) / (sigma + 1e-12)

    def pred_x0_from_v(self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        alpha = self._gather(self.sqrt_alpha_bars, t).view(-1, *([1] * (x_t.dim() - 1)))
        sigma = self._gather(self.sqrt_one_minus_alpha_bars, t).view(-1, *([1] * (x_t.dim() - 1)))
        return alpha * x_t - sigma * v

    def pred_eps_from_v(self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        alpha = self._gather(self.sqrt_alpha_bars, t).view(-1, *([1] * (x_t.dim() - 1)))
        sigma = self._gather(self.sqrt_one_minus_alpha_bars, t).view(-1, *([1] * (x_t.dim() - 1)))
        return sigma * x_t + alpha * v

    def v_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        alpha = self._gather(self.sqrt_alpha_bars, t).view(-1, *([1] * (x_t.dim() - 1)))
        sigma = self._gather(self.sqrt_one_minus_alpha_bars, t).view(-1, *([1] * (x_t.dim() - 1)))
        return (eps - sigma * x_t) / (alpha + 1e-12)

    def to_x0(self, x_t: torch.Tensor, t: torch.Tensor, pred: torch.Tensor, param_type: str) -> torch.Tensor:
        if param_type == "eps":
            return self.pred_x0_from_eps(x_t, t, pred)
        elif param_type == "v":
            return self.pred_x0_from_v(x_t, t, pred)
        elif param_type == "x0":
            return pred
        else:
            raise ValueError("param_type must be 'eps', 'v', or 'x0'")

    def to_eps(self, x_t: torch.Tensor, t: torch.Tensor, pred: torch.Tensor, param_type: str) -> torch.Tensor:
        if param_type == "eps":
            return pred
        elif param_type == "v":
            return self.pred_eps_from_v(x_t, t, pred)
        elif param_type == "x0":
            return self.pred_eps_from_x0(x_t, t, pred)
        else:
            raise ValueError("param_type must be 'eps', 'v', or 'x0'")

    # ----- DDIM sampler step -----
    @torch.no_grad()
    def ddim_sigma(self, t: torch.Tensor, t_prev: torch.Tensor, eta: float) -> torch.Tensor:
        """
        Compute per-sample σ_t for DDIM with temperature `eta`.
        """
        ab_t = self._gather(self.alpha_bars, t)
        ab_prev = self._gather(self.alpha_bars, t_prev)
        sigma = eta * torch.sqrt((1.0 - ab_prev) / (1.0 - ab_t)) * torch.sqrt(1.0 - ab_t / (ab_prev + 1e-12))
        return sigma.view(-1)

    @torch.no_grad()
    def ddim_step_from(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        t_prev: torch.Tensor,
        pred: torch.Tensor,
        param_type: str,
        eta: float = 0.0,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        One deterministic (eta=0) or stochastic (eta>0) DDIM step x_t -> x_{t-1}.

        Args:
            x_t:        Current sample, shape [B, ...]
            t:          Current timesteps, shape [B]
            t_prev:     Previous timesteps, shape [B] (typically t-1, clipped at 0)
            pred:       Network output (ε, v, or x₀ depending on param_type)
            param_type: One of {"eps", "v", "x0"}
            eta:        Stochasticity coefficient (0 => deterministic)
            noise:      Optional noise for the stochastic term

        Returns:
            x_{t-1} with same shape as x_t
        """
        if noise is None:
            noise = torch.randn_like(x_t)

        ab_prev = self._gather(self.alpha_bars, t_prev).view(-1, *([1] * (x_t.dim() - 1)))
        sigma = self.ddim_sigma(t, t_prev, eta).view(-1, *([1] * (x_t.dim() - 1)))

        x0_pred = self.to_x0(x_t, t, pred, param_type)
        eps_pred = self.to_eps(x_t, t, pred, param_type)

        dir_coeff = torch.clamp((1.0 - ab_prev) - sigma**2, min=0.0)
        dir_xt = torch.sqrt(dir_coeff) * eps_pred
        x_prev = torch.sqrt(ab_prev) * x0_pred + dir_xt + sigma * noise
        return x_prev


# ============================ Laplace pole logging ============================
def iter_laplace_bases(module: nn.Module):
    """
    Yield all LearnableLaplacianBasis submodules, if that class is available.
    """
    try:
        from Model.laptrans import LearnableLaplacianBasis  # type: ignore
    except Exception:
        return
    for m in module.modules():
        if isinstance(m, LearnableLaplacianBasis):
            yield m


@torch.no_grad()
def log_pole_health(modules: List[nn.Module], log_fn, step: int, tag_prefix: str = "") -> None:
    """
    Collect α (real) and ω (imag) pole stats across modules and log summary metrics
    via the provided logging function.
    """
    alphas, omegas = [], []
    for mod in modules:
        for lap in iter_laplace_bases(mod):
            tau = torch.nn.functional.softplus(lap._tau) + 1e-3
            alpha = lap.s_real.clamp_min(lap.alpha_min) * tau  # [k]
            omega = lap.s_imag * tau  # [k]
            alphas.append(alpha.detach().cpu())
            omegas.append(omega.detach().cpu())
    if not alphas:
        return
    alpha_cat = torch.cat([a.view(-1) for a in alphas])
    omega_cat = torch.cat([o.view(-1) for o in omegas])
    log_fn(
        {
            f"{tag_prefix}alpha_mean": alpha_cat.mean().item(),
            f"{tag_prefix}alpha_min": alpha_cat.min().item(),
            f"{tag_prefix}alpha_max": alpha_cat.max().item(),
            f"{tag_prefix}omega_abs_mean": omega_cat.abs().mean().item(),
        },
        step=step,
    )


def _print_log(metrics: dict, step: int, csv_path: Optional[str] = None) -> None:
    """
    Print pole metrics and optionally append to a CSV.
    """
    msg = " | ".join(f"{k}={v:.4g}" for k, v in metrics.items())
    print(f"[poles] step {step:>7d} | {msg}")
    if csv_path is not None:
        import csv
        import os

        head = ["step"] + list(metrics.keys())
        row = [step] + [metrics[k] for k in metrics]
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(head)
            w.writerow(row)


# ============================ VAE Latent stats helpers ============================
def compute_per_dim_stats(all_mu: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-dimension mean/std over [N, L, D].

    Returns:
        mu_per_dim [D], std_per_dim [D] (std is clamped to avoid zeros)
    """
    mu_per_dim = all_mu.mean(dim=(0, 1))
    std_per_dim = all_mu.std(dim=(0, 1)).clamp(min=1e-6)
    return mu_per_dim, std_per_dim


def normalize_and_check(all_mu: torch.Tensor, plot: bool = False):
    """
    Per-dimension normalize and (optionally) plot a histogram.

    Args:
        all_mu: [N, L, D]
        plot:   If True, shows a histogram of all normalized values.

    Returns:
        (all_mu_norm [N, L, D], mu_per_dim [D], std_per_dim [D])
    """
    mu_per_dim, std_per_dim = compute_per_dim_stats(all_mu)
    mu_b = mu_per_dim.view(1, 1, -1)
    std_b = std_per_dim.view(1, 1, -1)
    all_mu_norm = (all_mu - mu_b) / std_b

    all_vals = all_mu_norm.reshape(-1)
    print(f"Global mean (post-norm): {all_vals.mean().item():.6f}")
    print(f"Global std  (post-norm): {all_vals.std().item():.6f}")

    per_dim_mean = all_mu_norm.mean(dim=(0, 1))
    per_dim_std = all_mu_norm.std(dim=(0, 1))
    D = all_mu_norm.size(-1)
    print("\nPer-dim stats (first 10 dims or D if smaller):")
    for i in range(min(10, D)):  # safe preview
        print(f"  dim {i:2d}: mean={per_dim_mean[i]:7.4f}, std={per_dim_std[i]:7.4f}")

    if plot:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(4, 3))
        plt.hist(all_vals.cpu().numpy(), bins=500, range=(-5, 5))
        plt.title("Histogram of normalized μ values")
        plt.xlabel("Value")
        plt.ylabel("Count")
        plt.show()

    print(f"NaNs: {torch.isnan(all_mu_norm).sum().item()} | Infs: {torch.isinf(all_mu_norm).sum().item()}")
    print(f"Min: {all_mu_norm.min().item():.6f} | Max: {all_mu_norm.max().item():.6f}")
    return all_mu_norm, mu_per_dim, std_per_dim


def flatten_targets(
    yb: torch.Tensor, mask_bn: torch.Tensor, device: torch.device
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Flatten batched targets with a boolean mask.

    Args:
        yb:      [B, N, H]
        mask_bn: [B, N] boolean mask (True = keep)
        device:  Target device

    Returns:
        y_in:      [Beff, H, 1] or None if mask has no True
        batch_ids: [Beff] mapping each selected row back to its batch index, or None
    """
    y = yb.to(device)
    B, N, Hcur = y.shape
    y_flat = y.reshape(B * N, Hcur).unsqueeze(-1)  # [B*N, H, 1]
    m_flat = mask_bn.to(device).reshape(B * N)
    if not m_flat.any():
        return None, None
    y_in = y_flat[m_flat]
    batch_ids = torch.arange(B, device=device).unsqueeze(1).expand(B, N).reshape(B * N)[m_flat]
    return y_in, batch_ids


# ============================ EMA (for evaluation) ============================
class EMA:
    """
    Exponential Moving Average of model parameters (for eval).

    Usage:
        ema = EMA(model, decay=0.999)
        for step in training:
            ...
            ema.update(model)
        ema.store(model); ema.copy_to(model); evaluate(...); ema.restore(model)
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        self._backup = None  # filled by store()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].lerp_(p.detach(), 1.0 - self.decay)

    def store(self, model: nn.Module) -> None:
        self._backup = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    def copy_to(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.shadow[n].data)

    def restore(self, model: nn.Module) -> None:
        if self._backup is None:
            raise RuntimeError("EMA.restore() called before EMA.store().")
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self._backup[n].data)

    def state_dict(self) -> dict:
        """CPU copy of shadow weights."""
        return {k: v.cpu() for k, v in self.shadow.items()}

    def load_state_dict(self, sd: dict) -> None:
        for k, v in sd.items():
            if k in self.shadow:
                self.shadow[k] = v.clone()


# ============================ Latent helpers ============================
def ewma_std(x: torch.Tensor, lam: float = 0.94, eps: float = 1e-8) -> torch.Tensor:
    """
    Per-sample EWMA standard deviation across the time/window dimension.

    Args:
        x:   [B, L, D]
        lam: Decay (closer to 1 => heavier smoothing)
    Returns:
        [B, 1, D] EWMA std (broadcastable back to [B, L, D])
    """
    B, L, D = x.shape
    var = x.new_zeros(B, D)
    mean = x.new_zeros(B, D)
    for t in range(L):
        xt = x[:, t, :]
        mean = lam * mean + (1 - lam) * xt
        var = lam * var + (1 - lam) * (xt - mean) ** 2
    return (var + eps).sqrt().unsqueeze(1)


def two_stage_norm(
    mu: torch.Tensor,
    use_ewma: bool,
    ewma_lambda: float,
    mu_mean: torch.Tensor,
    mu_std: torch.Tensor,
    clip_val: float = 5.0,
) -> torch.Tensor:
    """
    Two-stage normalization used during diffusion training:
      1) Window-wise scaling (EWMA or simple per-window std)
      2) Global whitening (μ_mean, μ_std)

    Args:
        mu:          [B, L, Z]
        mu_mean:     [Z] (post-window statistics)
        mu_std:      [Z] (post-window statistics)
        clip_val:    Final clamp limit to stabilize tails

    Returns:
        mu_g: [B, L, Z]
    """
    if use_ewma:
        s = ewma_std(mu, lam=ewma_lambda)
    else:
        s = mu.std(dim=1, keepdim=True, correction=0).clamp_min(1e-6)
    mu_mean = mu_mean.to(device=mu.device, dtype=mu.dtype)
    mu_std = mu_std.to(device=mu.device, dtype=mu.dtype).clamp_min(1e-6)
    mu_w = (mu / s).clamp(-clip_val, clip_val)
    mu_g = (mu_w - mu_mean) / mu_std
    return mu_g.clamp(-clip_val, clip_val)


def normalize_cond_per_batch(cs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    z-score over (B, S) for each feature dimension; preserves gradients.

    Args:
        cs: [B, S, Hc]
    """
    m = cs.mean(dim=(0, 1), keepdim=True)
    v = cs.var(dim=(0, 1), keepdim=True, unbiased=False)
    return (cs - m) / (v.sqrt() + eps)


@torch.no_grad()
def compute_latent_stats(
    vae,
    dataloader,
    device: torch.device,
    use_ewma: bool,
    ewma_lambda: float,
    clip_val: float = 5.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute μ_mean, μ_std over window-normalized VAE latents.

    Expects dataloader to yield (_, y, meta) with:
      - y:    [B, N, H]
      - meta: dict containing "entity_mask": [B, N] (boolean)

    Returns:
      (mu_mean [Z], mu_std [Z])
    """
    all_mu_w = []
    for (_, y, meta) in dataloader:
        m = meta["entity_mask"]
        B, N, H = y.shape
        y_flat = y.reshape(B * N, H).unsqueeze(-1)
        m_flat = m.reshape(B * N)
        if not m_flat.any():
            continue
        y_in = y_flat[m_flat].to(device)
        _, mu, _ = vae(y_in)
        s = ewma_std(mu, lam=ewma_lambda) if use_ewma else mu.std(dim=1, keepdim=True, correction=0).clamp_min(1e-6)
        mu_w = (mu / s).clamp(-clip_val, clip_val)
        all_mu_w.append(mu_w.detach().cpu())

    if not all_mu_w:
        raise RuntimeError("No masked rows found while computing latent stats — check your entity_mask usage.")

    mu_cat = torch.cat(all_mu_w, dim=0)
    mu_mean = mu_cat.mean(dim=(0, 1)).to(device)
    mu_std = mu_cat.std(dim=(0, 1), correction=0).clamp_min(1e-6).to(device)
    return mu_mean, mu_std


@torch.no_grad()
def invert_two_stage_norm(
    x0_norm: torch.Tensor,
    mu_mean: torch.Tensor,
    mu_std: torch.Tensor,
    window_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Invert the two-stage normalization used for diffusion training.

    Args:
        x0_norm:     [B, L, Z] normalized latent (model's x₀ output)
        mu_mean:     [Z] global stats (post window-wise)
        mu_std:      [Z] global stats (post window-wise)
        window_scale:Optional scalar, [Z], or [B, 1, Z] (EWMA or per-window std). If None -> 1.0

    Returns:
        mu_est:      [B, L, Z] in original VAE latent space (μ)
    """
    mu_w = x0_norm * (mu_std.view(1, 1, -1)) + mu_mean.view(1, 1, -1)
    s = 1.0 if window_scale is None else window_scale
    mu_est = mu_w * s
    return mu_est


@torch.no_grad()
def decode_latents_with_vae(
    vae,
    x0_norm: torch.Tensor,
    mu_mean: torch.Tensor,
    mu_std: torch.Tensor,
    window_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Invert normalization and decode with the VAE decoder (no encoder skips).

    Args:
        x0_norm:      [B, L, Z]
        mu_mean/std:  [Z]
        window_scale: None, scalar, [Z] or [B, 1, Z]

    Returns:
        x_hat:        [B, L, 1] (same layout your VAE was trained on)
    """
    mu_est = invert_two_stage_norm(x0_norm, mu_mean, mu_std, window_scale=window_scale)
    x_hat = vae.decoder(mu_est, encoder_skips=None)
    return x_hat
