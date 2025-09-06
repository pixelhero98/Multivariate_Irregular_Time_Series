import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

import math
import torch
import torch.nn as nn
from typing import List, Optional, Tuple


# ============================LLapDiT utils============================
def set_torch():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    return device


def sample_t_uniform(scheduler, n: int, device: torch.device) -> torch.Tensor:
    return torch.randint(0, scheduler.timesteps, (n,), device=device)


def make_warmup_cosine(optimizer, total_steps, warmup_frac=0.05, base_lr=5e-4, min_lr=1e-6):
    warmup_steps = max(1, int(total_steps * warmup_frac))
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(min_lr / max(base_lr, 1e-12), 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _cosine_alpha_bar(t, s=0.008):
    """Continuous-time alpha_bar(t) from Nichol & Dhariwal (t in [0,1])."""
    return torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2


class NoiseScheduler(nn.Module):
    """
    Diffusion utilities with precomputed buffers and a DDIM sampler.
    Supports 'linear' or 'cosine' schedules and epsilon-/v-/x0-parameterization.
    """
    def __init__(self, timesteps: int = 1000, schedule: str = "cosine",
                 beta_start: float = 1e-4, beta_end: float = 2e-2):
        super().__init__()
        self.timesteps = int(timesteps)
        if schedule not in {"linear", "cosine"}:
            raise ValueError(f"Unknown schedule: {schedule}")
        self.schedule = schedule

        if schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, self.timesteps, dtype=torch.float32)
        else:
            ts = torch.linspace(0, 1, self.timesteps + 1, dtype=torch.float32)
            abar = _cosine_alpha_bar(ts)
            abar = abar / abar[0]
            alphas = torch.ones(self.timesteps, dtype=torch.float32)
            alphas[1:] = abar[1:self.timesteps] / abar[0:self.timesteps - 1]
            betas = (1.0 - alphas).clone()
            betas[1:] = betas[1:].clamp(min=1e-8, max=0.999)
            betas[0] = 0.0

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", 1.0 - betas)
        self.register_buffer("alpha_bars", torch.cumprod(1.0 - betas, dim=0))
        self.register_buffer("sqrt_alphas", torch.sqrt(1.0 - betas))
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(self.alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - self.alpha_bars))

    @torch.no_grad()
    def timesteps_desc(self):
        return torch.arange(self.timesteps - 1, -1, -1, dtype=torch.long, device=self.alpha_bars.device)

    def _gather(self, buf: torch.Tensor, t: torch.Tensor):
        t = t.clamp(min=0, max=self.timesteps - 1).to(device=buf.device, dtype=torch.long)
        return buf.gather(0, t)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab   = self._gather(self.sqrt_alpha_bars, t).view(-1, *([1] * (x0.dim()-1)))
        sqrt_1_ab = self._gather(self.sqrt_one_minus_alpha_bars, t).view(-1, *([1] * (x0.dim()-1)))
        x_t = sqrt_ab * x0 + sqrt_1_ab * noise
        return x_t, noise

    def pred_x0_from_eps(self, x_t, t, eps):
        alpha = self._gather(self.sqrt_alpha_bars, t).view(-1, *([1] * (x_t.dim()-1)))
        sigma = self._gather(self.sqrt_one_minus_alpha_bars, t).view(-1, *([1] * (x_t.dim()-1)))
        return (x_t - sigma * eps) / (alpha + 1e-12)

    def pred_eps_from_x0(self, x_t, t, x0):
        alpha = self._gather(self.sqrt_alpha_bars, t).view(-1, *([1] * (x_t.dim()-1)))
        sigma = self._gather(self.sqrt_one_minus_alpha_bars, t).view(-1, *([1] * (x_t.dim()-1)))
        return (x_t - alpha * x0) / (sigma + 1e-12)

    def pred_x0_from_v(self, x_t, t, v):
        alpha = self._gather(self.sqrt_alpha_bars, t).view(-1, *([1] * (x_t.dim()-1)))
        sigma = self._gather(self.sqrt_one_minus_alpha_bars, t).view(-1, *([1] * (x_t.dim()-1)))
        return alpha * x_t - sigma * v

    def pred_eps_from_v(self, x_t, t, v):
        alpha = self._gather(self.sqrt_alpha_bars, t).view(-1, *([1] * (x_t.dim()-1)))
        sigma = self._gather(self.sqrt_one_minus_alpha_bars, t).view(-1, *([1] * (x_t.dim()-1)))
        return sigma * x_t + alpha * v

    def v_from_eps(self, x_t, t, eps):
        alpha = self._gather(self.sqrt_alpha_bars, t).view(-1, *([1] * (x_t.dim()-1)))
        sigma = self._gather(self.sqrt_one_minus_alpha_bars, t).view(-1, *([1] * (x_t.dim()-1)))
        return (eps - sigma * x_t) / (alpha + 1e-12)

    def to_x0(self, x_t, t, pred, param_type: str):
        if     param_type == "eps": return self.pred_x0_from_eps(x_t, t, pred)
        elif   param_type == "v":   return self.pred_x0_from_v  (x_t, t, pred)
        elif   param_type == "x0":  return pred
        else: raise ValueError("param_type must be 'eps', 'v', or 'x0'")

    def to_eps(self, x_t, t, pred, param_type: str):
        if     param_type == "eps": return pred
        elif   param_type == "v":   return self.pred_eps_from_v (x_t, t, pred)
        elif   param_type == "x0":  return self.pred_eps_from_x0(x_t, t, pred)
        else: raise ValueError("param_type must be 'eps', 'v', or 'x0'")

    @torch.no_grad()
    def ddim_sigma(self, t, t_prev, eta: float):
        ab_t    = self._gather(self.alpha_bars, t)
        ab_prev = self._gather(self.alpha_bars, t_prev)
        sigma = eta * torch.sqrt((1.0 - ab_prev) / (1.0 - ab_t)) * torch.sqrt(1.0 - ab_t / (ab_prev + 1e-12))
        return sigma.view(-1)

    @torch.no_grad()
    def ddim_step_from(self, x_t, t, t_prev, pred, param_type: str, eta: float = 0.0, noise: torch.Tensor = None):
        if noise is None:
            noise = torch.randn_like(x_t)
        ab_prev = self._gather(self.alpha_bars, t_prev).view(-1, *([1] * (x_t.dim()-1)))
        sigma   = self.ddim_sigma(t, t_prev, eta).view(-1, *([1] * (x_t.dim()-1)))

        x0_pred = self.to_x0(x_t, t, pred, param_type)
        eps_pred= self.to_eps(x_t, t, pred, param_type)

        dir_coeff = torch.clamp((1.0 - ab_prev) - sigma**2, min=0.0)
        dir_xt = torch.sqrt(dir_coeff) * eps_pred
        x_prev  = torch.sqrt(ab_prev) * x0_pred + dir_xt + sigma * noise
        return x_prev


# ============================ Laplace pole logging ============================
def iter_laplace_bases(module):
    from Model.laptrans import LearnableLaplacianBasis
    for m in module.modules():
        if isinstance(m, LearnableLaplacianBasis):
            yield m


@torch.no_grad()
def log_pole_health(modules: List[nn.Module], log_fn, step: int, tag_prefix: str = ""):
    alphas, omegas = [], []
    for mod in modules:
        for lap in iter_laplace_bases(mod):
            tau = torch.nn.functional.softplus(lap._tau) + 1e-3
            alpha = lap.s_real.clamp_min(lap.alpha_min) * tau  # [k]
            omega = lap.s_imag * tau                            # [k]
            alphas.append(alpha.detach().cpu())
            omegas.append(omega.detach().cpu())
    if not alphas:
        return
    alpha_cat = torch.cat([a.view(-1) for a in alphas])
    omega_cat = torch.cat([o.view(-1) for o in omegas])
    log_fn({f"{tag_prefix}alpha_mean": alpha_cat.mean().item(),
            f"{tag_prefix}alpha_min": alpha_cat.min().item(),
            f"{tag_prefix}alpha_max": alpha_cat.max().item(),
            f"{tag_prefix}omega_abs_mean": omega_cat.abs().mean().item()}, step=step)


def _print_log(metrics: dict, step: int, csv_path: str = None):
    msg = " | ".join(f"{k}={v:.4g}" for k, v in metrics.items())
    print(f"[poles] step {step:>7d} | {msg}")
    if csv_path is not None:
        import csv, os
        head = ["step"] + list(metrics.keys())
        row  = [step]  + [metrics[k] for k in metrics]
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(head)
            w.writerow(row)


# ============================ VAE Latent stats helpers ============================
def flatten_targets(yb: torch.Tensor, mask_bn: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """yb: [B,N,H] -> y_in: [Beff,H,1], batch_ids: [Beff] mapping to B for cond rows"""
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
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].lerp_(p.detach(), 1.0 - self.decay)

    def store(self, model):
        self._backup = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    def copy_to(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.shadow[n].data)

    def restore(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self._backup[n].data)

    def state_dict(self):
        return {k: v.cpu() for k, v in self.shadow.items()}

    def load_state_dict(self, sd):
        for k, v in sd.items():
            if k in self.shadow:
                self.shadow[k] = v.clone()

# ============================ Latent helpers ============================
def ewma_std(x: torch.Tensor, lam: float = 0.94, eps: float = 1e-8) -> torch.Tensor:
    B, L, D = x.shape
    var = x.new_zeros(B, D)
    mean = x.new_zeros(B, D)
    for t in range(L):
        xt = x[:, t, :]
        mean = lam * mean + (1 - lam) * xt
        var  = lam * var  + (1 - lam) * (xt - mean) ** 2
    return (var + eps).sqrt().unsqueeze(1)


def two_stage_norm(mu: torch.Tensor, use_ewma: bool, ewma_lambda: float,
                   mu_mean: torch.Tensor, mu_std: torch.Tensor,
                   clip_val: float = 5.0) -> torch.Tensor:
    if use_ewma:
        s = ewma_std(mu, lam=ewma_lambda)
    else:
        s = mu.std(dim=1, keepdim=True, correction=0).clamp_min(1e-6)
    mu_mean = mu_mean.to(device=mu.device, dtype=mu.dtype)
    mu_std  = mu_std.to(device=mu.device, dtype=mu.dtype).clamp_min(1e-6)
    mu_w = (mu / s).clamp(-clip_val, clip_val)
    mu_g = (mu_w - mu_mean) / mu_std
    return mu_g.clamp(-clip_val, clip_val)


def normalize_cond_per_batch(cs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """z-score over (B,S) for each feature dim; keeps gradients. cs: [B,S,Hc]"""
    m = cs.mean(dim=(0, 1), keepdim=True)
    v = cs.var (dim=(0, 1), keepdim=True, unbiased=False)
    return (cs - m) / (v.sqrt() + eps)


@torch.no_grad()
def compute_latent_stats(vae, dataloader, device, use_ewma: bool, ewma_lambda: float,
                         clip_val: float = 5.0):
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

    mu_cat = torch.cat(all_mu_w, dim=0)
    mu_mean = mu_cat.mean(dim=(0, 1)).to(device)
    mu_std  = mu_cat.std (dim=(0, 1), correction=0).clamp_min(1e-6).to(device)
    return mu_mean, mu_std



@torch.no_grad()
def invert_two_stage_norm(x0_norm: torch.Tensor,
                          mu_mean: torch.Tensor,
                          mu_std: torch.Tensor,
                          window_scale: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Invert the two-stage normalization used for diffusion training.
      x0_norm:     [B,L,Z] normalized latent (what diffusion outputs as x0)
      mu_mean/std: [Z] global stats computed AFTER the window-wise step
      window_scale:[1] or [Z] or [B,1,Z] (EWMA/plained per-window std). If None, uses 1.0.
    Returns:
      mu_est:      [B,L,Z] in the original VAE latent space (μ)
    """
    # undo global whitening
    mu_w = x0_norm * (mu_std.view(1, 1, -1)) + mu_mean.view(1, 1, -1)
    # undo window-wise scaling (if provided)
    if window_scale is None:
        s = 1.0
    else:
        # allow scalar, per-dim [Z], or per-sample [B,1,Z]
        s = window_scale
    mu_est = mu_w * s
    return mu_est


@torch.no_grad()
def decode_latents_with_vae(vae, x0_norm: torch.Tensor,
                            mu_mean: torch.Tensor, mu_std: torch.Tensor,
                            window_scale: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Invert normalization and decode with the VAE decoder (no encoder skips).
      - x0_norm: [B,L,Z] normalized latent from diffusion
      - window_scale: None, scalar, [Z] or [B,1,Z]
    Returns:
      x_hat: [B,L,1]  (same layout your VAE was trained on)
    """
    mu_est = invert_two_stage_norm(x0_norm, mu_mean, mu_std, window_scale=window_scale)
    # your decoder accepts z with optional skips=None
    x_hat = vae.decoder(mu_est, encoder_skips=None)
    return x_hat


def build_context(model, V: torch.Tensor, T: torch.Tensor,
                  mask_bn: torch.Tensor, device: torch.device, norm: bool = True) -> torch.Tensor:
    """Returns normalized cond_summary: [B,S,Hm]"""
    with torch.no_grad():
        series_diff = T.permute(0, 2, 1, 3).to(device)  # [B,K,N,F]
        series      = V.permute(0, 2, 1, 3).to(device)  # [B,K,N,F]
        mask_bn     = mask_bn.to(device)
        cond_summary, _ = model.context(x=series, ctx_diff=series_diff, entity_mask=mask_bn)
        if norm:
            cond_summary = normalize_cond_per_batch(cond_summary)
    return cond_summary


def encode_mu_norm(vae, y_in: torch.Tensor, *, use_ewma: bool, ewma_lambda: float,
                   mu_mean: torch.Tensor, mu_std: torch.Tensor) -> torch.Tensor:
    """VAE encode then two-stage normalize; returns [Beff, H, Z]"""
    with torch.no_grad():
        _, mu, _ = vae(y_in)
        mu_norm = two_stage_norm(mu, use_ewma=use_ewma, ewma_lambda=ewma_lambda, mu_mean=mu_mean, mu_std=mu_std)
        mu_norm = torch.nan_to_num(mu_norm, nan=0.0, posinf=0.0, neginf=0.0)
    return mu_norm


def diffusion_loss(model, scheduler, x0_lat_norm: torch.Tensor, t: torch.Tensor,
                   *, cond_summary: Optional[torch.Tensor], predict_type: str = "v",
                   weight_scheme: str = "none", minsnr_gamma: float = 5.0,
                   sc_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    MSE on v/eps with channel-invariant reduction.
    Optional min-SNR weighting.
    """
    noise = torch.randn_like(x0_lat_norm)
    x_t, eps_true = scheduler.q_sample(x0_lat_norm, t, noise)
    pred = model(x_t, t, cond_summary=cond_summary, sc_feat=sc_feat)
    target = eps_true if predict_type == "eps" else scheduler.v_from_eps(x_t, t, eps_true)

    # [B,H,Z] -> per-sample loss: mean over H, sum over Z  (scale-invariant to Z)
    err = (pred - target).pow(2)              # [B,H,Z]
    per_sample = err.mean(dim=1).sum(dim=1)   # [B]

    if weight_scheme == 'none':
        return per_sample.mean()
    elif weight_scheme == 'weighted_min_snr':
        abar = scheduler.alpha_bars[t]
        snr  = abar / (1.0 - abar).clamp_min(1e-8)
        gamma = minsnr_gamma
        w = torch.minimum(snr, torch.as_tensor(gamma, device=snr.device, dtype=snr.dtype))
        w = w / (snr + 1.0)
        return (w.detach() * per_sample).mean()


@torch.no_grad()
def calculate_v_variance(scheduler, dataloader, vae, device, latent_stats,
                         use_ewma, ewma_lambda):
    """
    Calculates the variance of the v-prediction target over a given dataloader.
    """
    all_v_targets = []
    print("Calculating variance of v-prediction target...")

    # Unpack the pre-computed latent statistics
    mu_mean, mu_std = latent_stats

    # Loop through the validation set
    for xb, yb, meta in dataloader:
        # This block is the same as in your validate() function
        # It gets the normalized latent variable 'mu_norm' which is the x0 for diffusion
        y_in, _ = flatten_targets(yb, meta["entity_mask"], device)
        if y_in is None:
            continue

        mu_norm = encode_mu_norm(
            vae, y_in,
            use_ewma=use_ewma,
            ewma_lambda=ewma_lambda,
            mu_mean=mu_mean,
            mu_std=mu_std
        )

        # Now, simulate the process for creating the 'v' target
        # 1. Sample random timesteps
        t = sample_t_uniform(scheduler, mu_norm.size(0), device)

        # 2. Create the noise that would be added
        noise = torch.randn_like(mu_norm)

        # 3. Apply the forward process to get the noised latent x_t
        x_t, _ = scheduler.q_sample(mu_norm, t, noise)

        # 4. Calculate the ground-truth 'v' from x_t and the noise
        v_target = scheduler.v_from_eps(x_t, t, noise)
        all_v_targets.append(v_target.detach().cpu())

    # Concatenate all batches and compute the final variance
    if not all_v_targets:
        print("Warning: No valid data found to calculate variance.")
        return float('nan')

    all_v_targets_cat = torch.cat(all_v_targets, dim=0)
    v_variance = all_v_targets_cat.var().item()

    return v_variance
