
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

# ============================ Torch helpers ============================
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

        # ---- build betas ----
        if schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, self.timesteps, dtype=torch.float32)
            betas = betas.clamp(min=1e-8, max=0.999)
        else:
            # cosine ᾱ(t) from Nichol & Dhariwal; turn into alphas and then betas
            ts = torch.linspace(0, 1, self.timesteps + 1, dtype=torch.float32)
            abar = _cosine_alpha_bar(ts)
            abar = abar / abar[0]  # ᾱ(0) = 1
            alphas = torch.ones(self.timesteps, dtype=torch.float32)
            alphas[1:] = abar[1:self.timesteps] / abar[0:self.timesteps - 1]
            betas = (1.0 - alphas).clone()
            betas[1:] = betas[1:].clamp(min=1e-8, max=0.999)

        betas[0] = 0.0
        self.register_buffer("betas", betas)
        alphas = 1.0 - betas
        self.register_buffer("alphas", alphas)
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("alpha_bars", alpha_bars)
        ab = alpha_bars.clamp(0.0, 1.0)
        self.register_buffer("sqrt_alphas", torch.sqrt(alphas.clamp(0.0, 1.0)))
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(ab))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt((1.0 - ab).clamp(0.0, 1.0)))

    @torch.no_grad()
    def timesteps_desc(self):
        return torch.arange(self.timesteps - 1, -1, -1, dtype=torch.long, device=self.alpha_bars.device)

    def _gather(self, buf: torch.Tensor, t: torch.Tensor):
        t = t.clamp(min=0, max=self.timesteps - 1).to(device=buf.device, dtype=torch.long)
        return buf.gather(0, t)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self._gather(self.sqrt_alpha_bars, t).view(-1, *([1] * (x0.dim() - 1)))
        sqrt_1_ab = self._gather(self.sqrt_one_minus_alpha_bars, t).view(-1, *([1] * (x0.dim() - 1)))
        x_t = sqrt_ab * x0 + sqrt_1_ab * noise
        return x_t, noise

    def pred_x0_from_eps(self, x_t, t, eps):
        alpha = self._gather(self.sqrt_alpha_bars, t).view(-1, *([1] * (x_t.dim() - 1)))
        sigma = self._gather(self.sqrt_one_minus_alpha_bars, t).view(-1, *([1] * (x_t.dim() - 1)))
        return (x_t - sigma * eps) / (alpha + 1e-12)

    def pred_eps_from_x0(self, x_t, t, x0):
        alpha = self._gather(self.sqrt_alpha_bars, t).view(-1, *([1] * (x_t.dim() - 1)))
        sigma = self._gather(self.sqrt_one_minus_alpha_bars, t).view(-1, *([1] * (x_t.dim() - 1)))
        return (x_t - alpha * x0) / (sigma + 1e-12)

    def pred_x0_from_v(self, x_t, t, v):
        alpha = self._gather(self.sqrt_alpha_bars, t).view(-1, *([1] * (x_t.dim() - 1)))
        sigma = self._gather(self.sqrt_one_minus_alpha_bars, t).view(-1, *([1] * (x_t.dim() - 1)))
        return alpha * x_t - sigma * v

    def pred_eps_from_v(self, x_t, t, v):
        alpha = self._gather(self.sqrt_alpha_bars, t).view(-1, *([1] * (x_t.dim() - 1)))
        sigma = self._gather(self.sqrt_one_minus_alpha_bars, t).view(-1, *([1] * (x_t.dim() - 1)))
        return sigma * x_t + alpha * v

    def v_from_eps(self, x_t, t, eps):
        alpha = self._gather(self.sqrt_alpha_bars, t).view(-1, *([1] * (x_t.dim() - 1)))
        sigma = self._gather(self.sqrt_one_minus_alpha_bars, t).view(-1, *([1] * (x_t.dim() - 1)))
        return (eps - sigma * x_t) / (alpha + 1e-12)

    def to_x0(self, x_t, t, pred, param_type: str):
        if param_type == "eps":
            return self.pred_x0_from_eps(x_t, t, pred)
        elif param_type == "v":
            return self.pred_x0_from_v(x_t, t, pred)
        elif param_type == "x0":
            return pred
        else:
            raise ValueError("param_type must be 'eps', 'v', or 'x0'")

    def to_eps(self, x_t, t, pred, param_type: str):
        if param_type == "eps":
            return pred
        elif param_type == "v":
            return self.pred_eps_from_v(x_t, t, pred)
        elif param_type == "x0":
            return self.pred_eps_from_x0(x_t, t, pred)
        else:
            raise ValueError("param_type must be 'eps', 'v', or 'x0'")

    @torch.no_grad()
    def ddim_sigma(self, t, t_prev, eta: float):
        ab_t = self._gather(self.alpha_bars, t)
        ab_prev = self._gather(self.alpha_bars, t_prev)
        sigma = eta * torch.sqrt((1.0 - ab_prev) / (1.0 - ab_t)) * torch.sqrt(1.0 - ab_t / (ab_prev + 1e-12))
        return sigma.view(-1)

    @torch.no_grad()
    def ddim_step_from(self, x_t, t, t_prev, pred, param_type: str, eta: float = 0.0, noise: torch.Tensor = None):
        if noise is None:
            noise = torch.randn_like(x_t)
        ab_prev = self._gather(self.alpha_bars, t_prev).view(-1, *([1] * (x_t.dim() - 1)))
        sigma = self.ddim_sigma(t, t_prev, eta).view(-1, *([1] * (x_t.dim() - 1)))

        x0_pred = self.to_x0(x_t, t, pred, param_type)
        eps_pred = self.to_eps(x_t, t, pred, param_type)

        dir_coeff = torch.clamp((1.0 - ab_prev) - sigma ** 2, min=0.0)
        dir_xt = torch.sqrt(dir_coeff) * eps_pred
        x_prev = torch.sqrt(ab_prev) * x0_pred + dir_xt + sigma * noise
        return x_prev


# ============================ EMA Weights ============================
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


# ============================ Latent + context helpers ============================
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


def normalize_cond_per_batch(cs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """z-score over (B,S) for each feature dim; keeps gradients. cs: [B,S,Hc]"""
    m = cs.mean(dim=(0, 1), keepdim=True)
    v = cs.var(dim=(0, 1), keepdim=True, unbiased=False)
    return (cs - m) / (v.sqrt() + eps)


@torch.no_grad()
def build_context(model, V: torch.Tensor, T: torch.Tensor,
                  mask_bn: torch.Tensor, device: torch.device, norm: bool = False) -> torch.Tensor:
    """Returns cond_summary: [B,S,Hm]"""
    series_diff = T.permute(0, 2, 1, 3).to(device)  # [B,K,N,F]
    series = V.permute(0, 2, 1, 3).to(device)       # [B,K,N,F]
    mask_bn = mask_bn.to(device)
    cond_summary, _ = model.context(x=series, ctx_diff=series_diff, entity_mask=mask_bn)
    if norm:
        cond_summary = normalize_cond_per_batch(cond_summary)
    return cond_summary


# ===== Global z-score helpers for latents =====
def simple_norm(mu: torch.Tensor, mu_mean: torch.Tensor, mu_std: torch.Tensor, clip_val: float = None) -> torch.Tensor:
    mu_mean = mu_mean.to(device=mu.device, dtype=mu.dtype).view(1,1,-1)
    mu_std  = mu_std.to(device=mu.device, dtype=mu.dtype).clamp_min(1e-6).view(1,1,-1)
    out = (mu - mu_mean) / mu_std
    return out.clamp(-clip_val, clip_val) if clip_val is not None else out

def invert_simple_norm(x: torch.Tensor, mu_mean: torch.Tensor, mu_std: torch.Tensor) -> torch.Tensor:
    mu_mean = mu_mean.to(device=x.device, dtype=x.dtype).view(1,1,-1)
    mu_std  = mu_std.to(device=x.device, dtype=x.dtype).view(1,1,-1)
    return x * mu_std + mu_mean


@torch.no_grad()
def compute_latent_stats(vae, dataloader, device):
    """
    Compute dataset-level latent mean/std for z (reparameterized latent).
    Var(z) = Var(mu) + E[sigma^2].
    Returns (z_mean, z_std).
    """
    sum_mu = None
    sum_mu2 = None
    sum_sig2 = None
    count = 0
    for (_, y, meta) in dataloader:
        m = meta["entity_mask"]
        B, N, H = y.shape
        y_flat = y.reshape(B * N, H).unsqueeze(-1)
        m_flat = m.reshape(B * N)
        if not m_flat.any():
            continue
        y_in = y_flat[m_flat].to(device)
        _, mu, logvar = vae(y_in)
        sig2 = logvar.exp()
        # accumulate over (B,L)
        mu_sum_batch  = mu.sum(dim=(0,1))
        mu2_sum_batch = (mu**2).sum(dim=(0,1))
        sig2_sum_batch = sig2.sum(dim=(0,1))
        n_batch = mu.size(0) * mu.size(1)
        if sum_mu is None:
            sum_mu, sum_mu2, sum_sig2 = mu_sum_batch.detach().cpu(), mu2_sum_batch.detach().cpu(), sig2_sum_batch.detach().cpu()
        else:
            sum_mu  += mu_sum_batch.detach().cpu()
            sum_mu2 += mu2_sum_batch.detach().cpu()
            sum_sig2 += sig2_sum_batch.detach().cpu()
        count += n_batch
    if count == 0:
        raise RuntimeError("No valid samples for latent stats")
    mean_mu = (sum_mu / count).to(device)
    var_mu  = ((sum_mu2 / count).to(device) - mean_mu**2).clamp_min(0.0)
    mean_sig2 = (sum_sig2 / count).to(device)
    z_mean = mean_mu
    z_std  = (var_mu + mean_sig2).clamp_min(1e-6).sqrt()
    return z_mean, z_std


def encode_mu_norm(vae, y_in: torch.Tensor, *, mu_mean: torch.Tensor, mu_std: torch.Tensor) -> torch.Tensor:
    """VAE encode -> sample z -> global z-score; returns [Beff, H, Z]"""
    with torch.no_grad():
        _, mu, logvar = vae(y_in)
        z = vae.reparameterize(mu, logvar)
        z_norm = simple_norm(z, mu_mean, mu_std, clip_val=None)
        z_norm = torch.nan_to_num(z_norm, nan=0.0, posinf=0.0, neginf=0.0)
    return z_norm


def diffusion_loss(model, scheduler, x0_lat_norm: torch.Tensor, t: torch.Tensor,
                   *, cond_summary: Optional[torch.Tensor], predict_type: str = "v",
                   weight_scheme: str = "none", minsnr_gamma: float = 5.0,
                   sc_feat: Optional[torch.Tensor] = None,
                   reuse_xt_eps: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
    """
    MSE on v/eps with optional MinSNR weighting.
    If reuse_xt_eps=(x_t, eps_true) is provided, use that instead of re-sampling.
    """
    if reuse_xt_eps is None:
        noise = torch.randn_like(x0_lat_norm)
        x_t, eps_true = scheduler.q_sample(x0_lat_norm, t, noise)
    else:
        x_t, eps_true = reuse_xt_eps

    pred = model(x_t, t, cond_summary=cond_summary, sc_feat=sc_feat)
    target = eps_true if predict_type == "eps" else scheduler.v_from_eps(x_t, t, eps_true)

    err = (pred - target).pow(2)  # [B,H,Z]
    per_sample = err.mean(dim=1).sum(dim=1)  # [B]

    if weight_scheme == 'none':
        return per_sample.mean()
    elif weight_scheme == 'weighted_min_snr':
        abar = scheduler.alpha_bars[t]
        snr = abar / (1.0 - abar).clamp_min(1e-8)
        gamma = minsnr_gamma
        w = torch.minimum(snr, torch.as_tensor(gamma, device=snr.device, dtype=snr.dtype))
        w = w / (snr + 1.0)
        return (w.detach() * per_sample).mean()


def decode_latents_with_vae(vae, x0_norm: torch.Tensor,
                            mu_mean: torch.Tensor, mu_std: torch.Tensor) -> torch.Tensor:
    """
    Invert normalization to z and decode with the VAE decoder (no encoder skips).
    """
    z_est = invert_simple_norm(x0_norm, mu_mean, mu_std)
    x_hat = vae.decoder(z_est, encoder_skips=None)
    return x_hat


@torch.no_grad()
def calculate_v_variance(scheduler, dataloader, vae, device, latent_stats):
    """
    Calculates the variance of the v-prediction target over a given dataloader.
    """
    all_v_targets = []
    print("Calculating variance of v-prediction target...")

    # Unpack the pre-computed latent statistics
    z_mean, z_std = latent_stats

    # Loop through the validation set
    for xb, yb, meta in dataloader:
        y_in, _ = flatten_targets(yb, meta["entity_mask"], device)
        if y_in is None:
            continue

        z_norm = encode_mu_norm(
            vae, y_in,
            mu_mean=z_mean,
            mu_std=z_std
        )

        # 1. Sample random timesteps
        t = sample_t_uniform(scheduler, z_norm.size(0), device)
        # 2. Create the noise
        noise = torch.randn_like(z_norm)
        # 3. Apply forward process
        x_t, _ = scheduler.q_sample(z_norm, t, noise)
        # 4. v target
        v_target = scheduler.v_from_eps(x_t, t, noise)
        all_v_targets.append(v_target.detach().cpu())

    if not all_v_targets:
        print("Warning: No valid data found to calculate variance.")
        return float('nan')

    all_v_targets_cat = torch.cat(all_v_targets, dim=0)
    v_variance = all_v_targets_cat.var(correction=0).item()
    return v_variance

def ewma_std(x: torch.Tensor, lam: float = 0.94, eps: float = 1e-8) -> torch.Tensor:
    """Exponentially Weighted Moving Std over time dimension (L). x: [B,L,Z] -> [B,1,Z]"""
    B, L, D = x.shape
    var = x.new_zeros(B, D)
    mean = x.new_zeros(B, D)
    for t in range(L):
        xt = x[:, t, :]
        mean = lam * mean + (1 - lam) * xt
        var = lam * var + (1 - lam) * (xt - mean) ** 2
    return (var + eps).sqrt().unsqueeze(1)


@torch.no_grad()
def _flatten_for_mask(yb, mask_bn, device):
    return flatten_targets(yb, mask_bn, device)


def encode_z_norm(vae, y_in: torch.Tensor, *, z_mean: torch.Tensor, z_std: torch.Tensor) -> torch.Tensor:
    # alias for encode_mu_norm, but with z_* names
    return encode_mu_norm(vae, y_in, mu_mean=z_mean, mu_std=z_std)


def get_window_scale(vae, y_true: torch.Tensor, config) -> torch.Tensor:
    """
    Compute per-window latent scale from the *ground-truth* target window.
    Returns s: [B,1,Z] (EWMA or std across L).
    """
    with torch.no_grad():
        _, mu, logvar = vae(y_true)       # [B,L,Z]
        # use μ-based scale by default (historical choice); could also use z
        if getattr(config, "USE_EWMA", False):
            s = ewma_std(mu, lam=getattr(config, "EWMA_LAMBDA", 0.94))
        else:
            s = mu.std(dim=1, keepdim=True, correction=0).clamp_min(1e-6)
        return s

# Extend decode to accept optional window_scale and apply it multiplicatively on z
def decode_latents_with_vae(vae, x0_norm: torch.Tensor,
                            z_mean: torch.Tensor = None, z_std: torch.Tensor = None,
                            mu_mean: torch.Tensor = None, mu_std: torch.Tensor = None,
                            window_scale: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Invert normalization to z and decode. Optional per-window scale can be applied.
    Accepts either (z_mean,z_std) or legacy (mu_mean,mu_std) arg names.
    """
    if z_mean is None or z_std is None:
        # fallback to mu_* args
        z_mean, z_std = mu_mean, mu_std
    z_est = invert_simple_norm(x0_norm, z_mean, z_std)
    if window_scale is not None:
        # support scalar, [Z], or [B,1,Z]
        if window_scale.dim() == 1:
            window_scale = window_scale.view(1,1,-1).to(z_est.device, z_est.dtype)
        elif window_scale.dim() == 2:
            window_scale = window_scale.view(1,1,-1).to(z_est.device, z_est.dtype)
        z_est = z_est * window_scale.to(z_est.device, z_est.dtype)
    x_hat = vae.decoder(z_est, encoder_skips=None)
    return x_hat
