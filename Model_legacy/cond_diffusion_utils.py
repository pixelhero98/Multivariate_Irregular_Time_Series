import math
import torch
import torch.nn as nn
from typing import List, Optional, Tuple

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

    # ---------- conversions ----------
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

    # ---------- DDIM stepping ----------
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

# ---------------- Laplacian pole logging ----------------
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

# ---------------- Latent stats helpers ----------------
def compute_per_dim_stats(all_mu: torch.Tensor):
    """all_mu: [N, L, D] -> (mu_per_dim[D], std_per_dim[D])."""
    mu_per_dim  = all_mu.mean(dim=(0, 1))
    std_per_dim = all_mu.std (dim=(0, 1)).clamp(min=1e-6)
    return mu_per_dim, std_per_dim

def normalize_and_check(all_mu: torch.Tensor, plot: bool = False):
    """
    Per-dimension normalize and (optionally) plot a histogram.
    returns (all_mu_norm, mu_per_dim, std_per_dim)
    """
    mu_per_dim, std_per_dim = compute_per_dim_stats(all_mu)
    mu_b  = mu_per_dim.view(1, 1, -1)
    std_b = std_per_dim.view(1, 1, -1)
    all_mu_norm = (all_mu - mu_b) / std_b

    all_vals = all_mu_norm.reshape(-1)
    print(f"Global mean (post-norm): {all_vals.mean().item():.6f}")
    print(f"Global std  (post-norm): {all_vals.std().item():.6f}")

    per_dim_mean = all_mu_norm.mean(dim=(0, 1))
    per_dim_std  = all_mu_norm.std (dim=(0, 1))
    D = all_mu_norm.size(-1)
    print("\nPer-dim stats (first 10 dims or D if smaller):")
    for i in range(min(10, D)):  # fixed (was min(1e10, D))
        print(f"  dim {i:2d}: mean={per_dim_mean[i]:7.4f}, std={per_dim_std[i]:7.4f}")

    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(4, 3))
        plt.hist(all_vals.cpu().numpy(), bins=500, range=(-5, 5))
        plt.title("Histogram of normalized μ values")
        plt.xlabel("Value"); plt.ylabel("Count")
        plt.show()

    print(f"NaNs: {torch.isnan(all_mu_norm).sum().item()} | Infs: {torch.isinf(all_mu_norm).sum().item()}")
    print(f"Min: {all_mu_norm.min().item():.6f} | Max: {all_mu_norm.max().item():.6f}")
    return all_mu_norm, mu_per_dim, std_per_dim

# ---------------- EMA (for evaluation) ----------------
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

# ======================================================================
# NEW: inverse two-stage normalization + DDIM sampling + decode helpers
# ======================================================================

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

