import math
import torch
import torch.nn as nn

def _cosine_alpha_bar(t, s=0.008):
    """
    Continuous-time alpha_bar(t) from Nichol & Dhariwal 'Improved Denoising Diffusion Models'.
    t in [0,1]. Returns alpha_bar(t) in [0,1].
    """
    return torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2


class NoiseScheduler(nn.Module):
    """
    Diffusion utilities with precomputed buffers and a DDIM sampler.
    Supports 'linear' or 'cosine' schedules and epsilon- or v-parameterization.
    """
    def __init__(self, timesteps: int = 1000, schedule: str = "cosine",
                 beta_start: float = 1e-4, beta_end: float = 2e-2):
        super().__init__()
        self.timesteps = int(timesteps)
        if schedule not in {"linear", "cosine"}:
            raise ValueError(f"Unknown schedule: {schedule}")
        self.schedule = schedule

        # Build betas
        if schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, self.timesteps, dtype=torch.float32)
            alphas = 1.0 - betas
            alpha_bars = torch.cumprod(alphas, dim=0)
        else:
            # cosine: derive betas from alpha_bar using finite differences
            ts = torch.linspace(0, 1, self.timesteps + 1, dtype=torch.float32)
            abar = _cosine_alpha_bar(ts)  # length T+1
            abar = abar / abar[0]         # normalize so abar(0)=1
            alpha_bars = abar[1:]         # t=1..T
            alphas = alpha_bars / torch.cat([torch.tensor([1.0], dtype=alpha_bars.dtype), alpha_bars[:-1]], dim=0)
            betas = (1.0 - alphas).clamp(min=1e-8, max=0.999)

        # Buffers
        self.register_buffer("betas", betas)                    # [T]
        self.register_buffer("alphas", 1.0 - betas)             # [T]
        self.register_buffer("alpha_bars", torch.cumprod(1.0 - betas, dim=0))  # [T]
        self.register_buffer("sqrt_alphas", torch.sqrt(1.0 - betas))           # [T]
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(self.alpha_bars))   # [T]
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - self.alpha_bars))  # [T]

    @torch.no_grad()
    def timesteps_desc(self):
        return torch.arange(self.timesteps - 1, -1, -1, dtype=torch.long, device=self.alpha_bars.device)

    def _gather(self, buf: torch.Tensor, t: torch.Tensor):
        # t: [B] long; buf: [T]
        return buf.gather(0, t.clamp(min=0, max=self.timesteps - 1))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        """
        Sample x_t ~ q(x_t | x0).
        Returns (x_t, eps_used).
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_ab   = self._gather(self.sqrt_alpha_bars, t).view(-1, *([1] * (x0.dim()-1)))
        sqrt_1_ab = self._gather(self.sqrt_one_minus_alpha_bars, t).view(-1, *([1] * (x0.dim()-1)))
        x_t = sqrt_ab * x0 + sqrt_1_ab * noise
        return x_t, noise

    def v_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor):
        alpha = self._gather(self.sqrt_alpha_bars, t).view(-1, *([1] * (x_t.dim()-1)))
        sigma = self._gather(self.sqrt_one_minus_alpha_bars, t).view(-1, *([1] * (x_t.dim()-1)))
        # from ε = σ x_t + α v  ⇒  v = (ε − σ x_t) / α
        return (eps - sigma * x_t) / (alpha + 1e-12)

    def pred_eps_from_v(self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor):
        """
        Convert v-pred to eps using:
          x0 = α x_t - σ v
          ε  = σ x_t + α v
        where α = sqrt(alpha_bar_t), σ = sqrt(1 - alpha_bar_t).
        """
        alpha = self._gather(self.sqrt_alpha_bars, t).view(-1, *([1] * (x_t.dim()-1)))
        sigma = self._gather(self.sqrt_one_minus_alpha_bars, t).view(-1, *([1] * (x_t.dim()-1)))
        return sigma * x_t + alpha * v

    def pred_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor):
        alpha = self._gather(self.sqrt_alpha_bars, t).view(-1, *([1] * (x_t.dim()-1)))
        sigma = self._gather(self.sqrt_one_minus_alpha_bars, t).view(-1, *([1] * (x_t.dim()-1)))
        return (x_t - sigma * eps) / (alpha + 1e-12)

    @torch.no_grad()
    def ddim_sigma(self, t: torch.Tensor, t_prev: torch.Tensor, eta: float):
        """
        DDIM sigma between t and t_prev (per batch).
        """
        ab_t    = self._gather(self.alpha_bars, t)
        ab_prev = self._gather(self.alpha_bars, t_prev)
        sigma = eta * torch.sqrt((1.0 - ab_prev) / (1.0 - ab_t)) * torch.sqrt(1.0 - ab_t / (ab_prev + 1e-12))
        return sigma.view(-1, *([1] * 1))

    @torch.no_grad()
    def ddim_sample(self, x_t: torch.Tensor, t: torch.Tensor, t_prev: torch.Tensor,
                    eps_pred: torch.Tensor, eta: float = 0.0, noise: torch.Tensor = None):
        """
        One DDIM update given epsilon prediction.
        """
        if noise is None:
            noise = torch.randn_like(x_t)

        ab_t    = self._gather(self.alpha_bars, t).view(-1, *([1] * (x_t.dim()-1)))
        ab_prev = self._gather(self.alpha_bars, t_prev).view(-1, *([1] * (x_t.dim()-1)))

        x0_pred = self.pred_x0_from_eps(x_t, t, eps_pred)
        sigma   = self.ddim_sigma(t, t_prev, eta).view(-1, *([1] * (x_t.dim()-1)))

        dir_xt  = torch.sqrt((1.0 - ab_prev) - sigma**2).clamp(min=0) * eps_pred
        x_prev  = torch.sqrt(ab_prev) * x0_pred + dir_xt + sigma * noise
        return x_prev

