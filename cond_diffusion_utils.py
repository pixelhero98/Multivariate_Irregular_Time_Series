import torch
import torch.nn as nn


class NoiseScheduler(nn.Module):
    """
    Linear beta schedule + DDIM sampler.

    Notation:
      B = batch, T = total diffusion steps, D = feature dims (broadcasted)
    """
    def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 2e-2):
        super().__init__()
        self.timesteps = int(timesteps)

        # Register as buffers so they move with .to(device) and save in state_dict
        betas = torch.linspace(beta_start, beta_end, self.timesteps, dtype=torch.float32)      # [T]
        alphas = 1.0 - betas                                                                    # [T]
        alpha_bars = torch.cumprod(alphas, dim=0)                                               # [T]

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

    @torch.no_grad()
    def q_sample(self, x0: torch.Tensor, t: torch.LongTensor, noise: torch.Tensor | None = None):
        """
        Forward (noising) process.

        Args:
            x0:    [B, ...]
            t:     [B] integer timesteps in [0, T-1]
            noise: [B, ...] (optional)
        Returns:
            x_t:   [B, ...]
            eps:   [B, ...] (the actual noise added)
        """
        if noise is None:
            noise = torch.randn_like(x0)

        # gather ᾱ_t per batch element -> [B]
        # (buffers already on the right device)
        ab_t = self.alpha_bars[t]                                                                # [B]

        # reshape for broadcasting to x0
        view_shape = (-1, *([1] * (x0.dim() - 1)))                                              # [B, 1, ..., 1]
        sqrt_ab   = ab_t.sqrt().view(view_shape)                                                 # [B, 1, ..., 1]
        sqrt_1_ab = (1.0 - ab_t).sqrt().view(view_shape)                                         # [B, 1, ..., 1]

        x_t = sqrt_ab * x0 + sqrt_1_ab * noise
        return x_t, noise

    @torch.no_grad()
    def ddim_sample(
        self,
        xt: torch.Tensor,                 # [B, ...]
        t: torch.LongTensor,              # [B]
        t_prev: torch.LongTensor,         # [B] (can be -1 for "start")
        noise_pred: torch.Tensor,         # [B, ...]
        eta: float = 0.0
    ):
        """
        One DDIM step x_t -> x_{t-1}, elementwise-safe (handles mixed t/t_prev if needed).

        Returns:
            xt_prev: [B, ...]
        """
        B = xt.size(0)

        # ᾱ_t and ᾱ_{t-1}; handle t_prev == -1 per element by substituting 1.0
        ab_t = self.alpha_bars[t]                                                                 # [B]
        t_prev_clamped = torch.clamp(t_prev, min=0)                                               # [B]
        ab_t_prev = self.alpha_bars[t_prev_clamped]                                               # [B]
        prev_is_neg = (t_prev < 0)                                                                # [B]
        ab_t_prev = torch.where(prev_is_neg, torch.ones_like(ab_t_prev), ab_t_prev)               # [B]

        # reshape for broadcasting
        view_shape = (-1, *([1] * (xt.dim() - 1)))                                                # [B, 1, ..., 1]
        ab_t_b     = ab_t.view(view_shape)
        ab_tprev_b = ab_t_prev.view(view_shape)

        # 1) predict x0 from x_t and ε̂
        pred_x0 = (xt - (1.0 - ab_t_b).sqrt() * noise_pred) / ab_t_b.clamp(min=1e-12).sqrt()      # [B, ...]

        # 2) DDIM variance and direction
        #    sigma = η * sqrt( (1-ᾱ_{t-1})/(1-ᾱ_t) * (1 - ᾱ_t/ᾱ_{t-1}) )
        #    guard small negatives due to fp errors
        num   = (1.0 - ab_tprev_b).clamp(min=0.0)
        den   = (1.0 - ab_t_b).clamp(min=1e-12)
        frac  = (1.0 - (ab_t_b / ab_tprev_b).clamp(min=0.0, max=1.0)).clamp(min=0.0)
        sigma = eta * ( (num / den) * frac ).clamp(min=0.0).sqrt()                                 # [B, 1, ..., 1]

        direction = ( (1.0 - ab_tprev_b - sigma**2).clamp(min=0.0).sqrt() * noise_pred )          # [B, ...]

        # 3) noise (per-element: zero when t == 0)
        t_gt0 = (t > 0).view(view_shape)                                                          # [B, 1, ..., 1]
        eps = torch.randn_like(xt) * t_gt0                                                        # [B, ...] zeros where t==0

        # 4) final step
        xt_prev = ab_tprev_b.sqrt() * pred_x0 + direction + sigma * eps                           # [B, ...]
        return xt_prev
