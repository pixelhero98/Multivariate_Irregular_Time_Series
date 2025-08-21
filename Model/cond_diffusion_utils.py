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
        else:
            # cosine: derive betas from alpha_bar using finite differences
            ts = torch.linspace(0, 1, self.timesteps + 1, dtype=torch.float32)
            abar = _cosine_alpha_bar(ts)
            abar = abar / abar[0]  # ensure abar(0)=1
            # define alphas so that cumprod(alphas) = abar[:-1], i.e., ᾱ[0] = 1
            alphas = torch.ones(self.timesteps, dtype=torch.float32)
            alphas[1:] = abar[1:self.timesteps] / abar[0:self.timesteps - 1]
            betas = (1.0 - alphas).clone()
            betas[1:] = betas[1:].clamp(min=1e-8, max=0.999)
            betas[0] = 0.0  # exact no-noise at t=0
        # Buffers
        self.register_buffer("betas", betas)                         # [T]
        self.register_buffer("alphas", 1.0 - betas)                  # [T]
        self.register_buffer("alpha_bars", torch.cumprod(1.0 - betas, dim=0))          # [T]
        self.register_buffer("sqrt_alphas", torch.sqrt(1.0 - betas))                   # [T]
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(self.alpha_bars))           # [T]
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - self.alpha_bars))  # [T]

    @torch.no_grad()
    def timesteps_desc(self):
        return torch.arange(self.timesteps - 1, -1, -1, dtype=torch.long, device=self.alpha_bars.device)

    def _gather(self, buf: torch.Tensor, t: torch.Tensor):
        # t: [B] (any dtype/device) ; buf: [T]
        t = t.clamp(min=0, max=self.timesteps - 1).to(device=buf.device, dtype=torch.long)

        return buf.gather(0, t)

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

    # ---------- conversions among parameterizations ----------

    def pred_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor):
        alpha = self._gather(self.sqrt_alpha_bars, t).view(-1, *([1] * (x_t.dim()-1)))
        sigma = self._gather(self.sqrt_one_minus_alpha_bars, t).view(-1, *([1] * (x_t.dim()-1)))
        return (x_t - sigma * eps) / (alpha + 1e-12)

    def pred_eps_from_x0(self, x_t: torch.Tensor, t: torch.Tensor, x0: torch.Tensor):
        alpha = self._gather(self.sqrt_alpha_bars, t).view(-1, *([1] * (x_t.dim()-1)))
        sigma = self._gather(self.sqrt_one_minus_alpha_bars, t).view(-1, *([1] * (x_t.dim()-1)))
        return (x_t - alpha * x0) / (sigma + 1e-12)

    def pred_x0_from_v(self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor):
        """
        v-param: v = α ε − σ x0  ⇒  x0 = α x_t − σ v
        """
        alpha = self._gather(self.sqrt_alpha_bars, t).view(-1, *([1] * (x_t.dim()-1)))
        sigma = self._gather(self.sqrt_one_minus_alpha_bars, t).view(-1, *([1] * (x_t.dim()-1)))
        return alpha * x_t - sigma * v

    def pred_eps_from_v(self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor):
        """
        From v-param: ε = σ x_t + α v
        """
        alpha = self._gather(self.sqrt_alpha_bars, t).view(-1, *([1] * (x_t.dim()-1)))
        sigma = self._gather(self.sqrt_one_minus_alpha_bars, t).view(-1, *([1] * (x_t.dim()-1)))
        return sigma * x_t + alpha * v

    def v_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor):
        """
        Invert: v = (ε − σ x_t)/α
        """
        alpha = self._gather(self.sqrt_alpha_bars, t).view(-1, *([1] * (x_t.dim()-1)))
        sigma = self._gather(self.sqrt_one_minus_alpha_bars, t).view(-1, *([1] * (x_t.dim()-1)))
        return (eps - sigma * x_t) / (alpha + 1e-12)

    # unified helpers
    def to_x0(self, x_t: torch.Tensor, t: torch.Tensor, pred: torch.Tensor, param_type: str):
        if param_type == "eps":
            return self.pred_x0_from_eps(x_t, t, pred)
        elif param_type == "v":
            return self.pred_x0_from_v(x_t, t, pred)
        elif param_type == "x0":
            return pred
        else:
            raise ValueError("param_type must be 'eps', 'v', or 'x0'")

    def to_eps(self, x_t: torch.Tensor, t: torch.Tensor, pred: torch.Tensor, param_type: str):
        if param_type == "eps":
            return pred
        elif param_type == "v":
            return self.pred_eps_from_v(x_t, t, pred)
        elif param_type == "x0":
            return self.pred_eps_from_x0(x_t, t, pred)
        else:
            raise ValueError("param_type must be 'eps', 'v', or 'x0'")

    # ---------- DDIM stepping (native-param aware) ----------

    @torch.no_grad()
    def ddim_sigma(self, t: torch.Tensor, t_prev: torch.Tensor, eta: float):
        ab_t    = self._gather(self.alpha_bars, t)
        ab_prev = self._gather(self.alpha_bars, t_prev)
        sigma = eta * torch.sqrt((1.0 - ab_prev) / (1.0 - ab_t)) * torch.sqrt(1.0 - ab_t / (ab_prev + 1e-12))
        return sigma.view(-1)

    @torch.no_grad()
    def ddim_step_from(self, x_t: torch.Tensor, t: torch.Tensor, t_prev: torch.Tensor,
                       pred: torch.Tensor, param_type: str, eta: float = 0.0, noise: torch.Tensor = None):
        """
        DDIM update using a prediction in the given parameterization.
        Works for 'eps', 'v', or 'x0' (x0 rarely used directly).
        """
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

