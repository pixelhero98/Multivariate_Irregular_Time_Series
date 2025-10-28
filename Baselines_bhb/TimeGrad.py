import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class TimeGradConfig:
    seq_len: int
    pred_len: int
    hidden_size: int = 64
    encoder_layers: int = 1
    diffusion_steps: int = 16
    cond_hidden: int = 128
    noise_hidden: int = 128
    noise_layers: int = 2
    dropout: float = 0.1
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    deterministic_eval: bool = True


class SinusoidalTimeEmbedding(nn.Module):
    """Classic sinusoidal embedding for discrete diffusion steps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        if t.ndim != 1:
            raise ValueError("Time embedding expects a 1D tensor of steps")
        half_dim = self.dim // 2
        device = t.device
        if half_dim == 0:
            return torch.zeros(t.size(0), 0, device=device)
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half_dim, device=device, dtype=torch.float32)
            / max(half_dim - 1, 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ContextEncoder(nn.Module):
    """Encode historical values with a lightweight GRU."""

    def __init__(self, config: TimeGradConfig):
        super().__init__()
        self.gru = nn.GRU(
            input_size=1,
            hidden_size=config.hidden_size,
            num_layers=config.encoder_layers,
            batch_first=True,
            dropout=config.dropout if config.encoder_layers > 1 else 0.0,
        )

    def forward(self, x: Tensor) -> Tensor:
        output, h = self.gru(x)
        # Take the final layer's hidden state as the context representation.
        return h[-1]


class Conditioner(nn.Module):
    """Fuse context and diffusion step information into a conditioning vector."""

    def __init__(self, config: TimeGradConfig):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(config.hidden_size)
        self.net = nn.Sequential(
            nn.Linear(config.hidden_size + config.hidden_size, config.cond_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.cond_hidden, config.cond_hidden),
            nn.GELU(),
        )

    def forward(self, context: Tensor, step: Tensor) -> Tensor:
        t_emb = self.time_embed(step)
        cond = torch.cat([context, t_emb], dim=-1)
        return self.net(cond)


class NoisePredictor(nn.Module):
    """Predict the diffusion noise for each horizon step."""

    def __init__(self, config: TimeGradConfig):
        super().__init__()
        input_dim = config.cond_hidden + 1
        self.in_proj = nn.Linear(input_dim, config.noise_hidden)
        self.layers = nn.ModuleList(
            nn.Linear(config.noise_hidden, config.noise_hidden)
            for _ in range(max(0, config.noise_layers - 1))
        )
        self.dropout = nn.Dropout(config.dropout)
        self.out_proj = nn.Linear(config.noise_hidden, 1)

    def forward(self, sample: Tensor, cond: Tensor) -> Tensor:
        if sample.ndim != 2:
            raise ValueError("Sample must be a 2D tensor [B, pred_len]")
        cond_exp = cond.unsqueeze(1).expand(-1, sample.size(1), -1)
        x = torch.cat([sample.unsqueeze(-1), cond_exp], dim=-1)
        x = F.gelu(self.in_proj(x))
        for layer in self.layers:
            x = F.gelu(layer(self.dropout(x)))
        eps = self.out_proj(x).squeeze(-1)
        return eps


class TimeGradLight(nn.Module):
    """Lightweight TimeGrad-style diffusion forecaster."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        hidden_size: int = 64,
        encoder_layers: int = 1,
        diffusion_steps: int = 16,
        cond_hidden: int = 128,
        noise_hidden: int = 128,
        noise_layers: int = 2,
        dropout: float = 0.1,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        deterministic_eval: bool = True,
    ):
        super().__init__()
        if diffusion_steps <= 0:
            raise ValueError("diffusion_steps must be positive")
        self.config = TimeGradConfig(
            seq_len=seq_len,
            pred_len=pred_len,
            hidden_size=hidden_size,
            encoder_layers=encoder_layers,
            diffusion_steps=diffusion_steps,
            cond_hidden=cond_hidden,
            noise_hidden=noise_hidden,
            noise_layers=noise_layers,
            dropout=dropout,
            beta_start=beta_start,
            beta_end=beta_end,
            deterministic_eval=deterministic_eval,
        )

        self.context_encoder = ContextEncoder(self.config)
        self.conditioner = Conditioner(self.config)
        self.noise_predictor = NoisePredictor(self.config)

        betas = torch.linspace(beta_start, beta_end, diffusion_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas", torch.sqrt(alphas))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod + 1e-8),
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _initial_latent(self, batch_size: int, pred_len: int) -> Tensor:
        latents = torch.randn(batch_size, pred_len, device=self.device)
        if not self.training and self.config.deterministic_eval:
            latents.zero_()
        return latents

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.size(-1) != 1:
            raise ValueError("TimeGradLight expects input of shape [B, seq_len, 1]")

        context = self.context_encoder(x)
        B = x.size(0)
        pred_len = self.config.pred_len
        sample = self._initial_latent(B, pred_len)

        for step in reversed(range(self.config.diffusion_steps)):
            step_tensor = torch.full((B,), step, device=x.device, dtype=torch.long)
            cond = self.conditioner(context, step_tensor)
            eps = self.noise_predictor(sample, cond)

            beta_t = self.betas[step]
            alpha_t = self.alphas[step]
            alpha_bar_t = self.alphas_cumprod[step]

            coef = beta_t / torch.sqrt(1.0 - alpha_bar_t + 1e-8)
            sample = (sample - coef * eps) / torch.sqrt(alpha_t + 1e-8)

            if step > 0:
                if self.training:
                    noise = torch.randn_like(sample)
                elif self.config.deterministic_eval:
                    noise = torch.zeros_like(sample)
                else:
                    noise = torch.randn_like(sample)
                sigma_t = torch.sqrt(beta_t)
                sample = sample + sigma_t * noise

        return sample.unsqueeze(-1)