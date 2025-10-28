import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ScoreGradConfig:
    seq_len: int
    pred_len: int
    hidden_size: int = 64
    encoder_layers: int = 1
    dropout: float = 0.1
    cond_hidden: int = 128
    score_hidden: int = 128
    score_layers: int = 2
    time_embed_dim: int = 64
    sde_steps: int = 20
    beta_min: float = 0.1
    beta_max: float = 20.0
    deterministic_eval: bool = True


class SinusoidalTimeEmbedding(nn.Module):
    """Classic sinusoidal embedding for continuous-time steps."""

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

    def __init__(self, config: ScoreGradConfig):
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
        return h[-1]


class ScoreNetwork(nn.Module):
    """Predict the score (grad log-density) conditioned on context."""

    def __init__(self, config: ScoreGradConfig):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(config.time_embed_dim)
        self.cond_proj = nn.Sequential(
            nn.Linear(config.hidden_size + config.time_embed_dim, config.cond_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.in_proj = nn.Linear(config.cond_hidden + 1, config.score_hidden)
        self.hidden_layers = nn.ModuleList(
            nn.Linear(config.score_hidden, config.score_hidden)
            for _ in range(max(0, config.score_layers - 1))
        )
        self.dropout = nn.Dropout(config.dropout)
        self.out_proj = nn.Linear(config.score_hidden, 1)

    def forward(self, sample: Tensor, context: Tensor, time: Tensor) -> Tensor:
        if sample.ndim != 2:
            raise ValueError("Sample must be a 2D tensor [B, pred_len]")
        if context.ndim != 2:
            raise ValueError("Context must be a 2D tensor [B, hidden]")
        if time.ndim != 1:
            raise ValueError("time must be a 1D tensor [B]")

        t_emb = self.time_embed(time)
        cond = torch.cat([context, t_emb], dim=-1)
        cond = self.cond_proj(cond)
        cond_exp = cond.unsqueeze(1).expand(-1, sample.size(1), -1)
        x = torch.cat([sample.unsqueeze(-1), cond_exp], dim=-1)
        x = F.gelu(self.in_proj(x))
        for layer in self.hidden_layers:
            x = F.gelu(layer(self.dropout(x)))
        score = self.out_proj(x).squeeze(-1)
        return score


class ScoreGradLight(nn.Module):
    """Continuous-time score-based diffusion forecaster."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        hidden_size: int = 64,
        encoder_layers: int = 1,
        dropout: float = 0.1,
        cond_hidden: int = 128,
        score_hidden: int = 128,
        score_layers: int = 2,
        time_embed_dim: int = 64,
        sde_steps: int = 20,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        deterministic_eval: bool = True,
    ):
        super().__init__()
        if sde_steps <= 0:
            raise ValueError("sde_steps must be positive")
        if beta_min <= 0 or beta_max <= 0:
            raise ValueError("beta_min and beta_max must be positive")
        if beta_min >= beta_max:
            raise ValueError("beta_min must be smaller than beta_max")

        self.config = ScoreGradConfig(
            seq_len=seq_len,
            pred_len=pred_len,
            hidden_size=hidden_size,
            encoder_layers=encoder_layers,
            dropout=dropout,
            cond_hidden=cond_hidden,
            score_hidden=score_hidden,
            score_layers=score_layers,
            time_embed_dim=time_embed_dim,
            sde_steps=sde_steps,
            beta_min=beta_min,
            beta_max=beta_max,
            deterministic_eval=deterministic_eval,
        )

        self.context_encoder = ContextEncoder(self.config)
        self.score_network = ScoreNetwork(self.config)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _beta(self, t: Tensor) -> Tensor:
        beta_min = self.config.beta_min
        beta_max = self.config.beta_max
        return beta_min + (beta_max - beta_min) * t

    def _initial_latent(self, batch_size: int, pred_len: int) -> Tensor:
        latents = torch.randn(batch_size, pred_len, device=self.device)
        if not self.training and self.config.deterministic_eval:
            latents.zero_()
        return latents

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.size(-1) != 1:
            raise ValueError("ScoreGradLight expects input of shape [B, seq_len, 1]")

        context = self.context_encoder(x)
        B = x.size(0)
        sample = self._initial_latent(B, self.config.pred_len)

        steps = self.config.sde_steps
        dt = 1.0 / steps

        for idx in range(steps, 0, -1):
            t = torch.full((B,), idx / steps, device=x.device, dtype=torch.float32)
            beta_t = self._beta(t)
            score = self.score_network(sample, context, t)

            drift = -0.5 * beta_t.unsqueeze(1) * sample
            diffusion = torch.sqrt(beta_t + 1e-8).unsqueeze(1)
            sample = sample + (drift - beta_t.unsqueeze(1) * score) * dt

            if idx > 1:
                if self.training:
                    noise = torch.randn_like(sample)
                elif self.config.deterministic_eval:
                    noise = torch.zeros_like(sample)
                else:
                    noise = torch.randn_like(sample)
                sample = sample + diffusion * math.sqrt(dt) * noise

        return sample.unsqueeze(-1)