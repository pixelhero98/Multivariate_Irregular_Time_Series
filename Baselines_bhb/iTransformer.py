from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ITransformerConfig:
    seq_len: int
    pred_len: int
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 128
    dropout: float = 0.1
    head_hidden: int = 128


class InstanceEmbedding(nn.Module):
    """Project normalized scalar histories into a model dimension."""

    def __init__(self, config: ITransformerConfig):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(1, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.positional_encoding = nn.Parameter(
            torch.randn(1, config.seq_len, config.d_model)
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.size(-1) != 1:
            raise ValueError("InstanceEmbedding expects [B, seq_len, 1] input")
        tokens = self.proj(x)
        return tokens + self.positional_encoding


class ForecastHead(nn.Module):
    def __init__(self, config: ITransformerConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.head_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.head_hidden, config.pred_len),
        )

    def forward(self, pooled: Tensor) -> Tensor:
        return self.net(pooled)


class ITransformer(nn.Module):
    """Small iTransformer-style forecaster with instance normalization."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.1,
        head_hidden: int = 128,
    ):
        super().__init__()
        self.config = ITransformerConfig(
            seq_len=seq_len,
            pred_len=pred_len,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            head_hidden=head_hidden,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.embedding = InstanceEmbedding(self.config)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = ForecastHead(self.config)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.size(-1) != 1:
            raise ValueError("ITransformer expects input of shape [B, seq_len, 1]")

        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-5)
        x_norm = (x - mean) / std

        tokens = self.embedding(x_norm)
        encoded = self.encoder(tokens)
        pooled = encoded.mean(dim=1)
        pred_norm = self.head(pooled)
        forecast = pred_norm * std.squeeze(-1) + mean.squeeze(-1)
        return forecast.unsqueeze(-1)
