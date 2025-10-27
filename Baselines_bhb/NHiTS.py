import math
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, n_layers: int, activation=nn.ReLU):
        super().__init__()
        layers = []
        if n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if n_layers == 1:
            layers.append(nn.Linear(in_dim, hidden_dim))
        else:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activation())
            for _ in range(n_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(activation())
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.net = nn.Sequential(*layers)
        self.activation = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return self.activation(out)


class NHiTSBlock(nn.Module):
    def __init__(
        self,
        input_size: int,
        forecast_size: int,
        pooling: int,
        hidden_dim: int,
        n_layers: int,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.forecast_size = forecast_size
        self.pooling = max(1, pooling)
        self.downsample_size = math.ceil(input_size / self.pooling)

        self.mlp = _MLP(self.downsample_size, hidden_dim, n_layers)
        self.backcast_head = nn.Linear(hidden_dim, input_size)
        self.forecast_head = nn.Linear(hidden_dim, forecast_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, K]
        if self.pooling > 1:
            # ``adaptive_avg_pool1d`` on MPS requires ``input_size`` to be divisible by the
            # requested ``output_size``. When the ratio is not integral we pad the tail by
            # repeating the last sample so the requirement holds, then apply the average pool.
            x_1d = x.unsqueeze(1)
            remainder = x_1d.size(-1) % self.downsample_size
            if remainder:
                pad = self.downsample_size - remainder
                x_1d = F.pad(x_1d, (0, pad), mode="replicate")
            x_ds = F.adaptive_avg_pool1d(x_1d, self.downsample_size).squeeze(1)
        else:
            x_ds = x
        features = self.mlp(x_ds)
        backcast = self.backcast_head(features)
        forecast = self.forecast_head(features)
        return backcast, forecast


def _ensure_seq(obj: Iterable[int] | int, length: int) -> Sequence[int]:
    if isinstance(obj, Iterable) and not isinstance(obj, int):
        seq = list(obj)
        if len(seq) == 0:
            raise ValueError("pooling kernels cannot be empty")
        if len(seq) >= length:
            return seq
        return seq + [seq[-1]] * (length - len(seq))
    return [int(obj)] * length


class NHiTS(nn.Module):
    """Simplified N-HiTS forecaster for univariate inputs."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        n_stacks: int = 2,
        n_blocks: int = 3,
        n_layers: int = 2,
        hidden_size: int = 128,
        pooling_kernel_sizes: Iterable[int] | int = (8, 4, 1),
    ) -> None:
        super().__init__()
        if seq_len <= 0 or pred_len <= 0:
            raise ValueError("seq_len and pred_len must be positive")
        if n_stacks <= 0 or n_blocks <= 0:
            raise ValueError("n_stacks and n_blocks must be positive")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_stacks = n_stacks
        self.n_blocks = n_blocks

        pooling_seq = _ensure_seq(pooling_kernel_sizes, n_blocks)
        blocks = []
        for _ in range(n_stacks):
            for k in range(n_blocks):
                blocks.append(
                    NHiTSBlock(
                        input_size=seq_len,
                        forecast_size=pred_len,
                        pooling=pooling_seq[k],
                        hidden_dim=hidden_size,
                        n_layers=n_layers,
                    )
                )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, K, 1]
        x = x.squeeze(-1)
        residual = x
        forecast = torch.zeros(x.size(0), self.pred_len, device=x.device, dtype=x.dtype)
        for block in self.blocks:
            backcast, block_forecast = block(residual)
            residual = residual - backcast
            forecast = forecast + block_forecast
        return forecast.unsqueeze(-1)