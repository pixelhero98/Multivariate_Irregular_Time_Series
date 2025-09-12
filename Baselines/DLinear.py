# models/dlinear.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MovingAvg(nn.Module):
    """Moving average for trend extraction over time dimension (L).
    Expects x: [B, L, 1] or [B, L, C]; returns same shape."""
    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.kernel_size = int(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kernel_size <= 1:
            return x
        B, L, C = x.shape
        pad = (self.kernel_size - 1) // 2
        x_nc = x.permute(0, 2, 1)                          # [B,C,L]
        x_pad = F.pad(x_nc, (pad, pad), mode="replicate")    # pad time
        trend = F.avg_pool1d(x_pad, kernel_size=self.kernel_size, stride=1)
        return trend.permute(0, 2, 1)                      # [B,L,C]

class DLinear(nn.Module):
    """DLinear: decomposition (trend + seasonal) + two linear heads.

    Input  : x_hist [B, K, 1] (univariate target history)
    Output : y_hat  [B, H, 1] (univariate forecast)
    """
    def __init__(self, seq_len: int, pred_len: int, moving_avg: int = 25):
        super().__init__()
        self.K = int(seq_len)
        self.H = int(pred_len)
        self.mavg = MovingAvg(moving_avg)
        self.lin_seasonal = nn.Linear(self.K, self.H, bias=True)
        self.lin_trend    = nn.Linear(self.K, self.H, bias=True)
        nn.init.xavier_uniform_(self.lin_seasonal.weight); nn.init.zeros_(self.lin_seasonal.bias)
        nn.init.xavier_uniform_(self.lin_trend.weight);    nn.init.zeros_(self.lin_trend.bias)

    def forward(self, x_hist: torch.Tensor) -> torch.Tensor:
        """x_hist: [B,K,1] -> y_hat: [B,H,1]"""
        B, K, C = x_hist.shape
        assert K == self.K and C == 1, f"Expected [*,{self.K},1], got {x_hist.shape}"
        trend = self.mavg(x_hist)                          # [B,K,1]
        seasonal = x_hist - trend                          # [B,K,1]
        s = seasonal.squeeze(-1)                           # [B,K]
        t = trend.squeeze(-1)                              # [B,K]
        y = self.lin_seasonal(s) + self.lin_trend(t)       # [B,H]
        return y.unsqueeze(-1)                              # [B,H,1]
