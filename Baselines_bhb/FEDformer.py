import math

import torch
import torch.nn as nn

def _compl_mul1d_real(
    a_real: torch.Tensor,
    a_imag: torch.Tensor,
    b_real: torch.Tensor,
    b_imag: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Performs complex multiplication using real components only.

    This helper avoids relying on complex-valued kernels that are not available on
    the MPS backend by expanding the multiplication ``(a + ib) * (c + id)`` into
    real-valued einsums.  The arguments must therefore already be unpacked into
    their real and imaginary parts.
    """

    real = torch.einsum("bim, iom -> bom", a_real, b_real) - torch.einsum(
        "bim, iom -> bom", a_imag, b_imag
    )
    imag = torch.einsum("bim, iom -> bom", a_real, b_imag) + torch.einsum(
        "bim, iom -> bom", a_imag, b_real
    )
    return real, imag


class FourierBlock(nn.Module):
    """Applies learnable Fourier mixing on the first ``modes`` frequencies."""

    def __init__(self, in_channels: int, out_channels: int, seq_len: int, modes: int) -> None:
        super().__init__()
        self.modes = min(modes, seq_len // 2 + 1)
        self.scale = 1.0 / math.sqrt(in_channels * out_channels)
        self.weights = nn.Parameter(
            self.scale
            * torch.randn(in_channels, out_channels, self.modes, 2)
        )
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C]
        b, l, c = x.shape
        x_ft = torch.fft.rfft(x.permute(0, 2, 1), dim=-1)  # [B, C, L_ft]
        out_ft = torch.zeros(
            b,
            self.out_channels,
            x_ft.size(-1),
            2,
            dtype=x.real.dtype,
            device=x.device,
        )
        modes = min(self.modes, x_ft.size(-1))
        if modes > 0:
            x_modes = torch.view_as_real(x_ft[:, :, :modes])
            w = self.weights[:, :, :modes]
            real, imag = _compl_mul1d_real(
                x_modes[..., 0],
                x_modes[..., 1],
                w[..., 0],
                w[..., 1],
            )
            out_ft[:, :, :modes, 0] = real
            out_ft[:, :, :modes, 1] = imag
        x_out = torch.fft.irfft(torch.view_as_complex(out_ft), n=l, dim=-1)
        return x_out.permute(0, 2, 1)  # [B, L, out_channels]


class MovingAvg(nn.Module):
    """Moving average layer used for trend extraction."""

    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C]
        if self.kernel_size == 1:
            return x
        front = x[:, :1, :].repeat(1, self.padding, 1)
        end = x[:, -1:, :].repeat(1, self.padding, 1)
        x_pad = torch.cat([front, x, end], dim=1)
        x_pad = x_pad.transpose(1, 2)
        x_smooth = nn.functional.avg_pool1d(
            x_pad,
            kernel_size=self.kernel_size,
            stride=1,
        )
        return x_smooth.transpose(1, 2)


class SeriesDecomp(nn.Module):
    """Seasonal-trend decomposition using a moving average filter."""

    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class DataEmbedding(nn.Module):
    """Value + positional embedding used by the encoder and decoder."""

    def __init__(self, input_dim: int, d_model: int, max_len: int, dropout: float) -> None:
        super().__init__()
        self.value_embedding = nn.Linear(input_dim, d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape
        pos = self.position_embedding[:, :l, :]
        return self.dropout(self.value_embedding(x) + pos)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        seq_len: int,
        modes: int,
        moving_avg: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_fourier = FourierBlock(d_model, d_model, seq_len, modes)
        self.decomp = SeriesDecomp(moving_avg)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self Fourier mixing + residual
        x = x + self.dropout(self.self_fourier(self.norm1(x)))
        seasonal, trend = self.decomp(x)
        seasonal = seasonal + self.dropout(self.ff(self.norm2(seasonal)))
        return seasonal + trend


class CrossFourierBlock(nn.Module):
    """Cross Fourier mixing used inside the decoder."""

    def __init__(self, d_model: int, seq_len_q: int, seq_len_kv: int, modes: int) -> None:
        super().__init__()
        self.block = FourierBlock(d_model, d_model, seq_len_q + seq_len_kv, modes)
        self.seq_len_q = seq_len_q

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        cat = torch.cat([q, kv], dim=1)
        mixed = self.block(cat)
        return mixed[:, : self.seq_len_q, :]


class DecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        seq_len: int,
        memory_len: int,
        modes: int,
        moving_avg: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_fourier = FourierBlock(d_model, d_model, seq_len, modes)
        self.cross_fourier = CrossFourierBlock(d_model, seq_len, memory_len, modes)
        self.decomp = SeriesDecomp(moving_avg)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.self_fourier(self.norm1(x)))
        x = x + self.dropout(self.cross_fourier(self.norm2(x), memory))
        seasonal, trend = self.decomp(x)
        seasonal = seasonal + self.dropout(self.ff(self.norm3(seasonal)))
        return seasonal + trend


class FEDformer(nn.Module):
    """Frequency Enhanced Decomposed Transformer baseline."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        label_len: int,
        d_model: int = 64,
        d_ff: int = 128,
        dropout: float = 0.1,
        moving_avg: int = 25,
        modes: int = 16,
        enc_layers: int = 2,
        dec_layers: int = 1,
        c_in: int = 1,
        c_out: int = 1,
        max_len: int = 2048,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len

        self.enc_embedding = DataEmbedding(c_in, d_model, max_len, dropout)
        self.dec_embedding = DataEmbedding(c_in, d_model, max_len, dropout)

        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model=d_model,
                    d_ff=d_ff,
                    seq_len=seq_len,
                    modes=modes,
                    moving_avg=moving_avg,
                    dropout=dropout,
                )
                for _ in range(enc_layers)
            ]
        )

        decoder_seq_len = label_len + pred_len
        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(
                    d_model=d_model,
                    d_ff=d_ff,
                    seq_len=decoder_seq_len,
                    memory_len=seq_len,
                    modes=modes,
                    moving_avg=moving_avg,
                    dropout=dropout,
                )
                for _ in range(dec_layers)
            ]
        )

        self.projection = nn.Linear(d_model, c_out)
        self.decomp = SeriesDecomp(moving_avg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forecast ``pred_len`` steps given input history ``x``.

        Args:
            x: Tensor of shape ``(batch, seq_len, c_in)``.

        Returns:
            Tensor of shape ``(batch, pred_len, c_out)``.
        """

        b, l, _ = x.shape
        if l != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {l}")

        seasonal, trend = self.decomp(x)
        enc = self.enc_embedding(seasonal)
        for layer in self.encoder_layers:
            enc = layer(enc)

        # Prepare decoder input: last ``label_len`` observations + zeros for horizon
        label_len = min(self.label_len, self.seq_len)
        zeros = torch.zeros(b, self.pred_len, x.size(-1), device=x.device, dtype=x.dtype)
        dec_input = torch.cat([x[:, -label_len:, :], zeros], dim=1)
        seasonal_dec, trend_dec = self.decomp(dec_input)
        dec = self.dec_embedding(seasonal_dec)
        for layer in self.decoder_layers:
            dec = layer(dec, enc)

        seasonal_part = dec[:, -self.pred_len :, :]
        trend_part = trend_dec[:, -self.pred_len :, :]
        out = self.projection(seasonal_part) + trend_part
        return out

