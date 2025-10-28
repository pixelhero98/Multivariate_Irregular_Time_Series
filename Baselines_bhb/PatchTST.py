from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class PatchConfig:
    seq_len: int
    pred_len: int
    patch_len: int = 16
    stride: int = 8
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 128
    dropout: float = 0.1
    head_hidden: int = 128

    def effective_length(self) -> int:
        """Return total length after right-side padding."""
        pad = compute_required_padding(self.seq_len, self.patch_len, self.stride)
        return self.seq_len + pad

    def num_patches(self) -> int:
        L = self.effective_length()
        return 1 + (L - self.patch_len) // self.stride


def compute_required_padding(seq_len: int, patch_len: int, stride: int) -> int:
    """Minimal right padding so unfolding uses integral strides."""
    if seq_len <= patch_len:
        return patch_len - seq_len

    remainder = (seq_len - patch_len) % stride
    if remainder == 0:
        return 0
    return stride - remainder


class PatchEmbedding(nn.Module):
    """Turn overlapping patches into token embeddings."""

    def __init__(self, config: PatchConfig):
        super().__init__()
        self.config = config
        self.patch_len = config.patch_len
        self.stride = config.stride
        self.pad = compute_required_padding(config.seq_len, config.patch_len, config.stride)
        self.embed = nn.Sequential(
            nn.Linear(config.patch_len, config.d_model),
            nn.GELU(),
        )
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "positional_encoding",
            torch.randn(1, config.num_patches(), config.d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: [B, seq_len, 1] -> tokens: [B, num_patches, d_model]."""
        if x.ndim != 3 or x.size(-1) != 1:
            raise ValueError("PatchEmbedding expects input of shape [B, seq_len, 1]")

        # Prepare for unfold: [B, 1, L]
        x_seq = x.transpose(1, 2)
        if self.pad > 0:
            # Replicate final value to avoid introducing zeros at the boundary.
            pad_val = x_seq[..., -1:].expand(-1, -1, self.pad)
            x_seq = torch.cat([x_seq, pad_val], dim=-1)

        patches = x_seq.unfold(dimension=2, size=self.patch_len, step=self.stride)
        patches = patches.squeeze(1)  # [B, num_patches, patch_len]
        tokens = self.embed(patches)
        tokens = tokens + self.positional_encoding
        return self.dropout(tokens)


class FeedForwardHead(nn.Module):
    def __init__(self, config: PatchConfig):
        super().__init__()
        num_patches = config.num_patches()
        self.net = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Flatten(start_dim=1),
            nn.Linear(num_patches * config.d_model, config.head_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.head_hidden, config.pred_len),
        )

    def forward(self, encoded_tokens: Tensor) -> Tensor:
        return self.net(encoded_tokens)


class PatchTST(nn.Module):
    """Small patch-based Transformer for univariate forecasting."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.1,
        head_hidden: int = 128,
    ):
        super().__init__()
        self.config = PatchConfig(
            seq_len=seq_len,
            pred_len=pred_len,
            patch_len=patch_len,
            stride=stride,
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
        self.patch_embedding = PatchEmbedding(self.config)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = FeedForwardHead(self.config)

    def forward(self, x: Tensor) -> Tensor:
        tokens = self.patch_embedding(x)
        encoded = self.encoder(tokens)
        forecast = self.head(encoded)
        return forecast.unsqueeze(-1)