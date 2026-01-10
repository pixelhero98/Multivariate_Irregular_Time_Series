import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TVHead(nn.Module):
    """Single-hidden-layer MLP that projects features to scalar signals."""
    def __init__(self, feat_dim: int, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return scalar activations with shape [..., 1] squeezed to [...]."""
        return self.net(x).squeeze(-1)


class PanelHistoryAE(nn.Module):
    """Panel auto-encoder with soft patching and additive position embeddings."""

    def __init__(
        self,
        num_entities: int,
        feat_dim: int,
        window_size: int,
        *,
        mix_dim: int = 64,
        tv_hidden: int = 32,
        out_len: int = 16,
        context_dim: int = 256,
        enc_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
        patch_kernel: int = 3,
    ) -> None:
        super().__init__()
        
        # Validation for Soft Patching
        if patch_kernel % 2 == 0:
            raise ValueError(f"patch_kernel must be odd to maintain sequence length, got {patch_kernel}")

        self.N = num_entities
        self.D = feat_dim
        self.window_size = window_size
        self.Hc = context_dim
        self.S = out_len
        self.mix_dim = mix_dim

        # 1. Soft Patching Input Mixer (Conv1d)
        # Padding ensures output length equals input length K
        padding = (patch_kernel - 1) // 2
        
        self.input_mixer = nn.Sequential(
            # Input: [Batch, Channels=D, Length=K]
            nn.Conv1d(
                in_channels=self.D, 
                out_channels=self.mix_dim, 
                kernel_size=patch_kernel, 
                padding=padding,
                stride=1
            ),
        )
        self.mixer_norm = nn.LayerNorm(self.mix_dim)
        self.mixer_act = nn.GELU()

        # 2. Positional Embeddings Setup
        self.encoder_dim = self.mix_dim + 2 # +2 for v_sig and t_sig
        
        # Initial pos embedding (will be resized if needed for heads)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.window_size, self.encoder_dim) * 0.02)

        self.v_head = TVHead(self.D, tv_hidden)
        self.t_head = TVHead(self.D, tv_hidden)

        # 3. Temporal Encoder & Head Alignment
        # Ensure encoder_dim is divisible by n_heads
        if self.encoder_dim % n_heads != 0:
            new_dim = ((self.encoder_dim // n_heads) + 1) * n_heads
            self.input_pad = nn.Linear(self.encoder_dim, new_dim)
            self.encoder_dim = new_dim
            # Re-initialize pos embedding with corrected dimension
            self.pos_embedding = nn.Parameter(torch.randn(1, self.window_size, new_dim) * 0.02)
        else:
            self.input_pad = nn.Identity()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.encoder_dim, 
            nhead=n_heads, 
            dim_feedforward=self.encoder_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.history_encoder = nn.TransformerEncoder(encoder_layer, num_layers=enc_layers)

        # 4. Context Projection & Pooling
        self.token_proj = nn.Sequential(
            nn.Linear(self.encoder_dim, self.Hc), 
            nn.GELU(), 
            nn.Dropout(dropout)
        )
        
        # Learnable Queries for pooling
        self.queries = nn.Parameter(torch.randn(self.S, self.Hc) / math.sqrt(self.Hc))
        self.norm = nn.LayerNorm(self.Hc)

        # 5. Decoders
        self.decoder_net = nn.Sequential(
            nn.Linear(self.Hc, self.Hc * 2),
            nn.GELU(),
            nn.Linear(self.Hc * 2, self.window_size * self.N * self.D)
        )
        self.v_decoder = nn.Linear(self.Hc, self.window_size * self.N)
        self.t_decoder = nn.Linear(self.Hc, self.window_size * self.N)

    @staticmethod
    def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask_bn: torch.Tensor) -> torch.Tensor:
        """
        Calculates MSE loss only on valid entities.
        pred/target: [B, K, N, D] or [B, K, N]
        mask_bn: [B, N] boolean mask
        """
        # Align mask dimensions for broadcasting
        if pred.ndim == 4: # [B, K, N, D]
            # mask: [B, N] -> [B, 1, N, 1]
            m = mask_bn.to(dtype=pred.dtype).unsqueeze(1).unsqueeze(-1)
            # Count valid elements: sum(mask) * K * D
            denom = m.sum() * pred.size(1) * pred.size(3)
        else: # [B, K, N]
            # mask: [B, N] -> [B, 1, N]
            m = mask_bn.to(dtype=pred.dtype).unsqueeze(1)
            denom = m.sum() * pred.size(1)

        se = (pred - target).pow(2) * m
        return se.sum() / denom.clamp_min(1.0)

    def forward(
        self,
        x: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None, # Unused
        dt: Optional[torch.Tensor] = None,       # Unused
        ctx_diff: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        
        if ctx_diff is None: raise ValueError("ctx_diff required")
        B, K, N, D = x.shape
        assert N == self.N and D == self.D, f"Shape mismatch. Expected (..., {self.N}, {self.D})"

        # ---------------------------------------------------------
        # 1. Soft Patching (Conv1d)
        # ---------------------------------------------------------
        # CRITICAL FIX: Permute before reshaping to preserve data layout
        # [B, K, N, D] -> [B, N, K, D] -> [B*N, K, D]
        x_flat_in = x.permute(0, 2, 1, 3).reshape(B * N, K, D)
        
        # Conv1d expects [Batch, Channels, Length] -> [B*N, D, K]
        x_perm = x_flat_in.permute(0, 2, 1)
        
        # Apply Conv: [B*N, mix_dim, K]
        x_conv = self.input_mixer(x_perm)
        
        # Permute back: [B*N, K, mix_dim]
        x_mixed = x_conv.permute(0, 2, 1)
        
        # Apply Norm and Activation
        x_mixed = self.mixer_act(self.mixer_norm(x_mixed))

        # ---------------------------------------------------------
        # 2. Extract and Fuse Signals
        # ---------------------------------------------------------
        v_sig = self.v_head(x)        # [B, K, N]
        t_sig = self.t_head(ctx_diff) # [B, K, N]

        # Flatten N into B: [B, K, N] -> [B, N, K] -> [B*N, K, 1]
        v_flat = v_sig.permute(0, 2, 1).reshape(B * N, K, 1)
        t_flat = t_sig.permute(0, 2, 1).reshape(B * N, K, 1)

        # Concatenate: [B*N, K, mix_dim + 2]
        fused = torch.cat([x_mixed, v_flat, t_flat], dim=-1)
        
        # Pad features if needed for MultiHeadAttention
        fused = self.input_pad(fused)

        # ---------------------------------------------------------
        # 3. Temporal Encoding
        # ---------------------------------------------------------
        # Add Positional Embeddings (Broadcasting [1, K, Dim])
        fused = fused + self.pos_embedding[:, :K, :]

        # Transformer Encoder
        encoded_hist = self.history_encoder(fused) # [B*N, K, Dim]

        # ---------------------------------------------------------
        # 4. Aggregation & Token Projection
        # ---------------------------------------------------------
        # Reshape to separate Batch and Entities: [B, N, K, Dim]
        encoded_bn = encoded_hist.view(B, N, K, -1)
        
        # Mean Pooling over Entities -> [B, K, Dim]
        encoded_mean = encoded_bn.mean(dim=1) 

        # Project to context dimension
        tokens = self.token_proj(encoded_mean) # [B, K, Hc]
        
        # ---------------------------------------------------------
        # 5. Query Pooling (Cross-Attention)
        # ---------------------------------------------------------
        norm_tokens = self.norm(tokens)
        norm_queries = F.layer_norm(self.queries, (self.Hc,))
        
        # Attention scores: [B, K, S]
        # (B, K, Hc) @ (S, Hc)^T -> (B, K, S)
        att = torch.matmul(norm_tokens, norm_queries.t()) / math.sqrt(self.Hc)
        att = att.softmax(dim=1) # Softmax over Time (K)
        
        # Context: [B, S, Hc]
        # (B, S, K) @ (B, K, Hc) -> (B, S, Hc)
        context = torch.matmul(att.transpose(1, 2), tokens)

        # ---------------------------------------------------------
        # 6. Reconstruction
        # ---------------------------------------------------------
        # Pool context for global reconstruction
        context_pooled = context.mean(dim=1) # [B, Hc]
        
        # Reconstruct x: [B, K, N, D]
        x_hat = self.decoder_net(context_pooled).view(B, K, N, D)
        
        # Reconstruct signals: [B, K, N]
        v_hat = self.v_decoder(context_pooled).view(B, K, N)
        t_hat = self.t_decoder(context_pooled).view(B, K, N)

        aux = {
            'x': x, 'x_hat': x_hat,
            'v_sig': v_sig, 'v_hat': v_hat,
            't_sig': t_sig, 't_hat': t_hat
        }
        
        return context, aux

    def recon_loss(
        self,
        aux: Dict[str, torch.Tensor],
        entity_mask: torch.Tensor,
        weights: Tuple[float, float, float] = (1.0, 0.1, 0.1),
    ) -> torch.Tensor:
        """
        Computes weighted reconstruction loss.
        weights: (w_x, w_v, w_t)
        """
        w_x, w_v, w_t = weights
        loss_x = self._masked_mse(aux['x_hat'], aux['x'], entity_mask)
        loss_v = self._masked_mse(aux['v_hat'], aux['v_sig'], entity_mask)
        loss_t = self._masked_mse(aux['t_hat'], aux['t_sig'], entity_mask)

        return (w_x * loss_x) + (w_v * loss_v) + (w_t * loss_t)


class LaplaceAE(PanelHistoryAE):
    """Compatibility wrapper that exposes the historical summarizer as ``LaplaceAE``.

    Older training and evaluation scripts expect a ``LaplaceAE`` class with a
    ``lap_k`` constructor argument. The underlying implementation is provided by
    :class:`PanelHistoryAE`, where ``lap_k`` corresponds to the mixer output
    dimension (``mix_dim``).
    """

    def __init__(
        self,
        num_entities: int,
        feat_dim: int,
        window_size: int,
        *,
        lap_k: int = 64,
        tv_hidden: int = 32,
        out_len: int = 16,
        context_dim: int = 256,
        dropout: float = 0.1,
        enc_layers: int = 2,
        n_heads: int = 4,
        patch_kernel: int = 3,
    ) -> None:
        super().__init__(
            num_entities=num_entities,
            feat_dim=feat_dim,
            window_size=window_size,
            mix_dim=lap_k,
            tv_hidden=tv_hidden,
            out_len=out_len,
            context_dim=context_dim,
            enc_layers=enc_layers,
            n_heads=n_heads,
            dropout=dropout,
            patch_kernel=patch_kernel,
        )
