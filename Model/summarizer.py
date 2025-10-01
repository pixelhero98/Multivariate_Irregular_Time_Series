"""Lightweight Laplace auto-encoder used to summarise panel time-series data.

The original repository contained a minimally documented implementation.  This
rewrite keeps the public API intact while clarifying tensor shapes, tightening
validation and explicitly handling padded entities.  The module exposes the
`LaplaceAE` class which turns raw panel features into a compact context token
sequence alongside reconstruction auxiliaries that are consumed during
training.
"""
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from Model.laptrans import LearnableLaplaceBasis, LearnablepesudoInverse  # note: class name spelled as in repo


__all__ = ["LaplaceAE"]



class TVHead(nn.Module):
   """Project per-entity features to a single scalar signal.

    This head is used for both value (``V``) and time-difference (``T``)
    streams.  The sequential stack keeps the implementation compact while the
    layer normalisation guards against scale shifts between entities.
    """

    def __init__(self, feat_dim: int, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a scalar for each entity in ``x``.

        Args:
            x: ``[..., D]`` tensor containing per-entity features.

        Returns:
            ``[...,]`` tensor of scalar signals.
        """
        return self.net(x).squeeze(-1)


class LaplaceAE(nn.Module):
     """Single-layer Laplace auto-encoder for entity level signals.

    Inputs (matching the EfficientSummarizer contract):

    ``x``
        Value stream tensor with shape ``[B, K, N, D]``.
    ``ctx_diff``
        Auxiliary stream tensor (time-differences) with the same shape as
        ``x``.
    ``entity_mask``
        Boolean tensor ``[B, N]`` indicating which entities are real (``True``)
        versus padded (``False``).

    Returns a tuple ``(context, aux)`` where ``context`` are pooled tokens with
    shape ``[B, S, Hc]`` and ``aux`` contains the intermediate signals and
    reconstructions required to compute the reconstruction loss.
    """
    def __init__(
        self,
        num_entities: int,
        feat_dim: int,
        *,
        lap_k: int = 8,
        tv_hidden: int = 32,
        out_len: int = 16,
        context_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.N = num_entities
        self.D = feat_dim
        self.K = lap_k
        self.Hc = context_dim
        self.S = out_len

        # Per-entity feature→signal heads
        self.v_head = TVHead(self.D, tv_hidden)
        self.t_head = TVHead(self.D, tv_hidden)

        # 1-layer parallel Laplace encoders for V and T signals
        self.lap_v = LearnableLaplaceBasis(k=self.K, feat_dim=self.N, mode="parallel")
        self.lap_t = LearnableLaplaceBasis(k=self.K, feat_dim=self.N, mode="parallel")

        # Matching 1-layer inverses
        self.inv_v = LearnablepesudoInverse(self.lap_v)
        self.inv_t = LearnablepesudoInverse(self.lap_t)

        # Simple token projection from concatenated Laplace features
        self.token_proj = nn.Sequential(
            nn.Linear(2 * self.K, self.Hc), nn.GELU(), nn.Dropout(dropout)
        )
        # Learn S query tokens to pool over K time steps via cross-attn-lite (linear)
        self.queries = nn.Parameter(torch.randn(self.S, self.Hc) / math.sqrt(self.Hc))
        self.norm = nn.LayerNorm(self.Hc)

    @staticmethod
    def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask_bn: torch.Tensor) -> torch.Tensor:
        """pred/target: [B,K,N], mask_bn: [B,N] bool. Returns mean over valid elements."""
        m = mask_bn.to(dtype=pred.dtype)[..., None, :]  # [B,1,N]
        se = (pred - target).pow(2)  # [B,K,N]
        se = se * m
        denom = m.sum() * pred.size(1)  # K is unmasked; weight only valid entities
        denom = denom.clamp_min(1.0)
        return se.sum() / denom

    def forward(
        self,
        x: torch.Tensor,                    # [B,K,N,D]   (V series)
        pad_mask: Optional[torch.Tensor] = None,  # unused (keep signature)
        dt: Optional[torch.Tensor] = None,        # unused
        ctx_diff: Optional[torch.Tensor] = None,  # [B,K,N,D] (T series)
        entity_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if entity_mask is None:
            raise ValueError("entity_mask must be provided: [B,N] bool")
        if entity_mask.dtype != torch.bool:
            raise TypeError(f"entity_mask must be boolean, got {entity_mask.dtype}")
        if ctx_diff is None:
            raise ValueError("ctx_diff (T series) must be provided: [B,K,N,D]")

        B, K, N, D = x.shape
        if ctx_diff.shape != x.shape:
            raise ValueError(
                f"ctx_diff must have same shape as x. Got {ctx_diff.shape}, expected {x.shape}"
            )
        if entity_mask.shape != (B, N):
            raise ValueError(
                f"entity_mask must be of shape [B,N]={B,N}, got {tuple(entity_mask.shape)}"
            )
        assert N == self.N and D == self.D, f"Got (N,D)=({N},{D}), expected ({self.N},{self.D})"

        # ---- Build scalar signals per entity (EFF-style) ----
        mask = entity_mask.to(dtype=x.dtype, device=x.device)
        mask = mask.unsqueeze(1)  # [B,1,N] for broadcasting

        v_sig = self.v_head(x) * mask  # [B,K,N]
        t_sig = self.t_head(ctx_diff) * mask  # [B,K,N]

        # ---- 1-layer Laplace encoders (parallel) ----
        v_lap = self.lap_v(v_sig)  # [B,K,2K]
        t_lap = self.lap_t(t_sig)  # [B,K,2K]

        # ---- Reconstructions via single inverses ----
        v_hat = self.inv_v(v_lap) * mask  # [B,K,N]
        t_hat = self.inv_t(t_lap) * mask  # [B,K,N]

        # ---- Lightweight context tokens (for downstream conditioning) ----
        fused = torch.cat([v_lap, t_lap], dim=-1)          # [B,K,4K]
        # First compress back to 2K before projection (keeps param count modest)
        fused = fused.view(B, K, 2, 2 * self.K).sum(dim=2)  # [B,K,2K]
        tokens = self.token_proj(fused)                     # [B,K,Hc]
        # Pool K→S with learned queries via simple attention scores
        # (no expensive MHA; just content-based pooling)
        q = self.queries[None, :, :]                        # [1,S,Hc]
        att = torch.einsum('bkh,bsh->bks', self.norm(tokens), self.norm(q)) / math.sqrt(self.Hc)
        att = att.softmax(dim=1)                            # softmax over K
        context = torch.einsum('bks,bkh->bsh', att, tokens) # [B,S,Hc]

        aux = {
            'v_sig': v_sig, 't_sig': t_sig,
            'v_hat': v_hat, 't_hat': t_hat,
        }
        # loss is computed externally; keep forward pure when used inside bigger graphs
        return context, aux

    def recon_loss(self, aux: Dict[str, torch.Tensor], entity_mask: torch.Tensor) -> torch.Tensor:
        """Convenience helper for training scripts."""
        lv = self._masked_mse(aux['v_hat'], aux['v_sig'], entity_mask)
        lt = self._masked_mse(aux['t_hat'], aux['t_sig'], entity_mask)
        return lv + lt
