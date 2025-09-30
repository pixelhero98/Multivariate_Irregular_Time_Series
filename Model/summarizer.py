import math
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from Model.laptrans import LearnableLaplaceBasis, LearnablepesudoInverse  # note: class name spelled as in repo


class TVHead(nn.Module):
    """Per-entity feature → scalar signal head (works on tensors with ..., D)."""
    def __init__(self, feat_dim: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., D] → [...]
        return self.net(x).squeeze(-1)


class LaplaceAE(nn.Module):
    """
    Minimal autoencoder for EFF-style entity signals using a single parallel
    LearnableLaplaceBasis encoder and a single LearnablepesudoInverse decoder
    per stream (V and T). The module also produces a lightweight context token
    stream by projecting concatenated Laplace features.

    Inputs (match EfficientSummarizer signature so it can be dropped-in later):
        x:          [B,K,N,D]   ̶→ source series (we use it for V)
        ctx_diff:   [B,K,N,D]   ̶→ auxiliary series (we use it for T)
        entity_mask:[B,N]       ̶→ boolean mask of valid entities

    Returns:
        context:    [B,S,Hc]    ̶→ simple token projection (S = out_len)
        aux: dict with:
            v_sig, t_sig:       ground truth signals [B,K,N]
            v_hat, t_hat:       reconstructions       [B,K,N]
            recon_loss:         scalar
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
        self.inv_v = LearnablepesudoInverse(k=self.K, feat_dim=self.N, mode="parallel")
        self.inv_t = LearnablepesudoInverse(k=self.K, feat_dim=self.N, mode="parallel")

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
        if ctx_diff is None:
            raise ValueError("ctx_diff (T series) must be provided: [B,K,N,D]")

        B, K, N, D = x.shape
        assert N == self.N and D == self.D, f"Got (N,D)=({N},{D}), expected ({self.N},{self.D})"

        # ---- Build scalar signals per entity (EFF-style) ----
        v_sig = self.v_head(x)         # [B,K,N]
        t_sig = self.t_head(ctx_diff)  # [B,K,N]

        # ---- 1-layer Laplace encoders (parallel) ----
        v_lap = self.lap_v(v_sig)  # [B,K,2K]
        t_lap = self.lap_t(t_sig)  # [B,K,2K]

        # ---- Reconstructions via single inverses ----
        v_hat = self.inv_v(v_lap)  # [B,K,N]
        t_hat = self.inv_t(t_lap)  # [B,K,N]

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
