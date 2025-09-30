import os
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse the Laplace encoder/decoder from your repo
from Model.laptrans import LearnableLaplaceBasis, LearnablepesudoInverse


class TVHead(nn.Module):
    """
    Matches the light per-entity scalar heads used in the EFF summarizer.
    Input:  x  [B, T, N, D]
    Output: y  [B, T, N]  (one scalar per entity)
    """
    def __init__(self, feat_dim: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N, D = x.shape
        y = self.net(x).squeeze(-1)  # [B,T,N]
        return y


class EffSummarizerAE(nn.Module):
    """
    EFF-style *deterministic* summarizer pretrainer (autoencoder over entity signals).

    Encoder:   TVHead -> LearnableLaplaceBasis (mode='parallel')
    Decoder:   LearnablepesudoInverse (1 layer, complex-aware linear + optional tiny MLP)

    It learns two independent AEs, one for V and one for T, sharing the same architecture.

    Targets are entity-wise scalar signals (one per entity) derived from V and T via TVHead,
    which matches how the runtime EFF summarizer forms its Laplace inputs.
    """

    def __init__(
        self,
        num_entities: int,
        feat_dim: int,
        lap_k: int = 8,
        tv_dim: int = 32,
        use_residual_mlp: bool = True,
    ) -> None:
        super().__init__()
        self.N = int(num_entities)
        self.D = int(feat_dim)
        self.K = int(lap_k)

        # Heads that compress feature dimension to one scalar per entity
        self.v_head = TVHead(self.D, hidden=tv_dim)
        self.t_head = TVHead(self.D, hidden=tv_dim)

        # Parallel (time-invariant) Laplace encoders over the entity axis
        self.lap_v = LearnableLaplaceBasis(k=self.K, feat_dim=self.N, mode="parallel")
        self.lap_t = LearnableLaplaceBasis(k=self.K, feat_dim=self.N, mode="parallel")

        # Lightweight decoders back to entity signals
        self.dec_v = LearnablepesudoInverse(self.lap_v, use_mlp_residual=use_residual_mlp)
        self.dec_t = LearnablepesudoInverse(self.lap_t, use_mlp_residual=use_residual_mlp)

        # Small output-scale parameters (learnable) in case targets differ in magnitude
        init_raw = math.log(math.e - 1.0)
        self.v_gain_raw = nn.Parameter(torch.tensor(init_raw))
        self.t_gain_raw = nn.Parameter(torch.tensor(init_raw))

    @staticmethod
    def _apply_entity_mask(x_btnd: torch.Tensor, m_bn: Optional[torch.Tensor]) -> torch.Tensor:
        if m_bn is None:
            return x_btnd
        B, T, N, D = x_btnd.shape
        m = m_bn.to(dtype=x_btnd.dtype, device=x_btnd.device)
        if m.dim() == 2:  # [B,N]
            m = m.unsqueeze(1).expand(B, T, N)
        return x_btnd * m.unsqueeze(-1)

    @staticmethod
    def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return F.mse_loss(pred, target, reduction='mean')
        B, T, N = target.shape
        m = mask.to(dtype=target.dtype, device=target.device)
        if m.dim() == 2:
            m = m.unsqueeze(1).expand(B, T, N)
        diff2 = (pred - target) ** 2 * m
        denom = m.sum().clamp_min(1.0)
        return diff2.sum() / denom

    def encode_signals(
        self,
        V: torch.Tensor,   # [B,T,N,D]
        T: torch.Tensor,   # [B,T,N,D]
        entity_mask: Optional[torch.Tensor] = None,  # [B,N]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (V_sig, T_sig, L_v, L_t)."""
        V = self._apply_entity_mask(V, entity_mask)
        T = self._apply_entity_mask(T, entity_mask)
        V_sig = self.v_head(V)  # [B,T,N]
        T_sig = self.t_head(T)  # [B,T,N]
        L_v = self.lap_v(V_sig)  # [B,T,2K]
        L_t = self.lap_t(T_sig)  # [B,T,2K]
        return V_sig, T_sig, L_v, L_t

    def decode_signals(self, L_v: torch.Tensor, L_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        V_rec = self.dec_v(L_v)  # [B,T,N]
        T_rec = self.dec_t(L_t)  # [B,T,N]
        # Allow a simple learned positive rescale
        V_rec = F.softplus(self.v_gain_raw) * V_rec
        T_rec = F.softplus(self.t_gain_raw) * T_rec
        return V_rec, T_rec

    def forward(
        self,
        V: torch.Tensor,   # [B,T,N,D]
        T: torch.Tensor,   # [B,T,N,D]
        entity_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        V_sig, T_sig, L_v, L_t = self.encode_signals(V, T, entity_mask)
        V_rec, T_rec = self.decode_signals(L_v, L_t)
        aux = {
            "V_sig": V_sig.detach(),
            "T_sig": T_sig.detach(),
            "L_v": L_v.detach(),
            "L_t": L_t.detach(),
        }
        return (V_rec, T_rec), aux

    def recon_loss(
        self,
        V: torch.Tensor,
        T: torch.Tensor,
        entity_mask: Optional[torch.Tensor] = None,
        w_v: float = 1.0,
        w_t: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        V_sig, T_sig, L_v, L_t = self.encode_signals(V, T, entity_mask)
        V_rec, T_rec = self.decode_signals(L_v, L_t)
        mv = self._masked_mse(V_rec, V_sig, entity_mask)
        mt = self._masked_mse(T_rec, T_sig, entity_mask)
        loss = w_v * mv + w_t * mt
        return loss, {"mv": float(mv.item()), "mt": float(mt.item())}

    @torch.no_grad()
    def summarize(
        self,
        V: torch.Tensor,
        T: torch.Tensor,
        entity_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Deterministic summary tokens for downstream diffusion.
        Here we just return the concatenated Laplace features [L_v | L_t].
        Shape: [B, T, 4K]
        """
        self.eval()
        _, _, L_v, L_t = self.encode_signals(V, T, entity_mask)
        return torch.cat([L_v, L_t], dim=-1)
