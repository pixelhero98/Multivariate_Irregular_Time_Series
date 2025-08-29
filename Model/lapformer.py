import torch
import torch.nn as nn
from typing import Optional, Union, List
from laptrans import LearnableLaplacianBasis, LearnableInverseLaplacianBasis
from pos_time_emb import get_sinusoidal_pos_emb


# -------------------------------
# AdaLayerNorm
# -------------------------------
class AdaLayerNorm(nn.Module):
    """LayerNorm whose scale/shift come from a conditioning vector c:[B,Hc]."""
    def __init__(self, normalized_shape: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, elementwise_affine=False)
        self.to_ss = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * normalized_shape),
        )
        nn.init.zeros_(self.to_ss[-1].weight); nn.init.zeros_(self.to_ss[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        s, b = self.to_ss(c).chunk(2, dim=-1)
        return (1 + s).unsqueeze(1) * h + b.unsqueeze(1)


# -------------------------------
# Transformer block with optional additive attention bias
# -------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.0, attn_dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

        self.norm1 = AdaLayerNorm(hidden_dim, hidden_dim)
        self.norm2 = AdaLayerNorm(hidden_dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(hidden_dim * mlp_ratio), hidden_dim),
            nn.Dropout(dropout),
        )
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self,
                x: torch.Tensor,                                # [B,L,H]
                attn_mask: Optional[torch.Tensor] = None,       # [L,L] or [B,L,L]
                key_padding_mask: Optional[torch.Tensor] = None,# [B,L] bool
                attn_bias: Optional[torch.Tensor] = None,       # [L,L] or [B,L,L] additive
                t_vec: Optional[torch.Tensor] = None            # [B,H] (unused here; kept for API symmetry)
                ) -> torch.Tensor:
        B, L, H = x.shape
        h = self.norm1(x, t_vec)
        qkv = self.qkv(h).reshape(B, L, 3, self.num_heads, H // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # [B,heads,L,head_dim]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (k.shape[-1] ** 0.5)  # [B,heads,L,L]

        # Fold masks/bias (if provided)
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_scores = attn_scores + attn_mask  # [L,L] broadcast over batch/heads
            else:
                attn_scores = attn_scores + attn_mask
        if attn_bias is not None:
            if attn_bias.dim() == 2:
                attn_scores = attn_scores + attn_bias
            else:
                attn_scores = attn_scores + attn_bias
        if key_padding_mask is not None:
            # mask = True for PAD positions -> add -inf to those positions
            big_neg = torch.finfo(attn_scores.dtype).min
            attn_scores = attn_scores.masked_fill(key_padding_mask[:, None, None, :], big_neg)

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        attn_out = torch.matmul(attn_weights, v)                     # [B,heads,L,head_dim]
        attn_out = attn_out.transpose(1, 2).reshape(B, L, H)
        x = x + self.resid_dropout(self.proj(attn_out))
        x = x + self.mlp(self.norm2(x, t_vec))
        return x


# -------------------------------
# Cross-attn block (H <-- H)
# -------------------------------
class CrossAttnBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0, attn_dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.kv = nn.Linear(hidden_dim, hidden_dim * 2)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm_q = AdaLayerNorm(hidden_dim, hidden_dim)
        self.norm_kv = AdaLayerNorm(hidden_dim, hidden_dim)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor, t_vec: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, H = x_q.shape
        xq = self.norm_q(x_q, t_vec)
        xkv = self.norm_kv(x_kv, t_vec)

        q = self.q(xq).reshape(B, L, self.num_heads, H // self.num_heads).transpose(1, 2)       # [B,heads,L,d]
        kv = self.kv(xkv).reshape(B, x_kv.shape[1], 2, self.num_heads, H // self.num_heads)
        k = kv[:, :, 0].transpose(1, 2)                                                          # [B,heads,Lc,d]
        v = kv[:, :, 1].transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (k.shape[-1] ** 0.5)
        attn = torch.softmax(attn_scores, dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, L, H)
        return x_q + self.resid_dropout(self.proj(out))


# -------------------------------
# Laplace "sandwich" block
# -------------------------------
class LaplaceSandwichBlock(nn.Module):
    """
    time x:[B,L,D] --(LearnableLaplacianBasis k)-> z:[B,L,2k]
      -> Linear(2k->H) + pos + time (+ optional self-cond add in H)
      -> TransformerBlock(H) with additive bias over time axis
      -> CrossAttn(H  <-- summary2lap->H)
      -> Linear(H->2k) residual in Laplace domain
      -> LearnableInverseLaplacianBasis -> y:[B,L,D]
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int, k: int,
                 dropout: float = 0.0, attn_dropout: float = 0.0):
        super().__init__()
        # analysis / synthesis in Laplace domain
        self.analysis  = LearnableLaplacianBasis(k=k, feat_dim=input_dim)     # [B,L,2k]
        self.synthesis = LearnableInverseLaplacianBasis(self.analysis)        # 2k -> D
        # Laplace <-> hidden
        self.lap2hid = nn.Linear(2 * k, hidden_dim)
        self.hid2lap = nn.Linear(hidden_dim, 2 * k)
        # core attention blocks
        self.self_blk  = TransformerBlock(hidden_dim, num_heads, mlp_ratio=4.0, dropout=dropout, attn_dropout=attn_dropout)
        self.cross_blk = CrossAttnBlock(hidden_dim, num_heads, dropout=dropout, attn_dropout=attn_dropout)
        # start near-identity for stability
        nn.init.zeros_(self.hid2lap.weight); nn.init.zeros_(self.hid2lap.bias)

    def forward(
        self,
        x_time: torch.Tensor,
        pos_emb: torch.Tensor,
        t_vec: torch.Tensor,
        attn_bias: Optional[torch.Tensor],
        summary_kv_H: Optional[torch.Tensor] = None,
        sc_add_H: Optional[torch.Tensor] = None,
        dt: torch.Tensor | None = None
    ) -> torch.Tensor:
        # time -> Laplace
        z = self.analysis(x_time, dt=dt)                                 # [B,L,2k]
        # Laplace -> hidden
        h = self.lap2hid(z) + pos_emb + t_vec.unsqueeze(1)               # [B,L,H]
        if sc_add_H is not None:
            h = h + sc_add_H
        # self-attn (time axis) with shared bias
        h = self.self_blk(h, attn_mask=None, key_padding_mask=None, attn_bias=attn_bias, t_vec=t_vec)
        # cross-attn to context summary (already width H after summary2lap proj)
        if summary_kv_H is not None:
            h = self.cross_blk(h, summary_kv_H, t_vec=t_vec)
        # hidden -> Laplace (residual), Laplace -> time
        z_upd  = z + self.hid2lap(h)                              # [B,L,2k]
        y_time = self.synthesis(z_upd)                            # [B,L,D]
        return y_time


# -------------------------------
# LapFormer (stack)
# -------------------------------
class LapFormer(nn.Module):
    """
    Stack of Laplace-sandwich blocks (per-block k) with per-block summary2lap conditioning.
    Self-conditioning supported (no gates).
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int,
                 num_heads: int, laplace_k: Union[int, List[int]] = 8,
                 dropout: float = 0.0, attn_dropout: float = 0.0,
                 self_conditioning: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.self_conditioning = self_conditioning

        # per-block settings
        if isinstance(laplace_k, int):
            ks = [laplace_k] * num_layers
        else:
            ks = laplace_k
            assert len(ks) == num_layers, "laplace_k list must match num_layers"

        # sinusoidal pos emb over sequence length (cached max 1024, extended if needed at runtime)
        self.register_buffer("pos_cache", get_sinusoidal_pos_emb(L=1024, dim=hidden_dim, device=torch.device("cpu")), persistent=False)

        # per-block conditioning from summary (Laplace -> H)
        self.summary2lap = nn.ModuleList([
            nn.Linear(hidden_dim, 2 * k) for k in ks
        ])
        self.summary2hid = nn.ModuleList([
            nn.Linear(2 * k, hidden_dim) for k in ks
        ])

        # optional self-conditioning injection (time domain -> H)
        if self.self_conditioning:
            self.self_cond_proj = nn.Linear(input_dim, hidden_dim)
        else:
            self.self_cond_proj = None

        # sandwich blocks
        self.blocks = nn.ModuleList([
            LaplaceSandwichBlock(input_dim=input_dim, hidden_dim=hidden_dim, num_heads=num_heads,
                                 k=k, dropout=dropout, attn_dropout=attn_dropout)
            for k in ks
        ])

        # tiny head
        self.head_norm = nn.LayerNorm(input_dim)
        self.head_proj = nn.Linear(input_dim, input_dim)
        nn.init.zeros_(self.head_proj.weight); nn.init.zeros_(self.head_proj.bias)

    def _pos(self, L: int, B: int, device: torch.device) -> torch.Tensor:
        # ensure cache large enough
        if self.pos_cache.shape[1] < L:
            self.pos_cache = get_sinusoidal_pos_emb(L, self.hidden_dim, device=self.pos_cache.device)
        pos = self.pos_cache[:, :L, :].to(device)      # [1,L,H]
        return pos.expand(B, -1, -1)                   # [B,L,H]

    def forward(self,
                x_tokens: torch.Tensor,                       # [B,L,D] time-domain tokens
                t_vec: torch.Tensor,                          # [B,H] time embedding already in H
                cond_summary: Optional[torch.Tensor] = None,  # [B,Lh,H] summary features in H
                sc_feat: Optional[torch.Tensor] = None,       # [B,L,D] self-cond in time domain
                dt: Optional[torch.Tensor] = None             # [B,L] time step deltas
                ) -> torch.Tensor:
        B, L, D = x_tokens.shape
        device = x_tokens.device

        # positional embeddings shared across blocks (sum with lap2hid output)
        pos = self._pos(L, B, device)                         # [B,L,H]

        # optional self-conditioning addend (H)
        sc_add_H = None
        if self.self_conditioning and (sc_feat is not None) and (self.self_cond_proj is not None):
            sc_add_H = self.self_cond_proj(sc_feat)           # [B,L,H]

        # precompute per-block summary2lap -> H
        kvs: List[Optional[torch.Tensor]] = [None] * len(self.blocks)
        if cond_summary is not None:
            for i in range(len(self.blocks)):
                s_lap = self.summary2lap[i](cond_summary)              # [B,Lh,2k_i]
                kvs[i] = self.summary2hid[i](s_lap)                    # [B,Lh,H]

        # run the sandwich stack; each block returns TIME-domain [B,L,D]
        h_time = x_tokens
        for i, blk in enumerate(self.blocks):
            h_time = blk(h_time, pos, t_vec, attn_bias=None, summary_kv_H=kvs[i], sc_add_H=sc_add_H, dt=dt)

        # tiny head in time domain (identity at init)
        out = self.head_proj(self.head_norm(h_time))
        return out
