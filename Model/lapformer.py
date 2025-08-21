import torch
import torch.nn as nn
from typing import Optional, Union, List
from laptrans import LearnableLaplacianBasis, LearnableInverseLaplacianBasis
from pos_time_emb import get_sinusoidal_pos_emb

# -------------------------------
# Transformer blocks
# -------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0, attn_dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=attn_dropout)
        self.drop_path1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.drop_path2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: [B, L, D]
        attn_mask: [L, L] or [B*H, L, L] (additive). If provided together with attn_bias, they are added.
        key_padding_mask: [B, L] (bool) True=ignore
        attn_bias: [B*H, L, L] additive bias (e.g., per-head relative position bias)
        """
        B, L, D = x.shape
        h = self.norm1(x)

        # Combine mask and bias if both are provided
        combined_mask = None
        if (attn_mask is not None) and (attn_bias is not None):
            if attn_bias.dim() == 3:
                combined_mask = attn_mask + attn_bias
            else:
                combined_mask = attn_mask
        elif attn_bias is not None:
            combined_mask = attn_bias
        else:
            combined_mask = attn_mask

        out, _ = self.attn(h, h, h, attn_mask=combined_mask, key_padding_mask=key_padding_mask)
        x = x + self.drop_path1(out)

        h = self.norm2(x)
        h = self.mlp(h)
        x = x + self.drop_path2(h)
        return x


class CrossAttnBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0, attn_dropout: float = 0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=attn_dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """
        x:  [B, Lq, D]
        kv: [B, Lk, D]
        """
        q = self.norm_q(x)
        k = self.norm_kv(kv)
        h, _ = self.cross(q, k, k)  # no key_padding_mask: summary tokens are fixed-length
        return x + self.drop(h)


# -------------------------------
# LapFormer (multi-resolution + self-conditioning)
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
        h = self.lap2hid(z) + pos_emb + t_vec.unsqueeze(1)        # [B,L,H]
        if sc_add_H is not None:
            h = h + sc_add_H
        # self-attn (time axis) with shared bias
        h = self.self_blk(h, attn_mask=None, key_padding_mask=None, attn_bias=attn_bias)
        # cross-attn to context summary (already width H after summary2lap proj)
        if summary_kv_H is not None:
            h = self.cross_blk(h, summary_kv_H)
        # hidden -> Laplace (residual), Laplace -> time
        z_upd  = z + self.hid2lap(h)                              # [B,L,2k]
        y_time = self.synthesis(z_upd)                            # [B,L,D]
        return y_time


class LapFormer(nn.Module):
    """
    Stack of Laplace-sandwich blocks (per-block k) with per-block summary2lap conditioning.
    Self-conditioning supported (no gates).
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, num_heads: int,
                 laplace_k: Union[int, List[int]] = 16, dropout: float = 0.0, attn_dropout: float = 0.0,
                 self_conditioning: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.self_conditioning = bool(self_conditioning)

        if isinstance(laplace_k, int):
            per_layer_k = [laplace_k] * num_layers
        else:
            assert len(laplace_k) == num_layers, "laplace_k list must have length num_layers"
            per_layer_k = list(laplace_k)

        # per-block conditioning projections: summary(H) -> lap(2k) -> H
        self.summary2lap = nn.ModuleList([nn.Linear(hidden_dim, 2 * k) for k in per_layer_k])
        self.summary2hid = nn.ModuleList([nn.Linear(2 * k, hidden_dim) for k in per_layer_k])

        # the Laplace sandwich stack
        self.blocks = nn.ModuleList([
            LaplaceSandwichBlock(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                k=per_layer_k[i],
                dropout=dropout,
                attn_dropout=attn_dropout,
            ) for i in range(num_layers)
        ])

        # shared tiny head in TIME domain, zero-init (near identity)
        self.head_norm = nn.LayerNorm(input_dim)
        self.head_proj = nn.Linear(input_dim, input_dim)
        nn.init.eye_(self.head_proj.weight); nn.init.zeros_(self.head_proj.bias)

        # time embedding processor (expects [B, hidden_dim] from outer module)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        # optional self-conditioning addend (D -> H)
        self.self_cond_proj = nn.Linear(input_dim, hidden_dim)

    def forward(
        self,
        x_tokens: torch.Tensor,
        t_emb: torch.Tensor,
        cond_summary: Optional[torch.Tensor] = None,
        sc_feat: Optional[torch.Tensor] = None,
        dt: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x_tokens: [B, L, D] time-domain latents
        t_emb:    [B, H] time embedding processed outside
        cond_summary: [B, Lh, H] or None
        returns:  [B, L, D] native-param prediction in time domain
        """
        B, L, D = x_tokens.shape
        device = x_tokens.device

        # positional & time embeddings (shared across blocks)
        pos = get_sinusoidal_pos_emb(L, self.hidden_dim, device).to(x_tokens.dtype)        # [1,L,H]
        t_vec = self.time_mlp(t_emb).to(x_tokens.dtype)                                   # [B,H]

        # self-conditioning add in H (optional)
        sc_add_H = None
        if self.self_conditioning and (sc_feat is not None):
            # sc_feat is in time domain [B,L,D]
            sc_add_H = self.self_cond_proj(sc_feat)                     # [B,L,H]

        # precompute per-block summary2lap -> H
        kvs: List[Optional[torch.Tensor]] = [None] * len(self.blocks)
        if cond_summary is not None:
            for i in range(len(self.blocks)):
                s_lap = self.summary2lap[i](cond_summary)              # [B,Lh,2k_i]
                kvs[i] = self.summary2hid[i](s_lap)                    # [B,Lh,H]

        # run the sandwich stack; each block returns TIME-domain [B,L,D]
        h_time = x_tokens
        for i, blk in enumerate(self.blocks):
            h_time = blk(h_time, pos, t_vec, None, kvs[i], sc_add_H, dt=dt)

        # tiny head in time domain (identity at init)
        out = self.head_proj(self.head_norm(h_time))
        return out