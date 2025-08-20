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
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
        )
        self.drop_path2 = nn.Dropout(dropout)

        # Residual scaling init for stability (helps regression calibration)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None, attn_bias: Optional[torch.Tensor] = None):
        h = self.norm1(x)
        combined_mask = None
        if attn_mask is not None and attn_bias is not None:
            combined_mask = attn_mask + attn_bias
        elif attn_mask is not None:
            combined_mask = attn_mask
        else:
            combined_mask = attn_bias

        h, _ = self.attn(h, h, h, attn_mask=combined_mask, key_padding_mask=key_padding_mask)
        x = x + self.drop_path1(h)

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

    def forward(self, x: torch.Tensor, kv: torch.Tensor, kv_mask: Optional[torch.Tensor] = None):
        q = self.norm_q(x)
        k = self.norm_kv(kv)
        h, _ = self.cross(q, k, k, key_padding_mask=kv_mask)
        return x + self.drop(h)


# -------------------------------
# LapFormer (multi-resolution + self-conditioning)
# -------------------------------

class LaplaceSandwichBlock(nn.Module):
    """
    time x:[B,L,D] --(LearnableLaplacianBasis k)-> z:[B,L,2k]
      -> Linear(2k->H) + pos + time (+ optional self-cond add in H)
      -> TransformerBlock(H) with RPB over time axis
      -> CrossAttn(H  <-- summary2lap->H)
      -> Linear(H->2k) residual in Laplace domain
      -> LearnableInverseLaplacianBasis -> y:[B,L,D]
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int, k: int,
                 dropout: float = 0.0, attn_dropout: float = 0.0):
        super().__init__()
        # analysis / synthesis in Laplace domain
        self.analysis  = LearnableLaplacianBasis(k=k, feat_dim=input_dim)     # [B,L,2k]
        self.synthesis = LearnableInverseLaplacianBasis(self.analysis)         # 2k -> D
        # Laplace <-> hidden
        self.lap2hid = nn.Linear(2 * k, hidden_dim)
        self.hid2lap = nn.Linear(hidden_dim, 2 * k)
        # core attention blocks
        self.self_blk  = TransformerBlock(hidden_dim, num_heads, mlp_ratio=4.0,
                                          dropout=dropout, attn_dropout=attn_dropout)
        self.cross_blk = CrossAttnBlock(hidden_dim, num_heads, dropout=dropout,
                                        attn_dropout=attn_dropout)
        # start near-identity for stability
        nn.init.zeros_(self.hid2lap.weight); nn.init.zeros_(self.hid2lap.bias)

    def forward(self,
                x_time: torch.Tensor,
                pos_emb: torch.Tensor,
                t_vec: torch.Tensor,
                attn_bias: torch.Tensor,
                summary_kv_H: Optional[torch.Tensor] = None,
                kv_mask: Optional[torch.Tensor] = None,
                sc_add_H: Optional[torch.Tensor] = None) -> torch.Tensor:
        # time -> Laplace
        z = self.analysis(x_time)                                 # [B,L,2k]
        # Laplace -> hidden
        h = self.lap2hid(z) + pos_emb + t_vec.unsqueeze(1)        # [B,L,H]
        if sc_add_H is not None:
            h = h + sc_add_H
        # self-attn (time axis) with shared RPB
        h = self.self_blk(h, attn_mask=None, key_padding_mask=None, attn_bias=attn_bias)
        # cross-attn to context summary (already width H after summary2lap proj)
        if summary_kv_H is not None:
            h = self.cross_blk(h, summary_kv_H, kv_mask=kv_mask)
        # hidden -> Laplace (residual), Laplace -> time
        z_upd  = z + self.hid2lap(h)                              # [B,L,2k]
        y_time = self.synthesis(z_upd)                            # [B,L,D]
        return y_time

class LapFormer(nn.Module):
    """
    Stack of Laplace-sandwich blocks (per-block k) with per-block summary2lap conditioning.
    Self-conditioning supported (no gates). Public API unchanged.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, num_heads: int,
                 laplace_k: Union[int, List[int]] = 16, dropout: float = 0.0, attn_dropout: float = 0.0,
                 self_conditioning: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.self_conditioning = self_conditioning

        # ---------- per-block k parsing ----------
        # accepts: int -> broadcast; list -> one per block (if longer, use first num_layers after an optional stem)
        if isinstance(laplace_k, (list, tuple)):
            k_list = list(laplace_k)
            if len(k_list) == 0:
                raise ValueError("laplace_k list must be non-empty")
            if len(k_list) >= num_layers + 1:
                per_layer_k = k_list[1:1+num_layers]            # drop optional stem k0
            elif len(k_list) >= num_layers:
                per_layer_k = k_list[:num_layers]
            else:
                per_layer_k = k_list + [k_list[-1]] * (num_layers - len(k_list))
        else:
            per_layer_k = [int(laplace_k)] * num_layers

        # optional self-conditioning: project time-domain sc_feat to H (added pre-attn in H)
        if self.self_conditioning:
            self.self_cond_proj = nn.Linear(input_dim, hidden_dim)

        # timestep embedding -> H (kept)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        # per-block summary2lap: [B,Lh,H] -> Laplace(2k_i) -> H
        self.summary2lap = nn.ModuleList([
            LearnableLaplacianBasis(k=k_i, feat_dim=hidden_dim) for k_i in per_layer_k
        ])
        self.summary2hid = nn.ModuleList([
            nn.Linear(2 * k_i, hidden_dim) for k_i in per_layer_k
        ])

        # Laplace-sandwich blocks over time
        self.blocks = nn.ModuleList([
            LaplaceSandwichBlock(input_dim=input_dim, hidden_dim=hidden_dim, num_heads=num_heads,
                                 k=per_layer_k[i], dropout=dropout, attn_dropout=attn_dropout)
            for i in range(num_layers)
        ])

        # shared RPB and small head in TIME domain, zero-init
        self.head_norm = nn.LayerNorm(input_dim)
        self.head_proj = nn.Linear(input_dim, input_dim)
        nn.init.zeros_(self.head_proj.weight); nn.init.zeros_(self.head_proj.bias)

    def forward(self, x_tokens: torch.Tensor, t_emb: torch.Tensor,
                cond_summary: Optional[torch.Tensor] = None,
                cond_mask: Optional[torch.Tensor] = None,
                sc_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x_tokens: [B, L, D] time-domain latents
        returns:  [B, L, D] native-param prediction in time domain
        """
        B, L, D = x_tokens.shape
        device = x_tokens.device

        # positional & time embeddings (shared across blocks)
        pos = get_sinusoidal_pos_emb(L, self.hidden_dim, device)        # [1,L,H]
        t_vec = self.time_mlp(t_emb)                                    # [B,H]

        # self-conditioning add in H (optional)
        sc_add_H = self.self_cond_proj(sc_feat) if (self.self_conditioning and sc_feat is not None) else None

        # precompute per-block summary2lap -> H
        kvs = [None] * len(self.blocks)
        if cond_summary is not None:
            for i in range(len(self.blocks)):
                s_lap = self.summary2lap[i](cond_summary)              # [B,Lh,2k_i]
                kvs[i] = self.summary2hid[i](s_lap)                    # [B,Lh,H]

        # run the sandwich stack; each block returns TIME-domain [B,L,D]
        h_time = x_tokens
        for i, blk in enumerate(self.blocks):
            h_time = blk(h_time, pos, t_vec, None, kvs[i], cond_mask, sc_add_H)

        # tiny head in time domain (identity at init)
        out = self.head_proj(self.head_norm(h_time))
        return out