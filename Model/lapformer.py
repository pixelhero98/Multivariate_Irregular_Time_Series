import torch
import torch.nn as nn
from typing import Optional, Union, List
from Model.laptrans import LearnableLaplaceBasis, LearnablepesudoInverse
from Model.pos_time_emb import get_sinusoidal_pos_emb


def _canon_mode(mode: str) -> str:
    m = mode.lower()
    if m in {"parallel", "static", "global"}:
        return "parallel"
    if m in {"recurrent", "tv", "time_varying", "time-varying"}:
        return "recurrent"
    raise ValueError("lap_mode must be 'parallel' or 'recurrent' (or aliases)")


class AdaLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, elementwise_affine=False)
        self.to_ss = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 2 * normalized_shape))
        nn.init.zeros_((self.to_ss[-1]).weight);
        nn.init.zeros_(self.to_ss[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        s, b = self.to_ss(c).chunk(2, dim=-1)

        return (1 + s).unsqueeze(1) * h + b.unsqueeze(1)


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.0, attn_dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim;
        self.num_heads = num_heads
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = AdaLayerNorm(hidden_dim, hidden_dim)
        self.norm2 = AdaLayerNorm(hidden_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim * mlp_ratio)), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(int(hidden_dim * mlp_ratio), hidden_dim), nn.Dropout(dropout),
        )
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None,
                attn_bias: Optional[torch.Tensor] = None,
                t_vec: Optional[torch.Tensor] = None) -> torch.Tensor:

        B, L, H = x.shape
        h = self.norm1(x, t_vec)
        qkv = self.qkv(h).reshape(B, L, 3, self.num_heads, H // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (k.shape[-1] ** 0.5)

        if attn_mask is not None: attn_scores = attn_scores + attn_mask
        if attn_bias is not None: attn_scores = attn_scores + attn_bias
        if key_padding_mask is not None:
            big_neg = torch.finfo(attn_scores.dtype).min
            attn_scores = attn_scores.masked_fill(key_padding_mask[:, None, None, :], big_neg)

        attn = torch.softmax(attn_scores, dim=-1)
        attn = self.attn_dropout(attn)
        attn_out = torch.matmul(attn, v).transpose(1, 2).reshape(B, L, H)
        x = x + self.resid_dropout(self.proj(attn_out))
        x = x + self.mlp(self.norm2(x, t_vec))

        return x


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
        xq = self.norm_q(x_q, t_vec);
        xkv = self.norm_kv(x_kv, t_vec)
        q = self.q(xq).reshape(B, L, self.num_heads, H // self.num_heads).transpose(1, 2)
        kv = self.kv(xkv).reshape(B, x_kv.shape[1], 2, self.num_heads, H // self.num_heads)
        k = kv[:, :, 0].transpose(1, 2);
        v = kv[:, :, 1].transpose(1, 2)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (k.shape[-1] ** 0.5)
        attn = torch.softmax(attn_scores, dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, L, H)

        return x_q + self.resid_dropout(self.proj(out))


class LaplaceSandwichBlock(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int, k: int,
                 lap_mode: str = "parallel",
                 dropout: float = 0.0, attn_dropout: float = 0.0,
                 cross_first: bool = True):
        super().__init__()
        self.mode = _canon_mode(lap_mode)
        self.cross_first = cross_first

        self.analysis  = LearnableLaplaceBasis(k=k, feat_dim=input_dim, mode=self.mode)
        self.synthesis = LearnablepesudoInverse(self.analysis)

        self.lap2hid = nn.Linear(2 * k, hidden_dim)
        self.hid2lap = nn.Linear(hidden_dim, 2 * k)
        nn.init.zeros_(self.hid2lap.weight)
        nn.init.zeros_(self.hid2lap.bias)

        self.self_blk  = TransformerBlock(
            hidden_dim, num_heads, mlp_ratio=4.0,
            dropout=dropout, attn_dropout=attn_dropout
        )
        self.cross_blk = CrossAttnBlock(
            hidden_dim, num_heads,
            dropout=dropout, attn_dropout=attn_dropout
        )

    def forward(self,
                x_time: torch.Tensor,
                pos_emb: torch.Tensor,
                t_vec: torch.Tensor,
                attn_bias: Optional[torch.Tensor] = None,
                summary_kv_H: Optional[torch.Tensor] = None,
                sc_add_H: Optional[torch.Tensor] = None,
                dt: Optional[torch.Tensor] = None) -> torch.Tensor:

        # Laplace analysis (parallel ignores dt)
        z = self.analysis(x_time, dt=dt) if self.mode == "recurrent" else self.analysis(x_time)

        # Token embedding + time/pos + additive context (FiLM-style adds can be done before this if desired)
        h = self.lap2hid(z) + pos_emb + t_vec.unsqueeze(1)
        if sc_add_H is not None:
            h = h + sc_add_H

        def cross(h: torch.Tensor) -> torch.Tensor:
            if summary_kv_H is None:
                return h
            return self.cross_blk(h, summary_kv_H, t_vec=t_vec)

        if self.cross_first:
            # use context first, then mix locally
            h = cross(h)
            h = self.self_blk(h, attn_mask=None, key_padding_mask=None,
                              attn_bias=attn_bias, t_vec=t_vec)
        else:
            # mix locally first, then consult context
            h = self.self_blk(h, attn_mask=None, key_padding_mask=None,
                              attn_bias=attn_bias, t_vec=t_vec)
            h = cross(h)

        # Update Laplace features and synthesize back to time
        z_upd  = z + self.hid2lap(h)
        y_time = self.synthesis(z_upd)
        return y_time


class LapFormer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int,
                 num_heads: int, laplace_k: Union[int, List[int]] = 8,
                 lap_mode: str = "parallel",
                 dropout: float = 0.0, attn_dropout: float = 0.0,
                 self_conditioning: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.self_conditioning = self_conditioning
        self.lap_mode = _canon_mode(lap_mode)

        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads for multi-head attention")

        ks = [laplace_k] * num_layers if isinstance(laplace_k, int) else laplace_k
        assert len(ks) == num_layers, "laplace_k list must match num_layers"

        self.register_buffer("pos_cache", get_sinusoidal_pos_emb(L=1024, dim=hidden_dim, device=torch.device("cpu")),
                             persistent=False)

        self.summary2lap = nn.ModuleList([nn.Linear(hidden_dim, 2 * k) for k in ks])
        self.summary2hid = nn.ModuleList([nn.Linear(2 * k, hidden_dim) for k in ks])

        self.self_cond_proj = nn.Linear(input_dim, hidden_dim) if self.self_conditioning else None

        self.blocks = nn.ModuleList([
            LaplaceSandwichBlock(input_dim=input_dim, hidden_dim=hidden_dim, num_heads=num_heads, k=k,
                                 lap_mode=self.lap_mode, dropout=dropout, attn_dropout=attn_dropout)
            for k in ks
        ])

        self.head_norm = nn.LayerNorm(input_dim)
        self.head_proj = nn.Linear(input_dim, input_dim)
        nn.init.zeros_(self.head_proj.weight);
        nn.init.zeros_(self.head_proj.bias)

    def _pos(self, L: int, B: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.pos_cache.shape[1] < L:
            self.pos_cache = get_sinusoidal_pos_emb(L, self.hidden_dim, device=self.pos_cache.device).to(
                dtype=self.pos_cache.dtype)
        return self.pos_cache[:, :L, :].to(device=device, dtype=dtype).expand(B, -1, -1)

    def forward(self, x_tokens: torch.Tensor, t_vec: torch.Tensor,
                cond_summary: Optional[torch.Tensor] = None,
                sc_feat: Optional[torch.Tensor] = None,
                dt: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, D = x_tokens.shape
        device = x_tokens.device
        if t_vec.dim() != 2 or t_vec.shape[0] != B or t_vec.shape[1] != self.hidden_dim:
            raise ValueError(f"t_vec must be [B, hidden_dim]={B, self.hidden_dim}; got {tuple(t_vec.shape)}")
        pos = self._pos(L, B, device, x_tokens.dtype)

        sc_add_H = self.self_cond_proj(sc_feat) if (self.self_conditioning and sc_feat is not None) else None

        kvs: List[Optional[torch.Tensor]] = [None] * len(self.blocks)
        if cond_summary is not None:
            if cond_summary.dim() != 3 or cond_summary.shape[0] != B or cond_summary.shape[2] != self.hidden_dim:
                raise ValueError(
                    f"cond_summary must be [B, S, hidden_dim] with hidden_dim={self.hidden_dim}; got {tuple(cond_summary.shape)}"
                )
            for i, k in enumerate(self.summary2lap):
                s_lap = k(cond_summary)
                kvs[i] = self.summary2hid[i](s_lap)

        h_time = x_tokens
        for i, blk in enumerate(self.blocks):
            h_time = blk(h_time, pos, t_vec, attn_bias=None, summary_kv_H=kvs[i], sc_add_H=sc_add_H, dt=dt)

        return self.head_proj(self.head_norm(h_time))