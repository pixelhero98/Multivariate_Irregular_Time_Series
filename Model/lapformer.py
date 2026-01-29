from typing import Optional

import torch
import torch.nn as nn

# Support both repository layouts
try:
    from Model.laptrans import ModalPredictor, ModalSynthesizer
except Exception:
    from laptrans import ModalPredictor, ModalSynthesizer


class AdaLayerNorm(nn.Module):
    """LayerNorm with feature-wise affine parameters conditioned on a vector."""

    def __init__(self, normalized_shape: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, elementwise_affine=False)
        self.to_ss = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * normalized_shape),
        )
        nn.init.zeros_(self.to_ss[-1].weight)
        nn.init.zeros_(self.to_ss[-1].bias)

    def forward(self, x: torch.Tensor, c: Optional[torch.Tensor]) -> torch.Tensor:
        h = self.norm(x)
        if c is None:
            scale = torch.zeros(h.size(0), h.size(-1), device=h.device, dtype=h.dtype)
            bias = torch.zeros_like(scale)
        else:
            scale, bias = self.to_ss(c).chunk(2, dim=-1)
        return (1 + scale).unsqueeze(1) * h + bias.unsqueeze(1)


class TransformerBlock(nn.Module):
    """Self-attention block (over modal tokens) with AdaLayerNorm conditioning."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
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

    def forward(
        self,
        x: torch.Tensor,
        t_vec: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, H = x.shape
        h = self.norm1(x, t_vec)
        qkv = (
            self.qkv(h)
            .reshape(B, L, 3, self.num_heads, H // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B,heads,L,dh]
        attn = torch.matmul(q, k.transpose(-2, -1)) / (k.shape[-1] ** 0.5)
        if attn_bias is not None:
            attn = attn + attn_bias
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, L, H)
        x = x + self.resid_dropout(self.proj(out))
        x = x + self.mlp(self.norm2(x, t_vec))
        return x


class CrossAttnBlock(nn.Module):
    """Cross-attention block: modal tokens attend to summary tokens."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.kv = nn.Linear(hidden_dim, hidden_dim * 2)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm_q = AdaLayerNorm(hidden_dim, hidden_dim)
        self.norm_kv = AdaLayerNorm(hidden_dim, hidden_dim)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: torch.Tensor,
        t_vec: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, Lq, H = x_q.shape
        Lkv = x_kv.shape[1]
        xq = self.norm_q(x_q, t_vec)
        xkv = self.norm_kv(x_kv, t_vec)

        q = self.q(xq).reshape(B, Lq, self.num_heads, H // self.num_heads).transpose(1, 2)
        kv = self.kv(xkv).reshape(B, Lkv, 2, self.num_heads, H // self.num_heads)
        k = kv[:, :, 0].transpose(1, 2)
        v = kv[:, :, 1].transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / (k.shape[-1] ** 0.5)
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, Lq, H)
        return x_q + self.resid_dropout(self.proj(out))


class ModalSandwichBlock(nn.Module):
    """One modal-token refinement block (operates on theta in R^{2K x D})."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int,
        k: int,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        cross_first: bool = True,
    ) -> None:
        super().__init__()
        self.k = int(k)
        self.cross_first = bool(cross_first)

        # theta token embedding/projection
        self.coef2hid = nn.Linear(input_dim, hidden_dim)
        self.hid2coef = nn.Linear(hidden_dim, input_dim)
        nn.init.zeros_(self.hid2coef.weight)
        nn.init.zeros_(self.hid2coef.bias)

        self.mode_pos = nn.Parameter(torch.zeros(1, 2 * self.k, hidden_dim))
        nn.init.normal_(self.mode_pos, mean=0.0, std=0.02)

        self.self_blk = TransformerBlock(
            hidden_dim,
            num_heads,
            mlp_ratio=4.0,
            dropout=dropout,
            attn_dropout=attn_dropout,
        )
        self.cross_blk = CrossAttnBlock(
            hidden_dim,
            num_heads,
            dropout=dropout,
            attn_dropout=attn_dropout,
        )

    def forward(
        self,
        theta: torch.Tensor,              # [B,2K,D]
        t_vec: torch.Tensor,              # [B,H]
        summary_kv_H: Optional[torch.Tensor] = None,  # [B,S,H]
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if theta.dim() != 3 or theta.shape[1] != 2 * self.k:
            raise ValueError(f"theta must be [B, {2*self.k}, D]")

        h = self.coef2hid(theta) + self.mode_pos + t_vec.unsqueeze(1)

        def apply_cross(x: torch.Tensor) -> torch.Tensor:
            if summary_kv_H is None:
                return x
            return self.cross_blk(x, summary_kv_H, t_vec=t_vec)

        if self.cross_first:
            h = apply_cross(h)
            h = self.self_blk(h, t_vec=t_vec, attn_bias=attn_bias)
        else:
            h = self.self_blk(h, t_vec=t_vec, attn_bias=attn_bias)
            h = apply_cross(h)

        return theta + self.hid2coef(h)


class LapFormer(nn.Module):
    """Modal-token LapFormer.

    Flow:
        1) Analyze x_time -> theta (modal residues) and effective poles (rho, omega).
        2) Refine theta via a stack of ModalSandwichBlocks (self_blk kept).
        3) Synthesize hat{z}_0 in time domain with residual refinement.

    The analysis step can optionally disable the expensive time cross-attention and
    fall back to a fast diagonal projection.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        laplace_k: int = 8,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        cross_first: bool = True,
        use_mlp_residual: bool = True,
        self_conditioning: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.self_conditioning = bool(self_conditioning)
        self.k = int(laplace_k)

        # Modal analysis/synthesis
        self.analysis = ModalPredictor(
            k=self.k,
            feat_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            cond_dim=hidden_dim,
            attn_dropout=attn_dropout,
        )
        self.synthesis = ModalSynthesizer(
            self.analysis,
            hidden_dim=hidden_dim,
            use_mlp_residual=use_mlp_residual,
        )

        # Optional self-conditioning in modal space: project sc_feat -> theta_sc
        self.sc_gate = nn.Parameter(torch.tensor(1.0)) if self.self_conditioning else None

        self.blocks = nn.ModuleList(
            [
                ModalSandwichBlock(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    k=self.k,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    cross_first=cross_first,
                )
                for _ in range(int(num_layers))
            ]
        )

        self.head_norm = nn.LayerNorm(input_dim)
        self.head_proj = nn.Linear(input_dim, input_dim)
        nn.init.zeros_(self.head_proj.weight)
        nn.init.zeros_(self.head_proj.bias)

    def forward(
        self,
        x_tokens: torch.Tensor,                 # [B,T,D]
        t_vec: torch.Tensor,                    # [B,H]
        cond_summary: Optional[torch.Tensor] = None,  # [B,S,H]
        sc_feat: Optional[torch.Tensor] = None,       # [B,T,D]
        dt: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, D = x_tokens.shape
        if t_vec.dim() != 2 or t_vec.shape[0] != B or t_vec.shape[1] != self.hidden_dim:
            raise ValueError(
                f"t_vec must be [B, hidden_dim]={B, self.hidden_dim}; got {tuple(t_vec.shape)}"
            )
        if cond_summary is not None:
            if cond_summary.dim() != 3 or cond_summary.shape[0] != B or cond_summary.shape[2] != self.hidden_dim:
                raise ValueError(
                    f"cond_summary must be [B,S,hidden_dim]={B,'S',self.hidden_dim}; got {tuple(cond_summary.shape)}"
                )

        # Compute poles once per forward (reused for x and optional self-conditioning)
        rho, omega = self.analysis.effective_poles(
            B,
            x_tokens.dtype,
            x_tokens.device,
            diffusion_time_emb=t_vec,
            history_context=cond_summary,
        )

        # Modal analysis: x_time -> theta
        theta, _, _, _ = self.analysis(
            x_tokens,
            dt=dt,
            t=t,
            diffusion_time_emb=t_vec,
            history_context=cond_summary,
            poles=(rho, omega),
            return_t_rel=False,
        )

        # Optional modal self-conditioning (project sc_feat onto SAME poles)
        if self.self_conditioning and sc_feat is not None:
            theta_sc, _, _, _ = self.analysis(
                sc_feat,
                dt=dt,
                t=t,
                diffusion_time_emb=t_vec,
                history_context=cond_summary,
                poles=(rho, omega),
                return_t_rel=False,
            )
            gate = torch.tanh(self.sc_gate)  # bounded scalar
            theta = theta + gate * theta_sc

        # Modal-token refinement stack
        for blk in self.blocks:
            theta = blk(theta, t_vec=t_vec, summary_kv_H=cond_summary, attn_bias=None)

        # Synthesis (parallel over all queried timestamps)
        y_time = self.synthesis(theta, rho=rho, omega=omega, dt=dt, t=t, target_T=T)
        return self.head_proj(self.head_norm(y_time))
