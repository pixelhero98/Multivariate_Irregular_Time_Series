from typing import List, Optional, Union

import torch
import torch.nn as nn

# Support both repository layouts
try:
    from Model.laptrans import LaplaceTransformEncoder, LaplacePseudoInverse
except Exception:
    from laptrans import LaplaceTransformEncoder, LaplacePseudoInverse


class AdaLayerNorm(nn.Module):
    """LayerNorm with feature-wise affine parameters conditioned on input."""

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
        """Apply normalisation followed by conditioned scale and bias."""
        h = self.norm(x)
        if c is None:
            scale = torch.zeros(h.size(0), h.size(-1), device=h.device, dtype=h.dtype)
            bias = torch.zeros_like(scale)
        else:
            scale, bias = self.to_ss(c).chunk(2, dim=-1)
        return (1 + scale).unsqueeze(1) * h + bias.unsqueeze(1)


class TransformerBlock(nn.Module):
    """Self-attention block with AdaLayerNorm conditioning."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ) -> None:
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
            nn.Dropout(dropout)
        )
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
        t_vec: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        batch, seq_len, hidden = x.shape
        h = self.norm1(x, t_vec)
        qkv = (
            self.qkv(h)
            .reshape(batch, seq_len, 3, self.num_heads, hidden // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (k.shape[-1] ** 0.5)

        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask
        if attn_bias is not None:
            attn_scores = attn_scores + attn_bias
        if key_padding_mask is not None:
            big_neg = torch.finfo(attn_scores.dtype).min
            attn_scores = attn_scores.masked_fill(
                key_padding_mask[:, None, None, :], big_neg
            )

        attn = torch.softmax(attn_scores, dim=-1)
        attn = self.attn_dropout(attn)
        attn_out = torch.matmul(attn, v).transpose(1, 2).reshape(batch, seq_len, hidden)
        x = x + self.resid_dropout(self.proj(attn_out))
        x = x + self.mlp(self.norm2(x, t_vec))

        return x


class CrossAttnBlock(nn.Module):
    """Cross-attention block allowing the sequence to attend to context tokens."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
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
        
        batch, seq_len, hidden = x_q.shape
        xq = self.norm_q(x_q, t_vec)
        xkv = self.norm_kv(x_kv, t_vec)
        
        q = self.q(xq).reshape(batch, seq_len, self.num_heads, hidden // self.num_heads).transpose(1, 2)
        kv = self.kv(xkv).reshape(batch, x_kv.shape[1], 2, self.num_heads, hidden // self.num_heads)
        k = kv[:, :, 0].transpose(1, 2)
        v = kv[:, :, 1].transpose(1, 2)
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (k.shape[-1] ** 0.5)
        attn = torch.softmax(attn_scores, dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(batch, seq_len, hidden)

        return x_q + self.resid_dropout(self.proj(out))


class LaplaceSandwichBlock(nn.Module):
    """
    Bridge between Laplace coefficients and transformer style processing.
    """

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
        self.cross_first = cross_first
        self.k = k

        # Analysis with adaptive ridge
        self.analysis = LaplaceTransformEncoder(
            k=k,
            feat_dim=input_dim,
            cond_dim=2 * hidden_dim,
            ridge_lambda=1e-3,
            adaptive_ridge=True,
        )
        self.synthesis = LaplacePseudoInverse(self.analysis, hidden_dim=hidden_dim)

        # Coefficient-token processing (2K tokens)
        self.coef2hid = nn.Linear(input_dim, hidden_dim)
        self.hid2coef = nn.Linear(hidden_dim, input_dim)
        nn.init.zeros_(self.hid2coef.weight)
        nn.init.zeros_(self.hid2coef.bias)

        # Learned positional embedding over the modal token index
        self.mode_pos = nn.Parameter(torch.zeros(1, 2 * k, hidden_dim))

        self.self_blk = TransformerBlock(
            hidden_dim,
            num_heads,
            mlp_ratio=4.0,
            dropout=dropout,
            attn_dropout=attn_dropout
        )
        self.cross_blk = CrossAttnBlock(
            hidden_dim,
            num_heads,
            dropout=dropout,
            attn_dropout=attn_dropout
        )

    def forward(
        self,
        x_time: torch.Tensor,
        t_vec: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
        summary_kv_H: Optional[torch.Tensor] = None,
        sc_feat: Optional[torch.Tensor] = None,
        dt: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x_time:     [B, T, D_in]
            t_vec:      [B, H]
            sc_feat:    [B, T, D_in] - Self conditioning guess in data space
            dt:         [B, T] or [B, T, 1]
        """

        # Conditioning: concat(diffusion embedding, pooled summary).
        if summary_kv_H is None:
            summary_pool = torch.zeros_like(t_vec)
        else:
            summary_pool = summary_kv_H.mean(dim=1)
        cond_vec = torch.cat([t_vec, summary_pool], dim=-1)

        # 1. Modal projection: x_time -> Theta [B, 2K, D]
        # Uses adaptive ridge regression driven by cond_vec (noise level).
        theta, basis, _, _ = self.analysis(x_time, dt=dt, cond=cond_vec, return_basis=True)

        # 2. Embed coefficient tokens
        h = self.coef2hid(theta) + self.mode_pos + t_vec.unsqueeze(1)

        # 3. Modal Self-Conditioning:
        # Instead of projecting the mean of sc_feat, we project sc_feat onto the
        # SAME basis to get modal hints, preserving temporal structure.
        if sc_feat is not None:
            # Note: We reuse the same analysis module. Ideally, we would reuse 'basis'
            # directly to avoid re-computation, but calling forward is cleaner for integration.
            # cond_vec ensures the same poles/basis are generated.
            theta_sc = self.analysis(sc_feat, dt=dt, cond=cond_vec, return_basis=False)
            h = h + self.coef2hid(theta_sc)

        def apply_cross(block_input: torch.Tensor) -> torch.Tensor:
            if summary_kv_H is None:
                return block_input
            return self.cross_blk(block_input, summary_kv_H, t_vec=t_vec)

        # 4. Mixing
        if self.cross_first:
            h = apply_cross(h)
            h = self.self_blk(h, attn_mask=None, key_padding_mask=None, attn_bias=attn_bias, t_vec=t_vec)
        else:
            h = self.self_blk(h, attn_mask=None, key_padding_mask=None, attn_bias=attn_bias, t_vec=t_vec)
            h = apply_cross(h)

        # 5. Synthesis: Update modal residues and synthesize back to time domain.
        theta_upd = theta + self.hid2coef(h)
        return self.synthesis(theta_upd, basis)


class LapFormer(nn.Module):
    """Full LapFormer model composed of multiple Laplace sandwich blocks."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        laplace_k: Union[int, List[int]] = 8,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        self_conditioning: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.self_conditioning = self_conditioning

        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        ks = [laplace_k] * num_layers if isinstance(laplace_k, int) else laplace_k
        if len(ks) != num_layers:
            raise ValueError("laplace_k list must match num_layers")

        self.blocks = nn.ModuleList(
            [
                LaplaceSandwichBlock(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    k=k,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                )
                for k in ks
            ]
        )

        self.head_norm = nn.LayerNorm(input_dim)
        self.head_proj = nn.Linear(input_dim, input_dim)
        nn.init.zeros_(self.head_proj.weight)
        nn.init.zeros_(self.head_proj.bias)

    def forward(
        self,
        x_tokens: torch.Tensor,
        t_vec: torch.Tensor,
        cond_summary: Optional[torch.Tensor] = None,
        sc_feat: Optional[torch.Tensor] = None,
        dt: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Execute the full LapFormer stack.

        Args:
            x_tokens:       [B, T, D]
            t_vec:          [B, H]
            cond_summary:   [B, S, H]
            sc_feat:        [B, T, D] (Self-conditioning guess)
            dt:             [B, T] or [B, T, 1]
        """
        batch, seq_len, _ = x_tokens.shape
        if t_vec.dim() != 2 or t_vec.shape[0] != batch or t_vec.shape[1] != self.hidden_dim:
            raise ValueError(f"t_vec must be [B, hidden_dim]")
            
        if cond_summary is not None:
            if cond_summary.dim() != 3 or cond_summary.shape[2] != self.hidden_dim:
                raise ValueError("cond_summary dimension mismatch")

        h_time = x_tokens
        
        # Pass self-conditioning feature (sc_feat) explicitly to each block.
        # The block handles the projection onto the basis.
        sc_input = sc_feat if (self.self_conditioning and sc_feat is not None) else None
        
        for blk in self.blocks:
            h_time = blk(
                h_time,
                t_vec,
                attn_bias=None,
                summary_kv_H=cond_summary,
                sc_feat=sc_input,
                dt=dt,
            )

        return self.head_proj(self.head_norm(h_time))
