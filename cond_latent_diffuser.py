import torch
import torch.nn as nn
from latent_vae import LearnableLaplacianBasis, LearnableInverseLaplacianBasis
from cond_diffusion_utils import NoiseScheduler

def get_sinusoidal_pos_emb(L: int, dim: int, device: torch.device) -> torch.Tensor:
    """
    Create a [1, L, dim] sinusoidal positional embedding.
    """
    # position indices (L,1) and dimension indices (1,dim)
    pos = torch.arange(L, device=device).unsqueeze(1).float()
    i   = torch.arange(dim, device=device).unsqueeze(0).float()
    # compute the angles: 1/10000^(2*(i//2)/dim)
    angle_rates = 1 / (10000 ** (2 * (i // 2) / dim))
    angles = pos * angle_rates  # (L, dim)

    pe = torch.zeros(L, dim, device=device)
    pe[:, 0::2] = torch.sin(angles[:, 0::2])
    pe[:, 1::2] = torch.cos(angles[:, 1::2])
    return pe.unsqueeze(0)  # (1, L, dim)


class LapFormer(nn.Module):
    """
    Transformer block with:
      - Laplace transform → Norm → Self‑Attention → Cross‑Attention
      - Norm → Learned Inverse → Residual‑Lap skip → MLP
    """
    def __init__(self, dim, num_heads, k, cond_dim=None, mlp_dim=None):
        super().__init__()
        # Frequency-domain modules
        self.Lap = LearnableLaplacianBasis(k, dim)
        self.norm_lap = nn.LayerNorm(2 * k)        # before self-attn
        self.self_attn = nn.MultiheadAttention(embed_dim=2 * k,
                                               num_heads=num_heads,
                                               batch_first=True)

        if cond_dim is not None:
            self.cross_attn = nn.MultiheadAttention(embed_dim=2 * k,
                                                    num_heads=num_heads,
                                                    batch_first=True)
            self.norm_cross = nn.LayerNorm(2 * k)
        else:
            self.cross_attn = None

        self.norm_inv = nn.LayerNorm(2 * k)        # before inverse
        self.InvLap = LearnableInverseLaplacianBasis(self.Lap)

        # Time-domain MLP
        mlp_dim = mlp_dim or 4 * dim
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim)
        )

    def forward(self, x, cond=None):
        # 1) Laplace → Norm → Self‑Attention
        lap = self.Lap(x)                       # → (B, T, 2k)
        h_lap = self.norm_lap(lap)
        lap = lap + self.self_attn(h_lap, h_lap, h_lap)[0]

        # 2) (Optional) Cross‑Attention
        if cond is not None and self.cross_attn:
            h2 = self.norm_cross(lap)
            lap = lap + self.cross_attn(h2, cond, cond)[0]

        # 3) Norm → Inverse Laplace
        lap_normed = self.norm_inv(lap)
        x_rec = self.InvLap(lap_normed)        # → (B, T, dim)

        # 4) Residual “Lap skip”
        x_res = x + x_rec

        # 5) MLP + residual
        h3 = self.norm2(x_res)
        return x_res + self.mlp(h3)


class LapDiT(nn.Module):
    """
    Conditional diffusion model using a transformer backbone.
    """

    def __init__(
            self,
            latent_dim: int,
            time_embed_dim: int,
            lap_kernel: int = 32,
            cond_embed_dim: int = 64,
            num_layers: int = 4,
            num_heads: int = 8,
            mlp_dim: int = None,
            max_timesteps: int = 1000,
            num_classes: int = None
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.GELU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )

        max_L = 512

        # Optional class embedding
        self.class_embed = nn.Embedding(num_classes, cond_embed_dim) if num_classes else None

        # Regression scalar embedding
        self.scalar_embed = nn.Linear(1, cond_embed_dim)

        # Masked time-series embedding (project per-feature)
        # Assume input series shape (B, L, F)
        self.series_proj = nn.Linear(latent_dim, cond_embed_dim)

        # Combined hidden dimension (latent + time)
        hidden_dim = latent_dim + time_embed_dim
        self.hidden_dim = hidden_dim

        # Project and normalize condition embeddings to hidden dimension
        self.cond_proj = nn.Linear(cond_embed_dim, hidden_dim)
        self.cond_norm = nn.LayerNorm(hidden_dim)
        # Project hidden conditions into Lap domain for cross-attention
        self.cond_to_lap = nn.Linear(hidden_dim, 2 * lap_kernel)
        self.k = lap_kernel
        # Build transformer layers
        self.layers = nn.ModuleList([
            LapFormer(
                dim=hidden_dim,
                num_heads=num_heads,
                k=lap_kernel,
                cond_dim=2 * lap_kernel,  # now matches LapFormer cross-attn embedding
                mlp_dim=mlp_dim
            ) for _ in range(num_layers)
        ])

        self.register_buffer(
            'pos_table',
            get_sinusoidal_pos_emb(max_L, hidden_dim, torch.device('cpu'))
        )

        # Project back to latent space
        self.output_proj = nn.Linear(hidden_dim, latent_dim)

        # Noise scheduler
        self.scheduler = NoiseScheduler(timesteps=max_timesteps)

    def forward(self, x_latent, t, class_labels=None, series=None, scalars=None):
        """
        x_latent: [B, L, latent_dim]
        t: [B] integer timesteps
        class_labels: [B] (optional)
        series: [B, L, latent_dim] (optional)
        scalars: [B] (optional)
        """
        B, L, _ = x_latent.shape
        device = x_latent.device
        # Time embedding
        t_norm = t.float() / self.scheduler.timesteps
        t_emb = self.time_embed(t_norm.view(-1, 1))  # [B, time_embed_dim]
        t_emb = t_emb.unsqueeze(1).repeat(1, L, 1)  # [B, L, time_embed_dim]

        # Prepare conditioning tensor
        conds = []
        if class_labels is not None and self.class_embed:
            conds.append(self.class_embed(class_labels).unsqueeze(1))  # [B,1,cond_dim]
        if series is not None:
            conds.append(self.series_proj(series))  # [B,L,cond_dim]
        if scalars is not None:
            conds.append(self.scalar_embed(scalars.view(-1, 1)).unsqueeze(1))  # [B,1,cond_dim]
        cond_tensor = torch.cat(conds, dim=1) if conds else None  # [B, C', cond_dim]

        # Lift cond embeddings to hidden size
        if cond_tensor is not None:
            # Lift cond embeddings to hidden size
            cond_tensor = self.cond_proj(cond_tensor)  # [B, C', hidden_dim]
            cond_tensor = self.cond_norm(cond_tensor)
            # Project into Lap domain for cross-attention
            cond_tensor = self.cond_to_lap(cond_tensor)  # [B, C', 2*k]
            # Shape assertion for safety
            assert cond_tensor.shape[-1] == 2 * self.k, \
                f"Expected cond last dim {2 * self.k}, got {cond_tensor.shape[-1]}"

        # Input projection + concat time embedding
        h = torch.cat([x_latent, t_emb], dim=-1)  # [B, L, hidden_dim]
        # Add sinusoidal positional embedding
        pos_emb = self.pos_table[:, :L, :].to(device)  # (1,L,hidden_dim)
        h = h + pos_emb

        # Pass through transformer layers
        for layer in self.layers:
            h = layer(h, cond=cond_tensor)

        # Project to predict noise
        out = self.output_proj(h)
        # Predict noise of shape [B,L,latent_dim]
        return out



