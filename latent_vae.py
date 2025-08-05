import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import math


class LearnableLaplacianBasis(nn.Module):
    def __init__(self, k: int, feat_dim: int, alpha_min: float = 1e-6):
        """
        Args:
            k: number of complex Laplacian basis elements
            feat_dim: feature dimension of x
            alpha_min: minimum decay rate (small positive)
        """
        super().__init__()
        self.k = k
        self.alpha_min = alpha_min

        # trainable poles α_raw, β
        self.s_real = nn.Parameter(torch.empty(k))
        self.s_imag = nn.Parameter(torch.empty(k))
        self.reset_parameters()

        # learned projection from feat_dim → k with spectral normalization
        self.proj = spectral_norm(
            nn.Linear(feat_dim, k, bias=True),
            n_power_iterations=1,
            eps=1e-6
        )

    def reset_parameters(self):
        nn.init.uniform_(self.s_real, a=0.0, b=0.1)
        nn.init.uniform_(self.s_imag, a=-math.pi, b=math.pi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, feat_dim)
        Returns:
            lap_feats: (B, T, 2*k)
        """
        B, T, _ = x.shape
        device, dtype = x.device, x.dtype

        # build decaying/oscillatory poles
        alpha = F.softplus(self.s_real) + self.alpha_min
        beta = self.s_imag
        s = torch.complex(-alpha, beta)

        # time indices 0…T-1 → (T,1) cast to complex
        t_idx = torch.arange(T, device=device, dtype=dtype).unsqueeze(1).to(s.dtype)

        # compute Laplace basis kernels → (T, k)
        expo = torch.exp(t_idx * s.unsqueeze(0))
        re_basis, im_basis = expo.real, expo.imag

        # project features → (B, T, k)
        proj_feats = self.proj(x)

        # modulate and concat real+imag → (B, T, 2*k)
        real_proj = proj_feats * re_basis.unsqueeze(0)
        imag_proj = proj_feats * im_basis.unsqueeze(0)
        return torch.cat([real_proj, imag_proj], dim=2)


class LearnableInverseLaplacianBasis(nn.Module):
    def __init__(self, laplace_basis: LearnableLaplacianBasis):
        """
        Learnable inverse map (not strict inverse) from Laplace features → input space.
        """
        super().__init__()
        feat_dim = laplace_basis.proj.in_features
        self.inv_proj = spectral_norm(
            nn.Linear(2 * laplace_basis.k, feat_dim, bias=True),
            n_power_iterations=1,
            eps=1e-6
        )

    def forward(self, lap_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lap_feats: (B, T, 2*k)
        Returns:
            x_hat: (B, T, feat_dim)
        """
        return self.inv_proj(lap_feats)


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, seq_len, latent_dim, num_layers=2, num_heads=4, ff_dim=256, k=16):
        super().__init__()
        self.pos_emb     = nn.Parameter(torch.randn(1, seq_len, latent_dim))
        self.input_proj  = nn.Linear(input_dim, latent_dim)

        # stack of Transformer‐encoder layers (d_model=2*latent_dim after concat)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=latent_dim,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                activation='gelu',
                batch_first=True
            )
            for _ in range(num_layers)
        ])

        # learnable Laplacian basis + projection into latent_dim
        self.lap_feat = LearnableLaplacianBasis(k=k)
        self.lap_proj = nn.Linear(2 * k, latent_dim)
        self.joint_proj = nn.Linear(2 * latent_dim, latent_dim)

    def forward(self, x):
        # x: [B, T, input_dim]
        h = self.input_proj(x)    # [B, T, latent_dim]
        lap = self.lap_proj(self.lap_feat(x))      # [B, T, latent_dim]
        h = torch.cat((h, lap), dim=-1)            # [B, T, 2*latent_dim]
        h = self.joint_proj(h) + self.pos_emb #[B, T, latent_dim]
        skips = []
        for layer in self.layers:
            h = layer(h)
            skips.append(h)

        return skips


class TransformerDecoder(nn.Module):
    def __init__(self, latent_dim, seq_len, num_layers=2, num_heads=4, ff_dim=256, input_dim=None):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, latent_dim))

        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=latent_dim,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                activation='gelu',
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(latent_dim, input_dim or latent_dim)

    def forward(self, z, encoder_skips):
        """
        z:             [B, T, D]   sampled latent
        encoder_skips: list[L] of [B, T, 2*D] from encoder
        """
        h = z + self.pos_emb
        memory = encoder_skips[-1]

        # cross‐attn to the final encoder layer, with residual skips
        for dec_layer, skip in zip(self.layers, reversed(encoder_skips)):
            h = dec_layer(tgt=h, memory=memory)
            h = h + skip

        return self.out_proj(h)


class LatentVAE(nn.Module):
    def __init__(self, input_dim, seq_len, latent_dim,
                 enc_layers=2, enc_heads=4, enc_ff=256,
                 dec_layers=2, dec_heads=4, dec_ff=256, lap_k=8):
        super().__init__()
        # encoder returns list of skip‐connections
        self.encoder = TransformerEncoder(
            input_dim, seq_len, latent_dim,
            num_layers=enc_layers, num_heads=enc_heads, ff_dim=enc_ff, k=lap_k
        )

        # VAE heads
        self.mu_head     = nn.Linear(latent_dim, latent_dim)
        self.logvar_head = nn.Linear(latent_dim, latent_dim)

        # decoder consumes sampled z + skips
        self.decoder = TransformerDecoder(
            latent_dim, seq_len,
            num_layers=dec_layers, num_heads=dec_heads, ff_dim=dec_ff,
            input_dim=input_dim
        )

        # batch‐norm over latent channels
        self.latent_bn = nn.BatchNorm1d(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # encode
        encoder_skips = self.encoder(x)      # List of [B, T, 2*latent_dim]
        z0 = encoder_skips[-1]               # final encoder output

        # batch‐norm across feature dim
        z_perm = z0.permute(0, 2, 1)         # [B, D, T]
        z_norm = self.latent_bn(z_perm)
        z = z_norm.permute(0, 2, 1)          # [B, T, D]

        # VAE bottleneck
        mu     = self.mu_head(z)
        logvar = self.logvar_head(z)
        z_sample = self.reparameterize(mu, logvar)

        # decode
        x_hat = self.decoder(z_sample, encoder_skips)
        return x_hat, mu, logvar


