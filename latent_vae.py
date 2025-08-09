import torch
import torch.nn as nn


class TransformerDecoderBlock(nn.Module):
    """
    A decoder block with self-attention and optional cross-attention for skip connections.
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.0):
        super().__init__()
        # Standard PyTorch layers for self-attention and cross-attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # Feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )

        # Normalization and Dropout layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, skip=None):
        # 1. Self-Attention (with pre-norm and residual connection)
        x = x + self.dropout(self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x))[0])

        # 2. Conditionally perform Cross-Attention if a skip connection is provided
        if skip is not None:
            x = x + self.dropout(self.cross_attn(self.norm2(x), skip, skip)[0])

        # 3. Feed-Forward Network (with pre-norm and residual connection)
        x = x + self.dropout(self.mlp(self.norm3(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, seq_len, latent_dim, num_layers=2, num_heads=4, ff_dim=256):
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

        self.joint_proj = nn.Linear(latent_dim, latent_dim)

    def forward(self, x):
        # x: [B, T, input_dim]
        h = self.input_proj(x)    # [B, T, latent_dim]
        h = self.joint_proj(h) + self.pos_emb[:, :x.size(1), :] #[B, T, latent_dim]
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
            TransformerDecoderBlock(
                d_model=latent_dim,
                nhead=num_heads,
                dim_feedforward=ff_dim
            )
            for _ in range(num_layers)
        ])

        self.out_proj = nn.Linear(latent_dim, input_dim)

    def forward(self, z, encoder_skips=None):
        """
        z:             [B, T, D]   sampled latent
        encoder_skips: list[L] of [B, T, 2*D] from encoder
        """
        h = z + self.pos_emb[:, :z.size(1), :]
        if encoder_skips is not None:
            skips_reversed = reversed(encoder_skips)
        else:
            skips_reversed = [None] * len(self.layers)

        # cross‐attn to the final encoder layer, with residual skips
        for dec_layer, skip in zip(self.layers, skips_reversed):
            h = dec_layer(h, skip=skip)
            h = h

        h = self.out_proj(h)

        return h


class LatentVAE(nn.Module):
    def __init__(self, input_dim, seq_len, latent_dim,
                 enc_layers=2, enc_heads=4, enc_ff=256,
                 dec_layers=2, dec_heads=4, dec_ff=256):
        super().__init__()
        # encoder returns list of skip‐connections
        self.encoder = TransformerEncoder(
            input_dim, seq_len, latent_dim,
            num_layers=enc_layers, num_heads=enc_heads, ff_dim=enc_ff
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
        z = encoder_skips[-1]               # final encoder output

        # batch‐norm across feature dim
        # z_perm = z0.permute(0, 2, 1)         # [B, D, T]
        # z_norm = self.latent_bn(z_perm)
        # z = z_norm.permute(0, 2, 1)          # [B, T, D]

        # VAE bottleneck
        mu     = self.mu_head(z)
        logvar = self.logvar_head(z)
        z_sample = self.reparameterize(mu, logvar)

        # decode
        x_hat = self.decoder(z_sample)
        return x_hat, mu, logvar
