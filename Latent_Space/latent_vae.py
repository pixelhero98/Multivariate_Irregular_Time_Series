import torch
import torch.nn as nn


class TransformerDecoderBlock(nn.Module):
    """
    A decoder block with self-attention and optional cross-attention for skip connections.
    """

    def __init__(self, d_model: int, enc_dim: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
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
        self.skip_proj = nn.Linear(enc_dim, d_model)

    def forward(self, x, skip=None):
        x_norm = self.norm1(x)
        # 1. Self-Attention (with pre-norm and residual connection)
        x = x + self.dropout(self.self_attn(x_norm, x_norm, x_norm)[0])

        # 2. Conditionally perform Cross-Attention if a skip connection is provided
        if skip is not None:
            skip = self.skip_proj(skip)
            x = x + self.dropout(self.cross_attn(self.norm2(x), skip, skip)[0])

        # 3. Feed-Forward Network (with pre-norm and residual connection)
        x = x + self.dropout(self.mlp(self.norm3(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, seq_len, latent_dim,
                 num_layers=2, num_heads=4, ff_dim=256, dropout=0.1):
        super().__init__()
        self.pos_emb    = nn.Parameter(torch.randn(1, seq_len, latent_dim))
        self.input_proj = nn.Linear(input_dim, latent_dim)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=latent_dim, nhead=num_heads,
                dim_feedforward=ff_dim, dropout=dropout,
                activation='gelu', batch_first=True, norm_first=True
            ) for _ in range(num_layers)
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

        return skips # return List of [B, T, latent_dim] skip tensors


class TransformerDecoder(nn.Module):
    def __init__(self, latent_dim, enc_dim, seq_len, num_layers=2, num_heads=4, ff_dim=256,
                 input_dim=None, dropout=0.1):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, latent_dim))
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(latent_dim, enc_dim, num_heads, ff_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(latent_dim, input_dim)

    def forward(self, z, encoder_skips=None):
        """
        z:             [B, T, D]   sampled latent
        encoder_skips: list of [B, T, D] from encoder (last to first)
        """
        h = z + self.pos_emb[:, :z.size(1), :]
        skips = list(reversed(encoder_skips)) if encoder_skips is not None else [None] * len(self.layers)
        for i, dec_layer in enumerate(self.layers):
            skip = skips[i] if i < len(skips) else None
            h = dec_layer(h, skip=skip)

        h = self.out_proj(h)

        return h


class LatentVAE(nn.Module):
    def __init__(self, input_dim, seq_len, latent_dim, latent_channel,
                 enc_layers=2, enc_heads=4, enc_ff=256,
                 dec_layers=2, dec_heads=4, dec_ff=256):
        super().__init__()
        # encoder returns list of skip‐connections
        self.encoder = TransformerEncoder(
            input_dim, seq_len, latent_dim,
            num_layers=enc_layers, num_heads=enc_heads, ff_dim=enc_ff
        )
        # VAE heads
        self.mu_head     = nn.Linear(latent_dim, latent_channel)
        self.logvar_head = nn.Linear(latent_dim, latent_channel)

        # decoder consumes sampled z + skips
        self.decoder = TransformerDecoder(
            latent_channel, latent_dim, seq_len,
            num_layers=dec_layers, num_heads=dec_heads, ff_dim=dec_ff,
            input_dim=input_dim
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # encode
        encoder_hidden_states = self.encoder(x)     # list of [B, T, D]
        z_last = encoder_hidden_states[-1]          # [B, T, D]
    
        # VAE bottleneck
        mu     = self.mu_head(z_last)               # [B, T, D]
        logvar = self.logvar_head(z_last).clamp(min=-10.0, max=10.0)         # [B, T, D]
        z_sample = self.reparameterize(mu, logvar)  # [B, T, D]
    
        # decode with skips (use encoder_hidden_states)
        x_hat = self.decoder(z_sample, encoder_skips=encoder_hidden_states)
        return x_hat, mu, logvar
