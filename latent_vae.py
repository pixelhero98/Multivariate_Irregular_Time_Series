import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x


class TransformerVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        d_model: int = 128,
        latent_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        dim_ff: int = 512,
        dropout: float = 0.1,
        downsample_stride: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.downsample_stride = downsample_stride
        self.seq_len = seq_len

        # MLP downsample: flatten patches of length stride
        self.mlp_down = nn.Linear(downsample_stride * input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Latent projections
        self.fc_mu = nn.Linear(d_model, latent_dim)
        self.fc_logvar = nn.Linear(d_model, latent_dim)

        # Latent to sequence
        self.fc_z_to_seq = nn.Linear(latent_dim, d_model)

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation='gelu'
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )

        # MLP upsample: map back to patches
        self.mlp_up = nn.Linear(d_model, downsample_stride * input_dim)

    def encode(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        # x: (batch, T, D)
        batch, T, D = x.size()
        # ensure divisible
        assert T % self.downsample_stride == 0, "Sequence length must be divisible by stride"
        T_down = T // self.downsample_stride
        # reshape into patches
        x_patches = x.view(batch, T_down, self.downsample_stride * D)
        # MLP projection
        h = self.mlp_down(x_patches)  # (batch, T_down, d_model)
        h = self.pos_enc(h)
        h = h.transpose(0, 1)  # (T_down, batch, d_model)
        h = self.transformer_encoder(h)
        h = h.transpose(0, 1)  # (batch, T_down, d_model)
        # Pool over time
        h_pool = h.mean(dim=1)
        mu = self.fc_mu(h_pool)
        logvar = self.fc_logvar(h_pool)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # z: (batch, latent_dim)
        batch = z.size(0)
        T = self.seq_len
        D = self.input_dim
        T_down = T // self.downsample_stride
        # project z to sequence embeddings
        h_seq = self.fc_z_to_seq(z)  # (batch, d_model)
        h_seq = h_seq.unsqueeze(1).repeat(1, T_down, 1)
        h_seq = self.pos_enc(h_seq)
        tgt = h_seq.transpose(0, 1)
        memory = tgt
        decoded = self.transformer_decoder(tgt, memory)
        decoded = decoded.transpose(0, 1)  # (batch, T_down, d_model)
        # MLP upsample
        x_patches = self.mlp_up(decoded)  # (batch, T_down, stride*D)
        x_rec = x_patches.view(batch, T, D)
        return x_rec

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_rec = self.decode(z)
        return x_rec, mu, logvar


# Standalone loss function for reuse across models
def vae_loss(x: torch.Tensor, x_rec: torch.Tensor,
              mu: torch.Tensor, logvar: torch.Tensor,
              beta: float = 1.0) -> torch.Tensor:
    """
    Compute VAE loss as reconstruction MSE plus beta-weighted KL divergence.
    """
    recon_loss = F.mse_loss(x_rec, x, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss
