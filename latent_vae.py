import torch
import torch.nn as nn
import torch.nn.functional as F

# Sparse Top-K Graph Transformer Layer
class SparseGraphTransformerLayer(nn.Module):
    def __init__(self, dim, num_heads=8, k=16):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.k = k
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        B, N, D = x.shape
        residual = x
        x_norm = self.norm1(x)
        Q = self.q_proj(x_norm).view(B, N, self.num_heads, self.head_dim)
        K = self.k_proj(x_norm).view(B, N, self.num_heads, self.head_dim)
        V = self.v_proj(x_norm).view(B, N, self.num_heads, self.head_dim)
        logits = torch.einsum('bnhd,bmhd->bhnm', Q, K) * self.scale
        topk_vals, topk_idx = logits.topk(self.k, dim=-1)
        mask = torch.full_like(logits, float('-inf'))
        mask.scatter_(-1, topk_idx, topk_vals)
        attn = torch.softmax(mask, dim=-1)
        out = torch.einsum('bhnm,bmhd->bnhd', attn, V).contiguous().view(B, N, D)
        out = self.out_proj(out)
        x = residual + out
        return x + self.ff(self.norm2(x))

# Transformer-based Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, seq_len, latent_dim, num_layers=2, num_heads=4, ff_dim=256):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, latent_dim))
        self.input_proj = nn.Linear(input_dim, latent_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x):
        h = self.input_proj(x) + self.pos_emb  # [B, T, D]
        return self.encoder(h)

# Transformer-based Decoder
class TransformerDecoder(nn.Module):
    def __init__(self, latent_dim, seq_len, num_layers=2, num_heads=4, ff_dim=256, input_dim=None):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, latent_dim))
        layer = nn.TransformerDecoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            activation='gelu',
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        # Final projection back to original feature space
        self.out_proj = nn.Linear(latent_dim, input_dim or latent_dim)

    def forward(self, z):
        # z: [B, T, D]
        tgt = z + self.pos_emb
        # Self-attend with memory equal to z itself
        dec = self.decoder(tgt, memory=z)
        return self.out_proj(dec)

# VAE with GT Latent Processor and Symmetric Transformer Decoder
class VAEWithTransformerDecoder(nn.Module):
    def __init__(self, input_dim, seq_len, latent_dim,
                 enc_layers=2, enc_heads=4, enc_ff=256,
                 gt_layers=2, gt_heads=8, gt_k=8,
                 dec_layers=2, dec_heads=4, dec_ff=256):
        super().__init__()
        # Encoder
        self.encoder = TransformerEncoder(
            input_dim, seq_len, latent_dim,
            num_layers=enc_layers, num_heads=enc_heads, ff_dim=enc_ff
        )
        # GT latent processor
        self.gt_layers = nn.ModuleList([
            SparseGraphTransformerLayer(latent_dim, num_heads=gt_heads, k=gt_k)
            for _ in range(gt_layers)
        ])
        # VAE heads
        self.mu_head = nn.Linear(latent_dim, latent_dim)
        self.logvar_head = nn.Linear(latent_dim, latent_dim)
        # Decoder
        self.decoder = TransformerDecoder(
            latent_dim, seq_len,
            num_layers=dec_layers, num_heads=dec_heads, ff_dim=dec_ff,
            input_dim=input_dim
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def process_latent(self, z):
        for layer in self.gt_layers:
            z = z + layer(z)
        return z

    def forward(self, x):
        # x: [B, T, F]
        z = self.encoder(x)                 # [B, T, D]
        z = self.process_latent(z)         # GT processing
        mu = self.mu_head(z)
        logvar = self.logvar_head(z)
        z_sample = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z_sample)     # [B, T, F]
        return x_hat, mu, logvar
