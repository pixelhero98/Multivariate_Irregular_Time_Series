import torch
import torch.nn as nn
from torch import Tensor


# Sparse Top-K Graph Transformer Layer
class SparseGraphTransformerLayer(nn.Module):
    """
    Sparse Top-K Graph Transformer Layer with temporal relative bias, pre-norm, dropout, and FF dropout.
    Args:
        dim: input and output feature dimension.
        num_heads: number of attention heads.
        k: top-K sparse connections per query.
        dropout: dropout probability for attention and feed-forward.
        max_seq_len: maximum sequence length for relative time embeddings.
    """
    def __init__(self, dim: int, num_heads: int = 8, k: int = 16,
                 dropout: float = 0.1, max_seq_len: int = 512):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.k = k
        self.max_seq_len = max_seq_len

        # Projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(dim, dim)

        # Layer Norms
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Feed-Forward
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

        # Relative temporal bias embeddings
        # indices range from -(max_seq_len-1) to +(max_seq_len-1), offset by max_seq_len-1
        num_rel = 2 * max_seq_len - 1
        self.rel_time_emb = nn.Embedding(num_rel, num_heads)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, N, D]
        B, N, D = x.shape
        # Pre-norm
        x_norm = self.norm1(x)
        # Project Q, K, V
        Q = self.q_proj(x_norm).view(B, N, self.num_heads, self.head_dim)
        K = self.k_proj(x_norm).view(B, N, self.num_heads, self.head_dim)
        V = self.v_proj(x_norm).view(B, N, self.num_heads, self.head_dim)
        # Compute raw attention logits [B, heads, N, N]
        logits = torch.einsum('bnhd,bmhd->bhnm', Q, K) * self.scale

        # Add relative temporal bias
        pos = torch.arange(N, device=x.device)
        # shape [N, N], values in [-(N-1)...+(N-1)]
        d = pos[None, :] - pos[:, None]
        # shift to [0...(2*max_seq_len-2)] for embedding lookup
        d = d + (self.max_seq_len - 1)
        # embed and reshape to [1, heads, N, N]
        rel_bias = self.rel_time_emb(d)               # [N, N, heads]
        rel_bias = rel_bias.permute(2, 0, 1).unsqueeze(0)
        logits = logits + rel_bias

        # Top-K sparsity
        topk_vals, topk_idx = logits.topk(self.k, dim=-1)
        sparse_logits = torch.full_like(logits, float('-inf'))
        sparse_logits.scatter_(-1, topk_idx, topk_vals)

        # Softmax + dropout
        attn = torch.softmax(sparse_logits, dim=-1)
        attn = self.attn_dropout(attn)

        # Attention output
        out = torch.einsum('bhnm,bmhd->bnhd', attn, V).contiguous().view(B, N, D)
        out = self.out_proj(out)
        x = x + out  # residual connection

        # Feed-forward
        x_norm = self.norm2(x)
        ff_out = self.ff(x_norm)
        return x + ff_out

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
    """
    Transformer-based Decoder with causal masking and learnable positional embeddings.
    """
    def __init__(self, latent_dim: int, seq_len: int,
                 num_layers: int = 2, num_heads: int = 4, ff_dim: int = 256,
                 dropout: float = 0.1, input_dim: int = None):
        super().__init__()
        self.seq_len = seq_len
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, latent_dim))
        layer = nn.TransformerDecoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.out_proj = nn.Linear(latent_dim, input_dim or latent_dim)

    def _generate_causal_mask(self, T: int, device) -> Tensor:
        mask = torch.triu(torch.full((T, T), float('-inf'), device=device), diagonal=1)
        return mask

    def forward(self, z: Tensor) -> Tensor:
        """
        z: [B, T, latent_dim]
        returns: [B, T, input_dim]
        """
        B, T, D = z.size()
        assert T <= self.seq_len, "Latent sequence length exceeds maximum seq_len"
        tgt = z + self.pos_emb[:, :T, :]
        mask = self._generate_causal_mask(T, z.device)
        dec = self.decoder(tgt, memory=z, tgt_mask=mask)
        return self.out_proj(dec)

# VAE with GT Latent Processor and Symmetric Transformer Decoder
class LatentGTVAE(nn.Module):
    """
    VAE with GT Latent Processor and Symmetric Transformer Decoder.
    """
    def __init__(self, input_dim: int, seq_len: int, latent_dim: int,
                 enc_layers: int = 2, enc_heads: int = 4, enc_ff: int = 256,
                 enc_dropout: float = 0.1,
                 gt_layers: int = 2, gt_heads: int = 8, gt_k: int = 8,
                 gt_dropout: float = 0.1,
                 dec_layers: int = 2, dec_heads: int = 4, dec_ff: int = 256, dec_dropout: float = 0.1):
        super().__init__()
        self.encoder = TransformerEncoder(
            input_dim, seq_len, latent_dim,
            num_layers=enc_layers, num_heads=enc_heads, ff_dim=enc_ff, dropout=enc_dropout
        )
        self.gt_layers = nn.ModuleList([
            SparseGraphTransformerLayer(
                latent_dim,
                num_heads=gt_heads,
                k=gt_k,
                dropout=gt_dropout,
                max_seq_len=seq_len
            )
            for _ in range(gt_layers)
        ])
        self.mu_head = nn.Linear(latent_dim, latent_dim)
        self.logvar_head = nn.Linear(latent_dim, latent_dim)
        self.decoder = TransformerDecoder(
            latent_dim, seq_len,
            num_layers=dec_layers, num_heads=dec_heads, ff_dim=dec_ff,
            dropout=dec_dropout, input_dim=input_dim
        )
        self._init_parameters()

    def _init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        logvar = logvar.clamp(min=-10.0, max=10.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def process_latent(self, z: Tensor) -> Tensor:
        for layer in self.gt_layers:
            z = z + layer(z)
        return z

    def forward(self, x: Tensor):
        z = self.encoder(x)
        z = self.process_latent(z)
        mu = self.mu_head(z)
        logvar = self.logvar_head(z)
        z_sample = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z_sample)
        return x_hat, mu, logvar
