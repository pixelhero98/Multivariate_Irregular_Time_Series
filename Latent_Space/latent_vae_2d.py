import torch, math
import torch.nn as nn
import torch.nn.functional as F

def sinusoidal_position(d_model, length, device):
    pe = torch.zeros(length, d_model, device=device)
    position = torch.arange(0, length, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

class Patch2D(nn.Module):
    def __init__(self, p_n: int = 1, p_h: int = 4):
        super().__init__()
        self.p_n = int(p_n)
        self.p_h = int(p_h)
        assert self.p_n >= 1 and self.p_h >= 1

    def forward(self, x):
        B, N, H, D = x.shape
        assert D == 1, "Patch2D expects last dim=1"
        pn, ph = self.p_n, self.p_h
        Np = (N + pn - 1) // pn * pn
        Hp = (H + ph - 1) // ph * ph
        if Np != N or Hp != H:
            pad_N = Np - N
            pad_H = Hp - H
            x = F.pad(x, (0,0, 0,pad_H, 0,pad_N))
        x = x.view(B, Np // pn, pn, Hp // ph, ph, 1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        tokens = x.view(B, (Np // pn) * (Hp // ph), pn * ph)
        grid = (Np // pn, Hp // ph)
        orig = (N, H)
        return tokens, grid, orig

class Unpatch2D(nn.Module):
    def __init__(self, p_n: int = 1, p_h: int = 4):
        super().__init__()
        self.p_n = int(p_n); self.p_h = int(p_h)

    def forward(self, tokens, grid, orig):
        B, T, P = tokens.shape
        Np_, Hp_ = grid
        assert T == Np_ * Hp_
        pn, ph = self.p_n, self.p_h
        x = tokens.view(B, Np_, Hp_, pn, ph, 1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Np_ * pn, Hp_ * ph, 1)
        N, H = orig
        x = x[:, :N, :H, :]
        return x

class Pos2D(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, grid, device):
        Np, Hp = grid
        pn = sinusoidal_position(self.d_model, Np, device)
        ph = sinusoidal_position(self.d_model, Hp, device)
        pos = pn[:, None, :] + ph[None, :, :]
        return pos.view(Np*Hp, self.d_model)

class TransformerEncoderTokens(nn.Module):
    def __init__(self, d_in, d_model, n_heads=4, ff=256, n_layers=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(d_in, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, ff, dropout, batch_first=True, norm_first=True)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, key_padding_mask=None, pos=None):
        h = self.input_proj(x)
        if pos is not None:
            # pos: [T, d_model] -> broadcast over batch
            h = h + pos.unsqueeze(0)
        for layer in self.layers:
            h = layer(h, src_key_padding_mask=key_padding_mask)
        return self.norm(h)

class TransformerDecoderTokens(nn.Module):
    def __init__(self, d_model, n_heads=4, ff=256, n_layers=4, dropout=0.1, use_cross=True):
        super().__init__()
        self.use_cross = use_cross
        self.layers = nn.ModuleList([
            nn.ModuleDict(dict(
                self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True),
                cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True),
                mlp = nn.Sequential(nn.Linear(d_model, ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff, d_model)),
                n1 = nn.LayerNorm(d_model), n2 = nn.LayerNorm(d_model), n3 = nn.LayerNorm(d_model),
                drop = nn.Dropout(dropout),
            )) for _ in range(n_layers)
        ])
        self.skip_gate = nn.Parameter(torch.tensor(-4.0))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, memory=None, self_pad=None, mem_pad=None):
        h = x
        for blk in self.layers:
            z  = blk['self_attn'](blk['n1'](h), blk['n1'](h), blk['n1'](h), key_padding_mask=self_pad)[0]
            h  = h + blk['drop'](z)
            if self.use_cross and memory is not None:
                mem = memory.detach()
                gate = torch.sigmoid(self.skip_gate)
                zc = blk['cross_attn'](blk['n2'](h), mem, mem, key_padding_mask=mem_pad)[0]
                h  = h + blk['drop'](gate * zc)
            h = h + blk['drop'](blk['mlp'](blk['n3'](h)))
        return self.norm(h)

class LatentHead(nn.Module):
    def __init__(self, d_model, C):
        super().__init__()
        self.mu = nn.Linear(d_model, C)
        self.lv = nn.Linear(d_model, C)
    def forward(self, h):
        mu = self.mu(h)
        lv = self.lv(h)
        std = F.softplus(lv) + 1e-5
        eps = torch.randn_like(std)
        z = mu + eps * std
        logvar = (std**2 + 1e-12).log()
        return z, mu, logvar

class VAE2D(nn.Module):
    def __init__(self, d_model=128, C=8, p_n=1, p_h=4, n_layers=4, n_heads=4, ff=256, dropout=0.1, use_cross=True):
        super().__init__()
        self.pn, self.ph = int(p_n), int(p_h)
        self.patch   = Patch2D(self.pn, self.ph)
        self.pos2d   = Pos2D(d_model)
        self.encoder = TransformerEncoderTokens(d_in=self.pn*self.ph, d_model=d_model, n_heads=n_heads, ff=ff, n_layers=n_layers, dropout=dropout)
        self.latent  = LatentHead(d_model, C)
        self.to_dec  = nn.Linear(C, d_model)
        self.decoder = TransformerDecoderTokens(d_model, n_heads, ff, n_layers, dropout, use_cross=use_cross)
        self.out_proj = nn.Linear(d_model, self.pn*self.ph)
        self.unpatch = Unpatch2D(self.pn, self.ph)

    @staticmethod
    def kl_gaussian(mu, logvar, reduce='mean'):
        kl = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1.0)
        return kl.mean() if reduce=='mean' else kl.sum() if reduce=='sum' else kl

    def _coverage_from_mask(self, mask, grid):
        if mask is None:
            return None, None
        B, N, H = mask.shape
        pn, ph = self.pn, self.ph
        Np = (N + pn - 1) // pn * pn
        Hp = (H + ph - 1) // ph * ph
        m = mask
        if Np != N or Hp != H:
            m = F.pad(m, (0, Hp - H, 0, Np - N))
        m = m.view(B, Np//pn, pn, Hp//ph, ph).permute(0,1,3,2,4)  # [B,N',H',pn,ph]
        cov = m.float().mean(dim=(3,4), keepdim=False)            # [B,N',H']
        key_pad = (cov <= 0.0).view(B, -1)                        # [B,T]
        return cov, key_pad

    def forward(self, x, mask=None):
        B, N, H, D = x.shape
        tokens, grid, orig = self.patch(x)                        # [B,T,P], (N',H')
        pos = self.pos2d(grid, tokens.device)                     # [T, d_model]
        # coverage/padding
        coverage, key_pad = self._coverage_from_mask(mask, grid) if mask is not None else (None, None)
        enc = self.encoder(tokens, key_padding_mask=key_pad, pos=pos)      # [B,T,d_model]
        z, mu, logvar = self.latent(enc)                          # [B,T,C]
        dec_in = self.to_dec(z)
        dec = self.decoder(dec_in, memory=enc, self_pad=key_pad, mem_pad=key_pad)  # [B,T,d_model]
        out_tokens = self.out_proj(dec)                           # [B,T,P]
        x_hat = self.unpatch(out_tokens, grid, orig)              # [B,N,H,1]
        Np, Hp = grid
        mu_grid     = mu.view(B, Np, Hp, -1)                      # [B,N',H',C]
        logvar_grid = logvar.view(B, Np, Hp, -1)                  # [B,N',H',C]
        z_grid = mu_grid.permute(0,3,1,2).contiguous()            # [B,C,N',H']
        cov_grid = coverage.unsqueeze(1) if coverage is not None else None  # [B,1,N',H'] or None
        return x_hat, mu_grid, logvar_grid, z_grid, cov_grid, grid, orig
