import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

# -------------------------------
# Laplace basis (normalized time)
# -------------------------------

class LearnableLaplacianBasis(nn.Module):
    """
    x:[B, T, D] -> Laplace features:[B, T, 2k] using learnable complex poles.
    Normalized time t in [0,1], with learnable global timescale τ.
    """
    def __init__(self, k: int, feat_dim: int, alpha_min: float = 1e-6):
        super().__init__()
        self.k = k
        self.alpha_min = alpha_min

        self.s_real = nn.Parameter(torch.empty(k))
        self.s_imag = nn.Parameter(torch.empty(k))
        self.reset_parameters()

        self.proj = spectral_norm(nn.Linear(feat_dim, k, bias=True), n_power_iterations=1, eps=1e-7)
        self._tau = nn.Parameter(torch.tensor(0.0))  # softplus -> positive scale

    def reset_parameters(self):
        nn.init.uniform_(self.s_real, 0.01, 0.2)  # α > 0
        nn.init.uniform_(self.s_imag, -math.pi, math.pi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        device = x.device
        t_idx = torch.linspace(0, 1, T, device=device, dtype=torch.float32).unsqueeze(1)  # [T,1]
        tau = F.softplus(self._tau) + 1e-3
        s = (-self.s_real.clamp_min(self.alpha_min) + 1j * self.s_imag) * tau  # [k]

        expo = torch.exp(t_idx * s.unsqueeze(0))  # [T,k] complex
        re_basis, im_basis = expo.real, expo.imag

        proj_feats = self.proj(x)                                  # [B,T,k]
        real_proj = proj_feats * re_basis.unsqueeze(0)             # [B,T,k]
        imag_proj = proj_feats * im_basis.unsqueeze(0)             # [B,T,k]
        return torch.cat([real_proj, imag_proj], dim=2)            # [B,T,2k]


class LearnableInverseLaplacianBasis(nn.Module):
    """ Maps Laplace features [B, T, 2k] back to feature space [B, T, D]. """
    def __init__(self, laplace_basis: LearnableLaplacianBasis):
        super().__init__()
        feat_dim = laplace_basis.proj.in_features                  # D
        self.inv_proj = spectral_norm(nn.Linear(2 * laplace_basis.k, feat_dim, bias=True),
                                      n_power_iterations=1, eps=1e-7)

    def forward(self, lap_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lap_feats: [B, T, 2k]
        Returns:
            x_hat: [B, T, D]
        """
        return self.inv_proj(lap_feats)