import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

__all__ = ["LaplaceTransformEncoder", "LaplacePseudoInverse"]


class LaplaceTransformEncoder(nn.Module):
    """
    Project a time-domain sequence onto a stable damped-sinusoid basis.
    
    This implementation solely uses the 'effective' modal parameterization,
    learning stable poles (rho, omega) that are conditionally perturbed by 
    a summary embedding.

    Key Features:
    1. Effective Poles: Base poles + conditional perturbations.
    2. Adaptive Ridge: Regularization lambda scales with conditioning (noise level).
    3. Log-Frequency Init: Frequencies initialized on a log scale.
    """

    def __init__(
        self,
        k: int,
        feat_dim: int,
        alpha_min: float = 1e-6,
        omega_max: float = math.pi,
        cond_dim: Optional[int] = None,
        rho_perturb_scale: float = 0.5,
        omega_perturb_scale: float = 0.5,
        ridge_lambda: float = 1e-3,
        adaptive_ridge: bool = True,
    ) -> None:
        super().__init__()
        self.k = int(k)
        self.feat_dim = int(feat_dim)
        self.alpha_min = float(alpha_min)
        self.omega_max = float(omega_max)
        self.rho_perturb_scale = float(rho_perturb_scale)
        self.omega_perturb_scale = float(omega_perturb_scale)
        self.base_ridge_lambda = float(ridge_lambda)
        self.adaptive_ridge = adaptive_ridge and (cond_dim is not None)

        # Base pole parameters
        # rho > 0 via softplus + alpha_min
        # omega in (0, omega_max) via sigmoid
        self._rho_raw = nn.Parameter(torch.empty(self.k))
        self._omega_raw = nn.Parameter(torch.empty(self.k))

        self.cond_dim = cond_dim
        if cond_dim is not None:
            # Head for pole perturbation
            self.to_poles = nn.Sequential(
                nn.SiLU(),
                nn.Linear(cond_dim, 2 * self.k),
            )
            nn.init.zeros_(self.to_poles[-1].weight)
            nn.init.zeros_(self.to_poles[-1].bias)

            # Head for adaptive ridge regularization
            if self.adaptive_ridge:
                self.to_lambda = nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(cond_dim, 1)
                )
                nn.init.zeros_(self.to_lambda[-1].weight)
                nn.init.zeros_(self.to_lambda[-1].bias)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            # Initialize rho in a stable range [0.01, 0.2]
            target_rho = torch.empty_like(self._rho_raw).uniform_(0.01, 0.2)
            y = (target_rho - self.alpha_min).clamp_min(1e-8)
            self._rho_raw.copy_(torch.log(torch.expm1(y)))  # inverse softplus

            # Initialize omega logarithmically to cover (0, omega_max)
            # This avoids clustering at high frequencies and mimics Mel-scale coverage.
            low_log = math.log(0.01 * self.omega_max)
            high_log = math.log(0.95 * self.omega_max)
            target_omega = torch.exp(torch.empty_like(self._omega_raw).uniform_(low_log, high_log))
            
            # Inverse sigmoid for raw parameter
            p = (target_omega / self.omega_max).clamp(1e-4, 1 - 1e-4)
            self._omega_raw.copy_(torch.log(p) - torch.log1p(-p))

    def _get_lambda(self, cond: Optional[torch.Tensor]) -> Union[float, torch.Tensor]:
        """Compute ridge lambda, optionally conditioned on state (e.g. SNR)."""
        lam = self.base_ridge_lambda
        if self.adaptive_ridge and cond is not None:
            # Predict a multiplicative factor.
            # sigmoid output (0, 1) * 5 + 0.5 => scales lambda from 0.5x to 5.5x
            factor = 5.0 * torch.sigmoid(self.to_lambda(cond)) + 0.5
            return lam * factor
        return lam

    @staticmethod
    def _prepare_time_like(
        v: torch.Tensor,
        batch_size: int,
        seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Normalize time input to [B, T, 1]."""
        if v.dim() == 1:
            if v.numel() != seq_len:
                raise ValueError(f"Time input length {v.numel()} != sequence length {seq_len}.")
            v = v.view(1, seq_len).expand(batch_size, seq_len)
        elif v.dim() == 2:
            if v.size(1) != seq_len:
                raise ValueError(f"Time input T={v.size(1)} != sequence length {seq_len}.")
            if v.size(0) == 1 and batch_size > 1:
                v = v.expand(batch_size, seq_len)
            elif v.size(0) != batch_size:
                raise ValueError(f"Time input B={v.size(0)} != batch size {batch_size}.")
        elif v.dim() == 3:
            v = v.squeeze(-1)
            if v.size(1) != seq_len:
                raise ValueError(f"Time input T={v.size(1)} != sequence length {seq_len}.")
            if v.size(0) == 1 and batch_size > 1:
                v = v.expand(batch_size, seq_len)
            elif v.size(0) != batch_size:
                raise ValueError(f"Time input B={v.size(0)} != batch size {batch_size}.")
        else:
            raise ValueError("Time input must be [B,T], [T], or [B,T,1].")
        return v.to(device=device, dtype=dtype).unsqueeze(-1)

    def _relative_time(
        self,
        B: int,
        T: int,
        dtype: torch.dtype,
        device: torch.device,
        dt: Optional[torch.Tensor],
        t: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Return relative times t_rel in [B,T,1], shifted to start at 0."""
        if t is not None:
            t_rel = self._prepare_time_like(t, B, T, dtype, device)
            t_rel = t_rel - t_rel[:, :1]
            return t_rel
        if dt is not None:
            dt_ = self._prepare_time_like(dt, B, T, dtype, device)
            t_rel = torch.cumsum(dt_, dim=1)
            t_rel = t_rel - t_rel[:, :1]
            return t_rel
        
        # Default regular grid
        return torch.arange(T, device=device, dtype=dtype).view(1, T, 1).expand(B, T, 1)

    def _base_poles(self, dtype: torch.dtype, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        rho = F.softplus(self._rho_raw.to(device=device, dtype=dtype)) + self.alpha_min
        omega = self.omega_max * torch.sigmoid(self._omega_raw.to(device=device, dtype=dtype))
        return rho, omega

    def effective_poles(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
        cond: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (rho, omega) in shapes [B,K]."""
        rho0, omega0 = self._base_poles(dtype, device)
        rho = rho0.unsqueeze(0).expand(batch_size, self.k)
        omega = omega0.unsqueeze(0).expand(batch_size, self.k)

        if self.cond_dim is None or cond is None:
            return rho, omega
        
        # Apply conditional perturbations
        delta = self.to_poles(cond).view(batch_size, 2, self.k)
        d_rho = self.rho_perturb_scale * torch.tanh(delta[:, 0])
        d_omega = self.omega_perturb_scale * torch.tanh(delta[:, 1])

        rho = F.softplus((rho0.unsqueeze(0) + d_rho).to(dtype=dtype)) + self.alpha_min
        
        # Logit math for bounded omega perturbation
        p0 = (omega0 / self.omega_max).clamp(1e-4, 1 - 1e-4)
        logit0 = torch.log(p0) - torch.log1p(-p0)
        omega = self.omega_max * torch.sigmoid(logit0.unsqueeze(0) + d_omega)
        return rho, omega

    def basis_matrix(
        self,
        t_rel: torch.Tensor,
        rho: torch.Tensor,
        omega: torch.Tensor,
    ) -> torch.Tensor:
        """Return B(t) in [B,T,2K]."""
        rho_ = rho.unsqueeze(1)    # [B,1,K]
        omega_ = omega.unsqueeze(1) # [B,1,K]
        
        decay = torch.exp(-t_rel * rho_)
        angle = t_rel * omega_
        
        cos_basis = decay * torch.cos(angle)
        sin_basis = decay * torch.sin(angle)
        
        return torch.cat([cos_basis, sin_basis], dim=-1).contiguous()

    def forward(
        self,
        x: torch.Tensor,
        dt: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
        return_basis: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        
        if x.dim() != 3 or x.size(-1) != self.feat_dim:
            raise ValueError(f"Input x must be [B, T, {self.feat_dim}]")
        B, T, D = x.shape
        
        t_rel = self._relative_time(B, T, x.dtype, x.device, dt, t)
        rho, omega = self.effective_poles(B, x.dtype, x.device, cond)
        Bmat = self.basis_matrix(t_rel, rho, omega)  # [B,T,2K]

        # Ridge projection: Theta = (B^T B + lam * I)^(-1) B^T x
        Bt = Bmat.transpose(1, 2) # [B, 2K, T]
        BtB = torch.bmm(Bt, Bmat) # [B, 2K, 2K]
        BtX = torch.bmm(Bt, x)    # [B, 2K, D]

        lam_val = self._get_lambda(cond) # Float or [B,1]

        # Regularization matrix
        eye = torch.eye(2 * self.k, device=x.device, dtype=x.dtype).unsqueeze(0).expand(B, -1, -1)
        if isinstance(lam_val, torch.Tensor):
            lam_mat = lam_val.view(B, 1, 1) * eye
        else:
            lam_mat = lam_val * eye
            
        G = BtB + lam_mat
        
        # Cholesky solve for stability
        try:
            L = torch.linalg.cholesky(G)
            Theta = torch.cholesky_solve(BtX, L)
        except RuntimeError:
            # Fallback for ill-conditioned matrices (rare with sufficient ridge)
            # Add extra jitter if Cholesky fails
            G = G + 1e-4 * eye
            L = torch.linalg.cholesky(G)
            Theta = torch.cholesky_solve(BtX, L)

        if return_basis:
            return Theta.contiguous(), Bmat, rho, omega
        return Theta.contiguous()


class LaplacePseudoInverse(nn.Module):
    """
    Synthesize a time-domain sequence from modal coefficients.
    Reconstructs y = B * Theta + Residual(y).
    """

    def __init__(
        self,
        encoder: LaplaceTransformEncoder,
        hidden_dim: Optional[int] = None,
        use_mlp_residual: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.use_mlp_residual = bool(use_mlp_residual)

        D = encoder.feat_dim
        # Default hidden dim for residual MLP
        H = int(hidden_dim if hidden_dim is not None else D * 2)

        if self.use_mlp_residual:
            self.norm = nn.LayerNorm(D)
            # Gated Residual MLP (GLU-based) for better signal propagation
            self.mlp_in = spectral_norm(nn.Linear(D, H * 2)) 
            self.mlp_out = spectral_norm(nn.Linear(H, D))
            nn.init.zeros_(self.mlp_out.weight)
            nn.init.zeros_(self.mlp_out.bias)

    def forward(self, theta: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
        """Synthesize y from coefficients and basis."""
        if basis.size(0) != theta.size(0):
            raise ValueError("Basis and Theta must have the same batch size")
        if theta.shape[1] != basis.shape[2]:
             raise ValueError(f"Dimension mismatch: Basis has {basis.shape[2]} modes, Theta has {theta.shape[1]}")

        # Linear synthesis (Exact for the basis)
        y = torch.bmm(basis, theta)  # [B,T,D]

        if not self.use_mlp_residual:
            return y
            
        # Gated residual refinement
        # ResNet-style add: y = y + MLP(Norm(y))
        res = self.norm(y)
        res_gate, res_val = self.mlp_in(res).chunk(2, dim=-1)
        res = res_val * F.gelu(res_gate) # GLU
        y = y + self.mlp_out(res)
        
        return y
