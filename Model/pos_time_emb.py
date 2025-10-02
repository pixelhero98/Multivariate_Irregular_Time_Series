import math
from typing import Optional

import torch


# -------------------------------
# Positional / timestep embeddings
# -------------------------------

def get_sinusoidal_pos_emb(L: int, dim: int,
                           device: Optional[torch.device] = None
                           ) -> torch.Tensor:
    """
    Standard 1D sinusoidal positional embeddings (Transformer style).

    Args:
        L:     Sequence length
        dim:   Embedding dimension (must be even)
        device:Target device; defaults to CPU if None

    Returns:
        [1, L, dim]
    """
    if dim % 2 != 0:
        raise ValueError("pos_emb dim must be even")

    device = device or torch.device("cpu")
    pos = torch.arange(L, device=device).unsqueeze(1).float()  # [L, 1]
    i = torch.arange(dim // 2, device=device).float()  # [dim/2]
    # denom = 10000^(i/(dim/2)) == 10000^(2i/dim)
    denom = torch.exp((i / (dim // 2)) * math.log(10000.0))  # [dim/2]
    angles = pos / denom  # [L, dim/2]
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)  # [L, dim]
    return emb.unsqueeze(0)  # [1, L, dim]


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """
    ADM/DiT-style timestep embedding for (integer) timesteps.

    Args:
        t:          [B] integer timesteps (any dtype convertible to float)
        dim:        Embedding dimension (must be even)
        max_period: Controls the minimum frequency

    Returns:
        [B, dim]
    """
    if dim % 2 != 0:
        raise ValueError("timestep embedding dim must be even")
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, device=t.device).float() / half)  # [half]
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # [B, half]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)  # [B, dim]
    return emb
