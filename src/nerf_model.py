"""
nerf_model.py
-------------
Tiny NeRF: a small MLP that maps (x, y, z, direction) → (RGB, density).
Uses positional encoding to let the network learn high-frequency detail.
CPU-friendly — no CUDA required.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """
    Maps each input value to [sin(2^0 * pi * x), cos(2^0 * pi * x), ...,
                               sin(2^(L-1) * pi * x), cos(2^(L-1) * pi * x)]
    This helps the MLP learn high-frequency functions (sharp edges, details).
    """
    def __init__(self, num_frequencies: int = 6):
        super().__init__()
        self.num_frequencies = num_frequencies
        # Precompute frequency bands: [1, 2, 4, 8, ...]
        freqs = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
        self.register_buffer("freqs", freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., D) raw input
        Returns:
            (..., D * 2 * num_frequencies) encoded output
        """
        # x[..., None] * freqs → (..., D, num_frequencies)
        x_freq = x.unsqueeze(-1) * self.freqs * torch.pi
        encoded = torch.cat([torch.sin(x_freq), torch.cos(x_freq)], dim=-1)
        # flatten last two dims
        return encoded.flatten(-2)


# ---------------------------------------------------------------------------
# NeRF MLP
# ---------------------------------------------------------------------------

class TinyNeRF(nn.Module):
    """
    A minimal NeRF network.

    Architecture:
        - Positional encoding on 3D position (x,y,z)
        - 4-layer MLP → density (σ) + feature vector
        - Feature + encoded view direction → RGB

    The key insight: density depends only on position (geometry),
    but colour also depends on view direction (appearance/reflections).
    """

    def __init__(
        self,
        pos_freq: int = 6,
        dir_freq: int = 4,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.pos_enc = PositionalEncoding(pos_freq)
        self.dir_enc = PositionalEncoding(dir_freq)

        pos_input_dim = 3 * 2 * pos_freq   # 3 coords × 2 (sin+cos) × freqs
        dir_input_dim = 3 * 2 * dir_freq

        # --- Geometry branch: position → density + feature ---
        self.geo_net = nn.Sequential(
            nn.Linear(pos_input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),    nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),    nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),    nn.ReLU(),
        )
        self.density_head = nn.Linear(hidden_dim, 1)   # σ (volume density)
        self.feature_head = nn.Linear(hidden_dim, hidden_dim)

        # --- Appearance branch: feature + direction → RGB ---
        self.color_net = nn.Sequential(
            nn.Linear(hidden_dim + dir_input_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),  # R, G, B
        )

    def forward(
        self,
        positions: torch.Tensor,   # (N, 3)
        directions: torch.Tensor,  # (N, 3) unit vectors
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            rgb:     (N, 3)  values in [0, 1]
            density: (N,)    non-negative volume density
        """
        pos_enc = self.pos_enc(positions)    # (N, pos_input_dim)
        dir_enc = self.dir_enc(directions)   # (N, dir_input_dim)

        geo_feat = self.geo_net(pos_enc)                         # (N, hidden)
        density  = F.softplus(self.density_head(geo_feat))       # (N, 1) ≥ 0
        feature  = self.feature_head(geo_feat)                   # (N, hidden)

        color_input = torch.cat([feature, dir_enc], dim=-1)
        rgb = torch.sigmoid(self.color_net(color_input))         # (N, 3) in [0,1]

        return rgb, density.squeeze(-1)
