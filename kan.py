"""
B-spline KAN (Kolmogorov–Arnold Network) layer.

Based on the KAN paper: each edge carries a learnable univariate function
represented as a B-spline on a fixed grid, plus a residual SiLU branch.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class KANLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear that uses learnable B-spline
    activation on each (input, output) edge.

    out_j = sum_i [ w_base_{ij} * silu(x_i) + spline_{ij}(x_i) ]
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        grid_range: tuple[float, float] = (-1.0, 1.0),
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Grid: extended with `spline_order` points on each side
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.arange(-spline_order, grid_size + spline_order + 1, dtype=torch.float32) * h + grid_range[0]
        # shape: (in_features, grid_size + 2*spline_order + 1)
        grid = grid.unsqueeze(0).expand(in_features, -1)
        self.register_buffer("grid", grid)

        # Learnable spline coefficients
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order)
        )
        # Base linear weight (residual branch)
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))

        # Scaling factors
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.scale_noise = scale_noise

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        with torch.no_grad():
            noise = (torch.rand_like(self.spline_weight) - 0.5) * self.scale_noise
            self.spline_weight.copy_(self._curve2coeff(
                torch.linspace(-1, 1, self.grid_size + self.spline_order, device=self.spline_weight.device)
                .unsqueeze(0).expand(self.in_features, -1),
            ) + noise)

    def _curve2coeff(self, x: torch.Tensor) -> torch.Tensor:
        """Initialize spline coefficients via identity-ish mapping."""
        # x: (in_features, n_coeffs)
        # Returns: (1, in_features, n_coeffs) — broadcast-ready
        return x.unsqueeze(0).expand(self.out_features, -1, -1) * 0.1

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute B-spline basis values.
        x: (..., in_features)
        Returns: (..., in_features, grid_size + spline_order)
        """
        x = x.unsqueeze(-1)  # (..., in_features, 1)
        grid = self.grid  # (in_features, G) where G = grid_size + 2*spline_order + 1
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).float()

        for k in range(1, self.spline_order + 1):
            left_num = x - grid[:, :-(k + 1)]
            left_den = grid[:, k:-1] - grid[:, :-(k + 1)]
            right_num = grid[:, k + 1:] - x
            right_den = grid[:, k + 1:] - grid[:, 1:(-k) if (-k) != 0 else None]

            left = left_num / left_den.clamp(min=1e-7)
            right = right_num / right_den.clamp(min=1e-7)

            bases = left * bases[:, :, :-1] + right * bases[:, :, 1:]

        return bases  # (..., in_features, grid_size + spline_order)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., in_features)
        Returns: (..., out_features)
        """
        original_shape = x.shape
        # Flatten leading dims
        x_flat = x.reshape(-1, self.in_features)  # (B, in_features)

        # Base branch: silu activation + linear
        base_out = F.linear(F.silu(x_flat), self.base_weight) * self.scale_base

        # Spline branch
        splines = self.b_splines(x_flat)  # (B, in_features, n_coeffs)
        # spline_weight: (out_features, in_features, n_coeffs)
        # einsum: batch, in_features, coeffs × out_features, in_features, coeffs → batch, out_features
        spline_out = torch.einsum("bic,oic->bo", splines, self.spline_weight) * self.scale_spline

        out = base_out + spline_out
        # Restore shape
        out = out.reshape(*original_shape[:-1], self.out_features)
        return out
