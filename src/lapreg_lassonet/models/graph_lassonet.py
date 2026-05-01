from __future__ import annotations

from typing import Iterable, Tuple
import torch
import torch.nn as nn


class GraphLassoNet(nn.Module):
    """
    y_hat = x @ theta + MLP(x)
    - theta: sparse via proximal L1
    - optional Laplacian penalty: gamma * theta^T L theta
    """
    def __init__(self, input_dim: int, hidden_dims: Tuple[int, ...] = (20,)):
        super().__init__()
        self.input_dim = int(input_dim)
        self.theta = nn.Parameter(torch.zeros(self.input_dim, dtype=torch.float32))

        layers = []
        prev = self.input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, int(h)))
            layers.append(nn.ReLU())
            prev = int(h)
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [n,p]
        linear_res = x @ self.theta  # [n]
        mlp_out = self.mlp(x).squeeze(-1)  # [n]
        return linear_res + mlp_out


def prox_l1_inplace(theta: torch.Tensor, lam: float) -> None:
    """
    Soft-thresholding proximal operator for L1.
    """
    if lam <= 0:
        return
    with torch.no_grad():
        t = theta.data
        sign = torch.sign(t)
        new = torch.clamp(torch.abs(t) - lam, min=0.0) * sign
        t.copy_(new)
