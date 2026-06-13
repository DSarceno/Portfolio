"""Phase-3 dynamics: a hybrid physics-informed Neural ODE.

The continuous-time state derivative is

    dx/dt = -decay * x - diffusion * L x + residual_net([x, A_norm x])

i.e. the same interpretable physics as the Phase-1 linear model (decay + graph diffusion)
plus a *learned, network-aware residual*. ``decay`` and ``diffusion`` remain trainable but
constrained positive, so the model degrades gracefully toward the validated linear baseline
when the residual is small.

The wrapper is numpy-facing and exposes the same interface as
:class:`~city_twin.dynamics.graph_diffusion.GraphDiffusionDynamics` (see
:class:`~city_twin.dynamics.base.DynamicsModel`). Covariance is propagated by linearizing
the one-step map with autograd — so the existing Kalman Filter becomes an Extended Kalman
Filter automatically, with no change to the estimator or simulator.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import torch
from torch import nn
from torchdiffeq import odeint


class _ODEFunc(nn.Module):
    """Right-hand side of the hybrid ODE. Operates on the last axis (works batched)."""

    def __init__(self, laplacian: np.ndarray, adjacency: np.ndarray, hidden: int = 16) -> None:
        super().__init__()
        self.register_buffer("lap", torch.as_tensor(laplacian, dtype=torch.float32))
        adj = torch.as_tensor(adjacency, dtype=torch.float32)
        degree = adj.sum(dim=1, keepdim=True).clamp(min=1.0)
        self.register_buffer("adj_norm", adj / degree)  # row-normalized neighbour averaging
        # Constrained-positive physics coefficients (softplus of raw params).
        self.raw_decay = nn.Parameter(torch.tensor(-2.0))
        self.raw_diffusion = nn.Parameter(torch.tensor(-3.0))
        # Per-edge residual MLP on [x_e, (A_norm x)_e]; shared weights across edges.
        self.net = nn.Sequential(nn.Linear(2, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        decay = nn.functional.softplus(self.raw_decay)
        diffusion = nn.functional.softplus(self.raw_diffusion)
        lap = cast(torch.Tensor, self.lap)
        adj_norm = cast(torch.Tensor, self.adj_norm)
        lap_x = x @ lap.T
        neigh = x @ adj_norm.T
        physics = -decay * x - diffusion * lap_x
        feats = torch.stack([x, neigh], dim=-1)  # (..., N, 2)
        residual = self.net(feats).squeeze(-1)  # (..., N)
        return physics + residual


class NeuralODEDynamics:
    """Learnable continuous-time dynamics behind the standard dynamics interface."""

    def __init__(
        self,
        laplacian: np.ndarray,
        adjacency: np.ndarray,
        *,
        hidden: int = 16,
        step_size: float = 1.0,
        dt_minutes: float = 5.0,
        method: str = "rk4",
        seed: int = 0,
    ) -> None:
        torch.manual_seed(seed)
        self._n = laplacian.shape[0]
        self.func = _ODEFunc(laplacian, adjacency, hidden=hidden)
        self.step_size = step_size
        self.dt_minutes = dt_minutes
        self.method = method

    @property
    def n_edges(self) -> int:
        return self._n

    def _integrate(self, x: torch.Tensor) -> torch.Tensor:
        """Integrate one ``step_size`` interval. Differentiable; supports batching."""
        t = torch.tensor([0.0, self.step_size], dtype=torch.float32)
        return odeint(self.func, x, t, method=self.method)[-1]

    def step(self, x: np.ndarray, u: np.ndarray | None = None) -> np.ndarray:
        x_t = torch.as_tensor(np.asarray(x, dtype=np.float32))
        with torch.no_grad():
            nxt = self._integrate(x_t).numpy().astype(np.float64)
        if u is not None:
            nxt = nxt + np.asarray(u, dtype=np.float64)
        return nxt

    def _jacobian(self, x: np.ndarray) -> np.ndarray:
        x_t = torch.as_tensor(np.asarray(x, dtype=np.float32))
        jac = torch.autograd.functional.jacobian(self._integrate, x_t)
        return np.asarray(jac.detach().numpy(), dtype=np.float64)

    def predict(
        self, x: np.ndarray, cov: np.ndarray, process_cov: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        jac = self._jacobian(x)  # one-step Jacobian (EKF linearization)
        mean = self.step(x)
        new_cov = jac @ cov @ jac.T + process_cov
        return mean, 0.5 * (new_cov + new_cov.T)

    def steps_for_minutes(self, minutes: float) -> int:
        return int(round(minutes / self.dt_minutes))

    def forecast(
        self, x: np.ndarray, cov: np.ndarray, process_cov: np.ndarray, n_steps: int
    ) -> tuple[np.ndarray, np.ndarray]:
        means = np.empty((n_steps, self._n))
        covs = np.empty((n_steps, self._n, self._n))
        for i in range(n_steps):
            x, cov = self.predict(x, cov, process_cov)
            means[i] = x
            covs[i] = cov
        return means, covs


def train_neural_ode(
    model: NeuralODEDynamics,
    trajectories: np.ndarray,
    *,
    epochs: int = 400,
    lr: float = 0.02,
    seed: int = 0,
) -> float:
    """Fit one-step transitions to observed trajectories. Returns the final MSE loss.

    ``trajectories`` has shape ``(n_trajectories, T + 1, n_edges)``. Training matches the
    model's one-step integration to consecutive ``(x_t, x_{t+1})`` pairs.
    """
    torch.manual_seed(seed)
    data = np.asarray(trajectories, dtype=np.float32)
    n = model.n_edges
    x0 = torch.tensor(data[:, :-1, :].reshape(-1, n))
    x1 = torch.tensor(data[:, 1:, :].reshape(-1, n))

    optimizer = torch.optim.Adam(model.func.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    loss = torch.tensor(float("nan"))
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model._integrate(x0)
        loss = loss_fn(pred, x1)
        loss.backward()
        optimizer.step()
    return float(loss.item())
