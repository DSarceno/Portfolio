"""Phase-1 dynamics: graph diffusion + decay.

Congestion relaxes toward free flow (decay) and propagates to topological neighbours
(diffusion via the graph Laplacian). The transition is linear, which makes it compatible
with the exact Kalman update and easy to validate:

    x_{t+1} = F x_t (+ u_t),    F = I - decay * I - diffusion * L

This is the smallest dynamics capturing the two non-negotiable behaviours — dissipation
and network propagation. A Neural ODE replaces ``F`` in Phase 3 behind the same interface.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GraphDiffusionDynamics:
    """Linear graph-diffusion transition model.

    Parameters are per-step rates. For stability the spectral radius of ``F`` must be
    below 1; :meth:`is_stable` checks this. ``dt_minutes`` is metadata used to map forecast
    horizons (30/60/120 min) to a number of steps.
    """

    laplacian: np.ndarray
    decay: float = 0.05
    diffusion: float = 0.02
    dt_minutes: float = 5.0

    def __post_init__(self) -> None:
        if self.decay < 0 or self.diffusion < 0:
            raise ValueError("decay and diffusion must be non-negative")
        if self.dt_minutes <= 0:
            raise ValueError("dt_minutes must be positive")

    @property
    def n_edges(self) -> int:
        return self.laplacian.shape[0]

    def transition_matrix(self) -> np.ndarray:
        n = self.n_edges
        return np.eye(n) - self.decay * np.eye(n) - self.diffusion * self.laplacian

    def spectral_radius(self) -> float:
        eigenvalues = np.linalg.eigvals(self.transition_matrix())
        return float(np.max(np.abs(eigenvalues)))

    def is_stable(self) -> bool:
        return self.spectral_radius() < 1.0

    def step(self, x: np.ndarray, u: np.ndarray | None = None) -> np.ndarray:
        """Advance the state one step (no noise). ``u`` is an optional exogenous input."""
        nxt = self.transition_matrix() @ x
        if u is not None:
            nxt = nxt + u
        return nxt

    def predict(
        self, x: np.ndarray, cov: np.ndarray, process_cov: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Kalman predict step: propagate mean and covariance one step forward."""
        f = self.transition_matrix()
        return f @ x, f @ cov @ f.T + process_cov

    def steps_for_minutes(self, minutes: float) -> int:
        return int(round(minutes / self.dt_minutes))

    def forecast(
        self,
        x: np.ndarray,
        cov: np.ndarray,
        process_cov: np.ndarray,
        n_steps: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Roll the state forward ``n_steps`` with no observations.

        Returns ``(means, covs)`` of shape ``(n_steps, N)`` and ``(n_steps, N, N)``.
        """
        means = np.empty((n_steps, x.shape[0]))
        covs = np.empty((n_steps, x.shape[0], x.shape[0]))
        for i in range(n_steps):
            x, cov = self.predict(x, cov, process_cov)
            means[i] = x
            covs[i] = cov
        return means, covs
