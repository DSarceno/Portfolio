"""Common dynamics interface.

Any state-transition model the estimator and simulator use must implement this protocol.
The linear :class:`GraphDiffusionDynamics` and the :class:`NeuralODEDynamics` both conform,
which is what lets the Kalman Filter act as a plain KF for linear dynamics and as an
Extended Kalman Filter (Jacobian-linearized) for nonlinear dynamics — without changing the
estimator or simulator.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class DynamicsModel(Protocol):
    """State-transition model over per-edge congestion."""

    @property
    def n_edges(self) -> int: ...

    def step(self, x: np.ndarray, u: np.ndarray | None = None) -> np.ndarray:
        """Advance the mean state one step (deterministic)."""
        ...

    def predict(
        self, x: np.ndarray, cov: np.ndarray, process_cov: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Propagate mean and covariance one step (linear or Jacobian-linearized)."""
        ...

    def steps_for_minutes(self, minutes: float) -> int:
        """Number of steps spanning ``minutes`` of wall-clock time."""
        ...

    def forecast(
        self, x: np.ndarray, cov: np.ndarray, process_cov: np.ndarray, n_steps: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Roll the state forward ``n_steps`` with no observations."""
        ...
