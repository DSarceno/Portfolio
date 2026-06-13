"""Kalman Filter for hidden-congestion estimation.

The Phase-1 estimator is a linear Kalman Filter. Dynamics are linear (graph diffusion) and
the observation map is a pure selection matrix in congestion units, so the standard Kalman
equations are exact. The filter naturally handles:

* measurement noise (covariance ``R``),
* partial / time-varying observability (the observation mask selects observed edges),
* uncertainty on unobserved edges (carried in the covariance ``P``).

It uses the dynamics operator from the ``dynamics`` layer; it does not define dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from city_twin.dynamics.base import DynamicsModel
from city_twin.observation.schema import Observation


@dataclass
class GaussianState:
    """A Gaussian belief over the hidden state: mean and covariance."""

    mean: np.ndarray  # (N,)
    cov: np.ndarray  # (N, N)


class KalmanFilter:
    """Linear Kalman Filter over per-edge congestion.

    ``process_cov`` is the process-noise covariance ``Q``. ``measurement_var`` is the
    per-edge measurement variance (diagonal of ``R``), in congestion units.
    """

    def __init__(
        self,
        dynamics: DynamicsModel,
        process_cov: np.ndarray,
        measurement_var: np.ndarray,
    ) -> None:
        n = dynamics.n_edges
        if process_cov.shape != (n, n):
            raise ValueError(f"process_cov must have shape ({n}, {n})")
        if measurement_var.shape != (n,):
            raise ValueError(f"measurement_var must have shape ({n},)")
        self.dynamics = dynamics
        self.process_cov = process_cov
        self.measurement_var = measurement_var

    def predict(self, state: GaussianState) -> GaussianState:
        mean, cov = self.dynamics.predict(state.mean, state.cov, self.process_cov)
        return GaussianState(mean, cov)

    def update(self, state: GaussianState, observation: Observation) -> GaussianState:
        """Fuse a (partial) observation into the belief via the Kalman update."""
        idx = np.flatnonzero(observation.mask)
        if idx.size == 0:
            return GaussianState(state.mean.copy(), state.cov.copy())

        # H is the selection of observed edges; implemented by indexing rather than a matrix.
        z = observation.congestion[idx]
        innovation = z - state.mean[idx]
        cov_xz = state.cov[:, idx]  # P H^T  -> (N, m)
        innovation_cov = state.cov[np.ix_(idx, idx)] + np.diag(self.measurement_var[idx])
        gain = cov_xz @ np.linalg.inv(innovation_cov)  # (N, m)

        new_mean = state.mean + gain @ innovation
        new_cov = state.cov - gain @ cov_xz.T
        # keep symmetric
        new_cov = 0.5 * (new_cov + new_cov.T)
        return GaussianState(new_mean, new_cov)

    def filter(
        self, initial: GaussianState, observations: list[Observation]
    ) -> list[GaussianState]:
        """Run predict/update over a sequence of observations.

        ``observations[k]`` is the measurement at step ``k+1``. Returns the filtered
        beliefs aligned to steps ``0 .. len(observations)`` (index 0 is ``initial``).
        """
        beliefs = [initial]
        state = initial
        for obs in observations:
            state = self.predict(state)
            state = self.update(state, obs)
            beliefs.append(state)
        return beliefs
