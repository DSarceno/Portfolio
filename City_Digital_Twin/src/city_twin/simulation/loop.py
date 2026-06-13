"""Simulation loop.

Ties the layers together for a Phase-1 run: the synthetic generator produces the (hidden)
ground-truth world and noisy partial observations; the Kalman Filter estimates the hidden
state over time. The loop records both so estimation quality can be measured.

This module runs the estimation<->dynamics loop only. Scenario/intervention policy lives in
the ``scenarios`` layer and enters through the ground-truth generator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from city_twin.estimation.kalman import GaussianState, KalmanFilter
from city_twin.observation.schema import Observation
from city_twin.observation.synthetic import SyntheticObservations
from city_twin.scenarios.interventions import Intervention


@dataclass
class SimulationResult:
    """Outputs of a simulation run, aligned to steps ``0 .. n_steps``."""

    true_congestion: np.ndarray  # (T+1, N) hidden ground truth
    est_congestion: np.ndarray  # (T+1, N) filtered mean
    est_cov_trace: np.ndarray  # (T+1,) trace of covariance per step
    observations: list[Observation]

    def rmse(self) -> float:
        """Root-mean-square estimation error over all edges and steps."""
        return float(np.sqrt(np.mean((self.est_congestion - self.true_congestion) ** 2)))

    def persistence_rmse(self) -> float:
        """Baseline RMSE of a naive last-observed-value-carried-forward estimator.

        Edges never observed up to a step fall back to the prior mean (0.1). This is the
        bar the Kalman Filter must beat.
        """
        t_steps, n = self.true_congestion.shape
        last = np.full(n, 0.1)
        preds = np.empty_like(self.true_congestion)
        preds[0] = last
        for k, obs in enumerate(self.observations, start=1):
            idx = np.flatnonzero(obs.mask)
            last = last.copy()
            last[idx] = obs.congestion[idx]
            preds[k] = last
        return float(np.sqrt(np.mean((preds - self.true_congestion) ** 2)))


def run_simulation(
    synth: SyntheticObservations,
    kalman: KalmanFilter,
    n_steps: int,
    *,
    x0: np.ndarray | None = None,
    intervention: Intervention | None = None,
    initial_uncertainty: float = 1.0,
    initial_mean: float | None = None,
) -> SimulationResult:
    """Run a ground-truth + estimation simulation for ``n_steps`` steps."""
    network = synth.network
    n = network.n_edges

    ground_truth = synth.generate_ground_truth(n_steps, x0=x0, intervention=intervention)
    observations = synth.observe(ground_truth)

    mean0 = np.full(n, 0.1 if initial_mean is None else initial_mean)
    initial = GaussianState(mean=mean0, cov=np.eye(n) * initial_uncertainty)
    beliefs = kalman.filter(initial, observations)

    est = np.array([b.mean for b in beliefs])
    cov_trace = np.array([float(np.trace(b.cov)) for b in beliefs])

    return SimulationResult(
        true_congestion=ground_truth,
        est_congestion=est,
        est_cov_trace=cov_trace,
        observations=observations,
    )
