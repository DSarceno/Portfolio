"""Tests for the Kalman Filter state estimator."""

from __future__ import annotations

import numpy as np

from city_twin.dynamics.graph_diffusion import GraphDiffusionDynamics
from city_twin.estimation.kalman import GaussianState, KalmanFilter
from city_twin.ingestion.network import RoadNetwork
from city_twin.observation.schema import Observation


def _kf(dynamics: GraphDiffusionDynamics) -> KalmanFilter:
    n = dynamics.n_edges
    return KalmanFilter(
        dynamics=dynamics,
        process_cov=np.eye(n) * 1e-3,
        measurement_var=np.full(n, 1e-2),
    )


def test_update_reduces_uncertainty_on_observed_edge(
    dynamics: GraphDiffusionDynamics,
) -> None:
    kf = _kf(dynamics)
    n = dynamics.n_edges
    state = GaussianState(mean=np.full(n, 0.1), cov=np.eye(n))
    mask = np.zeros(n, dtype=bool)
    mask[0] = True
    obs = Observation(step=1, congestion=np.array([0.8, 0, 0, 0.0]), mask=mask)
    updated = kf.update(state, obs)
    assert updated.cov[0, 0] < state.cov[0, 0]  # observed edge less uncertain
    assert updated.mean[0] > state.mean[0]  # pulled toward the observation


def test_empty_observation_is_noop(dynamics: GraphDiffusionDynamics) -> None:
    kf = _kf(dynamics)
    n = dynamics.n_edges
    state = GaussianState(mean=np.full(n, 0.3), cov=np.eye(n) * 0.5)
    obs = Observation(step=1, congestion=np.zeros(n), mask=np.zeros(n, dtype=bool))
    updated = kf.update(state, obs)
    assert np.allclose(updated.mean, state.mean)
    assert np.allclose(updated.cov, state.cov)


def test_covariance_stays_symmetric(dynamics: GraphDiffusionDynamics) -> None:
    kf = _kf(dynamics)
    n = dynamics.n_edges
    state = GaussianState(mean=np.full(n, 0.1), cov=np.eye(n))
    mask = np.array([True, False, True, False])
    obs = Observation(step=1, congestion=np.array([0.7, 0, 0.6, 0]), mask=mask)
    updated = kf.predict(kf.update(state, obs))
    assert np.allclose(updated.cov, updated.cov.T)


def test_filter_recovers_constant_hidden_state(line_network: RoadNetwork) -> None:
    # With static ground truth and partial noisy observations, the filter should converge
    # close to the truth on every edge, including ones observed infrequently.
    dyn = GraphDiffusionDynamics(laplacian=line_network.laplacian(), decay=0.0, diffusion=0.0)
    kf = _kf(dyn)
    n = line_network.n_edges
    rng = np.random.default_rng(0)
    truth = np.array([0.2, 0.6, 0.4, 0.8])
    observations = []
    for step in range(1, 81):
        noisy = truth + rng.normal(0, 0.05, size=n)
        mask = rng.random(n) < 0.5
        observations.append(Observation(step=step, congestion=noisy, mask=mask))
    initial = GaussianState(mean=np.full(n, 0.1), cov=np.eye(n))
    beliefs = kf.filter(initial, observations)
    final = beliefs[-1].mean
    assert np.max(np.abs(final - truth)) < 0.1
