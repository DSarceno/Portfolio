"""Tests for the end-to-end simulation loop and forecast horizons."""

from __future__ import annotations

import numpy as np

from city_twin.dynamics.graph_diffusion import GraphDiffusionDynamics
from city_twin.estimation.kalman import GaussianState, KalmanFilter
from city_twin.ingestion.network import RoadNetwork
from city_twin.observation.synthetic import SyntheticObservations
from city_twin.simulation.loop import run_simulation


def _build(network: RoadNetwork, seed: int = 0):
    dyn = GraphDiffusionDynamics(
        laplacian=network.laplacian(), decay=0.05, diffusion=0.02, dt_minutes=5.0
    )
    synth = SyntheticObservations(
        network=network,
        dynamics=dyn,
        noise_speed_kmh=4.0,
        observed_fraction=0.6,
        process_noise=0.01,
        seed=seed,
    )
    kf = KalmanFilter(
        dynamics=dyn,
        process_cov=np.eye(network.n_edges) * 1e-3,
        measurement_var=synth.measurement_var(),
    )
    return dyn, synth, kf


def test_simulation_estimates_hidden_state_better_than_persistence(
    grid_network: RoadNetwork,
) -> None:
    _dyn, synth, kf = _build(grid_network, seed=1)
    result = run_simulation(synth, kf, n_steps=60)
    assert result.true_congestion.shape == result.est_congestion.shape
    assert np.all(np.isfinite(result.est_congestion))
    # The Digital Twin's core claim: fuse noisy partial observations into a better estimate.
    assert result.rmse() < result.persistence_rmse()


def test_simulation_handles_fully_missing_observations(grid_network: RoadNetwork) -> None:
    _dyn, synth, kf = _build(grid_network, seed=2)
    synth.observed_fraction = 0.0  # nothing is ever observed
    # Start from a confident prior so the "missing data" effect is visible: with no
    # observations the belief can only lose certainty, growing toward the dynamics'
    # steady-state covariance.
    result = run_simulation(synth, kf, n_steps=20, initial_uncertainty=1e-4)
    assert np.all(np.isfinite(result.est_congestion))
    assert result.est_cov_trace[-1] > result.est_cov_trace[0]


def test_forecast_horizons_are_stable(grid_network: RoadNetwork) -> None:
    dyn, synth, kf = _build(grid_network, seed=3)
    result = run_simulation(synth, kf, n_steps=40)
    # Forecast from the final estimate over the three required horizons.
    final = GaussianState(
        mean=result.est_congestion[-1], cov=np.eye(grid_network.n_edges) * 1e-2
    )
    q = np.eye(grid_network.n_edges) * 1e-3
    rmses = {}
    for minutes in (30, 60, 120):
        steps = dyn.steps_for_minutes(minutes)
        means, _covs = dyn.forecast(final.mean, final.cov, q, steps)
        rmses[minutes] = float(np.sqrt(np.mean(means[-1] ** 2)))
        assert np.all(np.isfinite(means))
        assert np.max(np.abs(means)) <= 1.5  # bounded, no blow-up
    assert set(rmses) == {30, 60, 120}
