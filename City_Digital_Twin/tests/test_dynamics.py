"""Tests for graph-diffusion dynamics and forecasting."""

from __future__ import annotations

import numpy as np

from city_twin.dynamics.graph_diffusion import GraphDiffusionDynamics
from city_twin.ingestion.network import RoadNetwork


def test_transition_is_stable(dynamics: GraphDiffusionDynamics) -> None:
    assert dynamics.is_stable()
    assert dynamics.spectral_radius() < 1.0


def test_decay_dissipates_congestion(line_network: RoadNetwork) -> None:
    # Pure decay, no diffusion: uniform congestion shrinks toward free flow.
    dyn = GraphDiffusionDynamics(
        laplacian=line_network.laplacian(), decay=0.1, diffusion=0.0
    )
    x = np.full(line_network.n_edges, 0.8)
    nxt = dyn.step(x)
    assert np.all(nxt < x)
    assert np.allclose(nxt, 0.8 * (1 - 0.1))


def test_diffusion_propagates_to_neighbours(line_network: RoadNetwork) -> None:
    # A single congested edge raises its topological neighbours after one step.
    dyn = GraphDiffusionDynamics(
        laplacian=line_network.laplacian(), decay=0.0, diffusion=0.05
    )
    x = np.zeros(line_network.n_edges)
    x[0] = 1.0  # edge (0,1) congested
    nxt = dyn.step(x)
    assert nxt[1] > 0.0  # neighbour edge (1,2) gains congestion
    assert nxt[3] == 0.0  # far edge unaffected after one step


def test_steps_for_minutes(dynamics: GraphDiffusionDynamics) -> None:
    assert dynamics.steps_for_minutes(30) == 6
    assert dynamics.steps_for_minutes(60) == 12
    assert dynamics.steps_for_minutes(120) == 24


def test_forecast_shapes_and_no_blowup(dynamics: GraphDiffusionDynamics) -> None:
    n = dynamics.n_edges
    x = np.full(n, 0.5)
    cov = np.eye(n) * 0.1
    q = np.eye(n) * 1e-3
    means, covs = dynamics.forecast(x, cov, q, n_steps=24)
    assert means.shape == (24, n)
    assert covs.shape == (24, n, n)
    assert np.all(np.isfinite(means))
    assert np.max(np.abs(means)) <= np.max(np.abs(x)) + 1e-9  # stable: no growth


def test_forecast_uncertainty_grows_over_horizon(dynamics: GraphDiffusionDynamics) -> None:
    n = dynamics.n_edges
    x = np.full(n, 0.5)
    cov = np.eye(n) * 1e-3
    q = np.eye(n) * 1e-3
    _means, covs = dynamics.forecast(x, cov, q, n_steps=24)
    trace_30 = np.trace(covs[dynamics.steps_for_minutes(30) - 1])
    trace_120 = np.trace(covs[dynamics.steps_for_minutes(120) - 1])
    assert trace_120 > trace_30  # uncertainty grows without observations
