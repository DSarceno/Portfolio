"""Tests for scenario interventions and their physical sanity."""

from __future__ import annotations

import numpy as np

from city_twin.dynamics.graph_diffusion import GraphDiffusionDynamics
from city_twin.ingestion.network import RoadNetwork
from city_twin.observation.synthetic import SyntheticObservations
from city_twin.scenarios.compare import compare_intervention
from city_twin.scenarios.interventions import Accident, Rainfall, RoadClosure


def _synth(network: RoadNetwork, seed: int = 0) -> SyntheticObservations:
    dyn = GraphDiffusionDynamics(
        laplacian=network.laplacian(), decay=0.05, diffusion=0.05, dt_minutes=5.0
    )
    return SyntheticObservations(
        network=network, dynamics=dyn, process_noise=0.0, seed=seed
    )


def test_road_closure_raises_downstream_congestion(grid_network: RoadNetwork) -> None:
    # Close edge 0 = (0,1); its neighbours should end up more congested than baseline.
    synth = _synth(grid_network)
    closure = RoadClosure(edge_ids=[0])
    cmp = compare_intervention(synth, closure, n_steps=30, x0=np.full(grid_network.n_edges, 0.1))
    # Closed edge pinned high.
    assert np.isclose(cmp.scenario[-1, 0], 1.0)
    # Neighbours of edge 0 (those adjacent in the network) are more congested than baseline.
    neighbours = np.flatnonzero(grid_network.adjacency[0] > 0)
    assert np.all(cmp.delta[-1, neighbours] > 0)


def test_accident_spikes_then_relaxes(grid_network: RoadNetwork) -> None:
    synth = _synth(grid_network)
    accident = Accident(edge_ids=[2], severity=0.4, start=5, duration=5)
    cmp = compare_intervention(synth, accident, n_steps=40, x0=np.full(grid_network.n_edges, 0.1))
    during = cmp.delta[9, 2]  # within accident window
    after = cmp.delta[-1, 2]  # long after it clears
    assert during > 0.1  # congestion elevated during the accident
    assert after < during  # relaxes back toward baseline afterwards


def test_rainfall_raises_network_wide_congestion(grid_network: RoadNetwork) -> None:
    synth = _synth(grid_network)
    rain = Rainfall(intensity=0.05, start=0, duration=20)
    cmp = compare_intervention(synth, rain, n_steps=20, x0=np.full(grid_network.n_edges, 0.1))
    mean_delta = cmp.mean_delta_over_time()
    assert np.all(mean_delta[1:] > 0)  # every edge, every step, more congested while raining


def test_baseline_matches_seed(grid_network: RoadNetwork) -> None:
    # No intervention -> scenario equals baseline exactly (same seed, no perturbation).
    synth = _synth(grid_network)
    from city_twin.scenarios.interventions import Intervention

    cmp = compare_intervention(synth, Intervention(), n_steps=10)
    assert np.allclose(cmp.delta, 0.0)
