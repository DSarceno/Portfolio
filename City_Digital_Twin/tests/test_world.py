"""Phase-4 validation: saturation and bottleneck propagation in the world model."""

from __future__ import annotations

import numpy as np

from city_twin.ingestion.network import RoadNetwork
from city_twin.observation.synthetic import SyntheticObservations
from city_twin.scenarios.compare import compare_intervention
from city_twin.scenarios.interventions import Accident
from city_twin.simulation.world import WorldModel


def test_congestion_stays_bounded(line_network: RoadNetwork) -> None:
    # Even with strong growth/backpressure and near-saturated start, x stays in [0, 1].
    world = WorldModel(line_network, decay=0.0, growth=0.9, backpressure=0.9)
    x = np.full(line_network.n_edges, 0.99)
    for _ in range(50):
        x = world.step(x)
        assert np.all(x >= 0.0) and np.all(x <= 1.0)


def test_saturation_is_self_reinforcing_then_vanishes(line_network: RoadNetwork) -> None:
    world = WorldModel(line_network, decay=0.0, diffusion=0.0, growth=0.2, backpressure=0.0)
    # Mid-range congestion grows (self-reinforcing).
    mid = np.full(line_network.n_edges, 0.3)
    assert np.all(world.step(mid) > mid)
    # The growth term vanishes at the free-flow and fully-saturated limits.
    assert np.allclose(world.derivative(np.zeros(line_network.n_edges)), 0.0)
    assert np.allclose(world.derivative(np.ones(line_network.n_edges)), 0.0)


def test_bottleneck_backs_up_upstream_not_downstream(line_network: RoadNetwork) -> None:
    # Path 0->1->2->3->4; an accident on edge 2 = (2,3) should back up its UPSTREAM
    # neighbour edge 1 = (1,2), but not its downstream neighbour edge 3 = (3,4).
    # Diffusion is off so only directional back-pressure propagates.
    world = WorldModel(line_network, decay=0.05, diffusion=0.0, growth=0.0, backpressure=0.3)
    synth = SyntheticObservations(line_network, world, process_noise=0.0, seed=0)
    accident = Accident(edge_ids=[2], severity=0.5, start=0, duration=15)
    cmp = compare_intervention(synth, accident, n_steps=20, x0=np.full(line_network.n_edges, 0.1))

    upstream_delta = cmp.delta[-1, 1]  # edge (1,2), upstream of the accident
    downstream_delta = cmp.delta[-1, 3]  # edge (3,4), downstream of the accident
    assert upstream_delta > 0.05  # congestion backs up upstream
    assert downstream_delta < upstream_delta  # propagation is directional, not symmetric
    assert np.isclose(downstream_delta, 0.0, atol=1e-6)


def test_world_model_satisfies_step_interface(grid_network: RoadNetwork) -> None:
    from city_twin.dynamics.base import StepModel

    world = WorldModel(grid_network)
    assert isinstance(world, StepModel)
    out = world.step(np.full(grid_network.n_edges, 0.2))
    assert out.shape == (grid_network.n_edges,)
