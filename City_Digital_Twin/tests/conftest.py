"""Shared deterministic fixtures (hand-built graphs, no network access)."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from city_twin.dynamics.graph_diffusion import GraphDiffusionDynamics
from city_twin.ingestion.network import RoadNetwork


@pytest.fixture
def line_network() -> RoadNetwork:
    """A 4-edge directed path: 0->1->2->3->4."""
    graph = nx.DiGraph()
    for u, v in [(0, 1), (1, 2), (2, 3), (3, 4)]:
        graph.add_edge(u, v, length=500.0, free_flow_speed=50.0, lanes=2)
    return RoadNetwork.from_digraph(graph)


@pytest.fixture
def grid_network() -> RoadNetwork:
    """A small directed network with a branch, for propagation tests.

    Topology (directed):
        0->1->2->3 (main corridor)
        1->4->3    (parallel branch off node 1, rejoining at 3)
    """
    graph = nx.DiGraph()
    edges = [(0, 1), (1, 2), (2, 3), (1, 4), (4, 3)]
    for u, v in edges:
        graph.add_edge(u, v, length=400.0, free_flow_speed=60.0, lanes=2)
    return RoadNetwork.from_digraph(graph)


@pytest.fixture
def dynamics(line_network: RoadNetwork) -> GraphDiffusionDynamics:
    return GraphDiffusionDynamics(
        laplacian=line_network.laplacian(), decay=0.05, diffusion=0.02, dt_minutes=5.0
    )


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)
