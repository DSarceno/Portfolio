"""Tests for the road-network representation."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from city_twin.ingestion.network import RoadNetwork


def test_edge_index_is_stable_and_sorted(line_network: RoadNetwork) -> None:
    assert line_network.edges == ((0, 1), (1, 2), (2, 3), (3, 4))
    # Rebuilding from an equivalent graph yields the same canonical order.
    graph = nx.DiGraph()
    for u, v in [(3, 4), (0, 1), (2, 3), (1, 2)]:  # inserted out of order
        graph.add_edge(u, v, length=500.0, free_flow_speed=50.0)
    rebuilt = RoadNetwork.from_digraph(graph)
    assert rebuilt.edges == line_network.edges


def test_attributes_present_and_typed(line_network: RoadNetwork) -> None:
    assert line_network.n_edges == 4
    assert line_network.length_m.shape == (4,)
    assert np.all(line_network.free_flow_speed_kmh == 50.0)
    assert np.all(line_network.lanes == 2.0)


def test_missing_speed_uses_default() -> None:
    graph = nx.DiGraph()
    graph.add_edge(0, 1, length=100.0)  # no free_flow_speed
    net = RoadNetwork.from_digraph(graph, default_speed_kmh=45.0)
    assert net.free_flow_speed_kmh[0] == 45.0
    assert np.isnan(net.lanes[0])  # unknown lanes -> NaN


def test_adjacency_reflects_consecutive_edges(line_network: RoadNetwork) -> None:
    a = line_network.adjacency
    # consecutive path edges are adjacent (symmetric), non-consecutive are not
    assert a[0, 1] == 1.0 and a[1, 0] == 1.0
    assert a[0, 2] == 0.0
    assert np.all(np.diag(a) == 0.0)  # no self-loops


def test_laplacian_is_symmetric_psd(grid_network: RoadNetwork) -> None:
    laplacian = grid_network.laplacian()
    assert np.allclose(laplacian, laplacian.T)
    eigenvalues = np.linalg.eigvalsh(laplacian)
    assert np.all(eigenvalues >= -1e-9)  # positive semidefinite
    assert np.isclose(eigenvalues.min(), 0.0, atol=1e-9)  # connected -> single zero mode


def test_rejects_undirected_graph() -> None:
    with pytest.raises(ValueError):
        RoadNetwork.from_digraph(nx.Graph())  # type: ignore[arg-type]
