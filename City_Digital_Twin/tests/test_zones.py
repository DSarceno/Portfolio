"""Phase-4 validation: zone-level stress aggregation."""

from __future__ import annotations

import numpy as np

from city_twin.ingestion.network import RoadNetwork
from city_twin.simulation.zones import length_weighted_zone_stress, zone_stress


def test_zone_mean_stress() -> None:
    congestion = np.array([0.2, 0.4, 0.8, 0.6])
    edge_zone = np.array([0, 0, 1, 1])
    stress = zone_stress(congestion, edge_zone, aggfunc="mean")
    assert set(stress) == {0, 1}
    assert np.isclose(stress[0], 0.3)
    assert np.isclose(stress[1], 0.7)


def test_zone_max_surfaces_hotspot() -> None:
    congestion = np.array([0.1, 0.95, 0.2])
    edge_zone = np.array([0, 0, 1])
    stress = zone_stress(congestion, edge_zone, aggfunc="max")
    assert np.isclose(stress[0], 0.95)  # hotspot surfaced


def test_zone_under_load_has_higher_stress() -> None:
    edge_zone = np.array([0, 0, 1, 1])
    light = np.array([0.1, 0.1, 0.1, 0.1])
    loaded = np.array([0.1, 0.1, 0.8, 0.9])  # zone 1 under load
    base = zone_stress(light, edge_zone)
    stressed = zone_stress(loaded, edge_zone)
    assert stressed[1] > base[1]
    assert stressed[1] > stressed[0]


def test_length_weighting_counts_long_segments_more() -> None:
    graph_edges = np.array([0, 0])
    congestion = np.array([0.2, 0.8])
    # Build a tiny network with unequal segment lengths in the same zone.
    import networkx as nx

    g = nx.DiGraph()
    g.add_edge(0, 1, length=100.0, free_flow_speed=50.0)
    g.add_edge(2, 3, length=900.0, free_flow_speed=50.0)
    net = RoadNetwork.from_digraph(g)
    weighted = length_weighted_zone_stress(net, congestion, graph_edges)
    # Long congested segment dominates -> weighted mean well above the plain mean (0.5).
    assert weighted[0] > 0.6


def test_rejects_mismatched_shapes() -> None:
    import pytest

    with pytest.raises(ValueError):
        zone_stress(np.zeros(3), np.zeros(4))
