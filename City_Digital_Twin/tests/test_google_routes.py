"""Tests for the Google Routes observation adapter (mocked transport, no live calls)."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from city_twin.ingestion.network import RoadNetwork
from city_twin.observation.adapters.google_routes import (
    GoogleRoutesAdapter,
    UrllibTransport,
    _parse_duration_seconds,
)

# Network geometry: a 0->1->2->3 path with known coordinates; every edge is 100 m at a
# 36 km/h free-flow speed, so free-flow duration is 10 s and speed = 100/duration*3.6.
NODE_COORDS = {0: (14.60, -90.51), 1: (14.61, -90.51), 2: (14.62, -90.51), 3: (14.63, -90.51)}
OK = "ROUTE_EXISTS"


def _coord_network(*, with_uncoorded_edge: bool = False) -> RoadNetwork:
    graph = nx.DiGraph()
    for node, (lat, lon) in NODE_COORDS.items():
        graph.add_node(node, y=lat, x=lon)  # OSM convention: x=lon, y=lat
    for u, v in [(0, 1), (1, 2), (2, 3)]:
        graph.add_edge(u, v, length=100.0, free_flow_speed=36.0)
    if with_uncoorded_edge:
        graph.add_node(4)  # node 4 has no coordinates
        graph.add_edge(3, 4, length=100.0, free_flow_speed=36.0)
    return RoadNetwork.from_digraph(graph)


class FakeTransport:
    """Returns a canned computeRouteMatrix grid; diagonal durations keyed by origin lat."""

    def __init__(self, by_origin_lat: dict[float, tuple[str | None, str]]) -> None:
        self.by_origin_lat = by_origin_lat
        self.calls = 0

    def post(self, url, headers, body):  # noqa: ANN001
        self.calls += 1
        origins = body["origins"]
        destinations = body["destinations"]
        elements = []
        for i, origin in enumerate(origins):
            lat = round(origin["waypoint"]["location"]["latLng"]["latitude"], 5)
            for j in range(len(destinations)):
                if i == j:
                    duration, condition = self.by_origin_lat[lat]
                else:
                    duration, condition = "5s", "ROUTE_EXISTS"
                element = {"originIndex": i, "destinationIndex": j, "condition": condition}
                if duration is not None:
                    element["duration"] = duration
                elements.append(element)
        return elements


def _transport(d0, d1, d2) -> FakeTransport:  # noqa: ANN001
    return FakeTransport({14.6: d0, 14.61: d1, 14.62: d2})


def test_fetch_derives_congestion_from_duration() -> None:
    net = _coord_network()
    transport = _transport(("10s", OK), ("20s", OK), ("40s", OK))
    obs = GoogleRoutesAdapter(net, transport=transport).fetch(step=3)
    obs.validate(net.n_edges)
    assert obs.step == 3
    assert np.all(obs.mask)
    # 10s -> free flow (0), 20s -> half speed (0.5), 40s -> quarter speed (0.75)
    assert np.allclose(obs.congestion, [0.0, 0.5, 0.75])


def test_route_not_found_is_unobserved() -> None:
    net = _coord_network()
    transport = _transport(("10s", OK), (None, "ROUTE_NOT_FOUND"), ("40s", OK))
    obs = GoogleRoutesAdapter(net, transport=transport).fetch()
    assert obs.mask.tolist() == [True, False, True]


def test_edges_without_coordinates_are_skipped() -> None:
    net = _coord_network(with_uncoorded_edge=True)
    transport = _transport(("10s", OK), ("10s", OK), ("10s", OK))
    obs = GoogleRoutesAdapter(net, transport=transport).fetch()
    assert not net.has_coordinates(3)  # the (3,4) edge lacks a coordinate
    assert obs.mask[3] == np.False_
    assert obs.mask[:3].all()


def test_chunking_issues_multiple_calls() -> None:
    net = _coord_network()
    transport = _transport(("10s", OK), ("10s", OK), ("10s", OK))
    GoogleRoutesAdapter(net, transport=transport, chunk=1).fetch()
    assert transport.calls == 3  # one call per edge


def test_urllib_transport_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("GOOGLE_MAPS_API_KEY", raising=False)
    with pytest.raises(ValueError):
        UrllibTransport("")
    with pytest.raises(ValueError):
        GoogleRoutesAdapter(_coord_network())  # no key, no transport


def test_parse_duration() -> None:
    assert _parse_duration_seconds("123s") == 123.0
    assert _parse_duration_seconds("45.5s") == 45.5
    assert _parse_duration_seconds(12) == 12.0
    assert _parse_duration_seconds(None) is None
    assert _parse_duration_seconds("0s") is None  # non-positive rejected
    assert _parse_duration_seconds("bad") is None
