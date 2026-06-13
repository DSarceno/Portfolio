"""Road-network representation for the Digital Twin.

``RoadNetwork`` wraps a directed road graph as a set of ordered directed edges (road
segments). The *edge ordering is canonical and stable* — its integer index defines the
index of every state and observation vector in the system (see ARCHITECTURE.md).

Two helpers build a network:

* :meth:`RoadNetwork.from_digraph` — from any NetworkX (multi)digraph; used by fixtures
  and tests (no network access).
* :meth:`RoadNetwork.from_osm` — from OpenStreetMap via OSMnx (lazy import); used in
  production. Not exercised by unit tests, which must not hit the network.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np

# Free-flow speed (km/h) defaults per OSM highway class for the strategic subgraph.
DEFAULT_FREE_FLOW_KMH: dict[str, float] = {
    "motorway": 90.0,
    "motorway_link": 60.0,
    "trunk": 80.0,
    "trunk_link": 50.0,
    "primary": 60.0,
    "primary_link": 40.0,
    "secondary": 50.0,
    "secondary_link": 40.0,
}

STRATEGIC_HIGHWAYS: frozenset[str] = frozenset(DEFAULT_FREE_FLOW_KMH)

DEFAULT_SPEED_KMH = 50.0


@dataclass(frozen=True)
class RoadNetwork:
    """An ordered set of directed road segments with per-edge attributes.

    Attributes are aligned arrays of length ``n_edges`` indexed by the canonical edge
    order in :attr:`edges`.
    """

    edges: tuple[tuple, ...]  # canonical ordered edge keys, e.g. (u, v) or (u, v, key)
    length_m: np.ndarray  # (N,)
    free_flow_speed_kmh: np.ndarray  # (N,)
    lanes: np.ndarray  # (N,) float; NaN where unknown
    adjacency: np.ndarray  # (N, N) symmetric 0/1, no self-loops

    def __post_init__(self) -> None:
        n = len(self.edges)
        for name, arr in (
            ("length_m", self.length_m),
            ("free_flow_speed_kmh", self.free_flow_speed_kmh),
            ("lanes", self.lanes),
        ):
            if arr.shape != (n,):
                raise ValueError(f"{name} must have shape ({n},), got {arr.shape}")
        if self.adjacency.shape != (n, n):
            raise ValueError(f"adjacency must have shape ({n}, {n})")
        if np.any(self.free_flow_speed_kmh <= 0):
            raise ValueError("free_flow_speed_kmh must be strictly positive")

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    def laplacian(self) -> np.ndarray:
        """Symmetric graph Laplacian ``L = D - A`` over the edge adjacency.

        Used by the dynamics layer to propagate congestion between connected segments.
        ``L`` is symmetric positive-semidefinite, which keeps the diffusion update stable.
        """
        degree = self.adjacency.sum(axis=1)
        return np.diag(degree) - self.adjacency

    # ------------------------------------------------------------------ builders
    @classmethod
    def from_digraph(
        cls, graph: nx.DiGraph, *, default_speed_kmh: float = DEFAULT_SPEED_KMH
    ) -> RoadNetwork:
        """Build a :class:`RoadNetwork` from a NetworkX (multi)digraph.

        Each edge may carry ``length`` (m), ``free_flow_speed`` (km/h) and ``lanes``.
        Missing length defaults to 1.0; missing speed to ``default_speed_kmh``.
        """
        if not graph.is_directed():
            raise ValueError("RoadNetwork requires a directed graph")

        if graph.is_multigraph():
            triples = sorted(graph.edges(keys=True))
            edge_keys: tuple[tuple, ...] = tuple((u, v, k) for (u, v, k) in triples)
        else:
            pairs = sorted(graph.edges())
            triples = [(u, v, None) for (u, v) in pairs]
            edge_keys = tuple((u, v) for (u, v) in pairs)

        n = len(triples)
        length = np.empty(n)
        free_flow = np.empty(n)
        lanes = np.full(n, np.nan)
        tails = [t[0] for t in triples]
        heads = [t[1] for t in triples]

        for i, (u, v, k) in enumerate(triples):
            data = graph.get_edge_data(u, v, k) if k is not None else graph.get_edge_data(u, v)
            length[i] = float(data.get("length", 1.0))
            free_flow[i] = float(data.get("free_flow_speed", default_speed_kmh))
            raw_lanes = data.get("lanes")
            if isinstance(raw_lanes, list):
                raw_lanes = raw_lanes[0] if raw_lanes else None
            if raw_lanes is not None:
                try:
                    lanes[i] = float(raw_lanes)
                except (TypeError, ValueError):
                    pass

        # Edge i feeds edge j when head(i) == tail(j); symmetrize for diffusion.
        out_edges: dict = defaultdict(list)
        for j, u in enumerate(tails):
            out_edges[u].append(j)
        adjacency = np.zeros((n, n))
        for i, v in enumerate(heads):
            for j in out_edges.get(v, ()):
                if i != j:
                    adjacency[i, j] = 1.0
        adjacency = np.maximum(adjacency, adjacency.T)
        np.fill_diagonal(adjacency, 0.0)

        return cls(edge_keys, length, free_flow, lanes, adjacency)

    @classmethod
    def from_osm(
        cls, query: str, *, cache_path: str | Path | None = None, simplify: bool = True
    ) -> RoadNetwork:
        """Build a strategic-subgraph :class:`RoadNetwork` from OSM via OSMnx.

        OSMnx is imported lazily so the scientific core and tests do not depend on the
        heavy geospatial stack. The downloaded graph is cached to ``cache_path`` (a
        GraphML file under the gitignored ``data/`` directory) to avoid repeat downloads.
        """
        import osmnx as ox  # lazy, optional dependency

        cache = Path(cache_path) if cache_path else None
        if cache is not None and cache.exists():
            graph = ox.load_graphml(cache)
        else:
            graph = ox.graph_from_place(query, network_type="drive", simplify=simplify)
            graph = strategic_subgraph(graph)
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                ox.save_graphml(graph, cache)

        for _u, _v, _k, data in graph.edges(keys=True, data=True):
            highway = data.get("highway")
            if isinstance(highway, list):
                highway = highway[0] if highway else None
            data.setdefault("length", 1.0)
            data["free_flow_speed"] = DEFAULT_FREE_FLOW_KMH.get(highway, DEFAULT_SPEED_KMH)

        return cls.from_digraph(graph)


def strategic_subgraph(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Return the subgraph of edges whose highway class is in :data:`STRATEGIC_HIGHWAYS`."""
    keep = []
    for u, v, k, data in graph.edges(keys=True, data=True):
        highway = data.get("highway")
        if isinstance(highway, list):
            highway = highway[0] if highway else None
        if highway in STRATEGIC_HIGHWAYS:
            keep.append((u, v, k))
    return graph.edge_subgraph(keep).copy()
