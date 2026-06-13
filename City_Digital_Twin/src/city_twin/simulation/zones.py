"""Zone-level congestion stress aggregation (Phase 4).

Maps per-edge congestion to per-zone stress so saturation can be read at the scale a
planner cares about ("which zones are approaching congestion saturation?"). Stress is a
length-weighted mean congestion by default — longer congested segments contribute more —
with a max option to surface localized hotspots.
"""

from __future__ import annotations

import numpy as np

from city_twin.ingestion.network import RoadNetwork


def zone_stress(
    congestion: np.ndarray,
    edge_zone: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    aggfunc: str = "mean",
) -> dict:
    """Aggregate per-edge congestion into per-zone stress.

    ``edge_zone[i]`` is the zone id of edge ``i``. ``aggfunc`` is ``"mean"`` (optionally
    ``weights``-weighted) or ``"max"``. Returns ``{zone_id: stress}``.
    """
    congestion = np.asarray(congestion, dtype=float)
    edge_zone = np.asarray(edge_zone)
    if congestion.shape != edge_zone.shape:
        raise ValueError("congestion and edge_zone must have the same shape")
    if aggfunc not in {"mean", "max"}:
        raise ValueError("aggfunc must be 'mean' or 'max'")

    result: dict = {}
    for zone in np.unique(edge_zone):
        mask = edge_zone == zone
        values = congestion[mask]
        if aggfunc == "max":
            result[zone.item()] = float(values.max())
        elif weights is None:
            result[zone.item()] = float(values.mean())
        else:
            w = np.asarray(weights, dtype=float)[mask]
            result[zone.item()] = float(np.average(values, weights=w))
    return result


def length_weighted_zone_stress(
    network: RoadNetwork, congestion: np.ndarray, edge_zone: np.ndarray
) -> dict:
    """Convenience: length-weighted mean congestion per zone."""
    return zone_stress(congestion, edge_zone, weights=network.length_m, aggfunc="mean")
