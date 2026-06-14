"""Google Routes API observation adapter — the first real observation source.

For each directed edge ``(u, v)`` the adapter queries Google's Routes API
``computeRouteMatrix`` (``TRAFFIC_AWARE``) with the edge's endpoint coordinates, reads the
in-traffic ``duration``, and derives

    speed_kmh   = edge_length_m / duration_s * 3.6
    congestion  = 1 - speed / free_flow_speed        (clipped to [0, 1])

This is the dataset-interpretation assumption (confirmed): an edge's speed is its own
length divided by the in-traffic duration of the point-to-point query along it.

The HTTP call sits behind the :class:`Transport` boundary so the adapter is fully testable
with canned responses — **no live API calls or API key are needed to run the test suite**.
Real use reads the key from the ``GOOGLE_MAPS_API_KEY`` environment variable; it is never
hardcoded. Output is a standard :class:`~city_twin.observation.schema.Observation`, so it
feeds the Kalman/EKF unchanged.

Note on cost: ``computeRouteMatrix`` returns the full origins x destinations grid, but only
the diagonal (origin tail i paired with destination head i) is an actual edge, so each
chunk of ``k`` edges costs ``k * k`` matrix elements. ``chunk`` is capped at 25 (the 625
element-per-call limit). For large networks the single-route variant would be cheaper; this
matrix form is what was chosen for batching.
"""

from __future__ import annotations

import json
import os
from typing import Any, Protocol, runtime_checkable
from urllib.request import Request, urlopen

import numpy as np

from city_twin.ingestion.network import RoadNetwork
from city_twin.observation.schema import Observation, speed_to_congestion

ROUTE_MATRIX_URL = "https://routes.googleapis.com/distanceMatrix/v2:computeRouteMatrix"
FIELD_MASK = "originIndex,destinationIndex,duration,distanceMeters,condition"
MAX_CHUNK = 25  # 25 x 25 = 625 elements, the per-call matrix limit


@runtime_checkable
class Transport(Protocol):
    """Minimal HTTP POST boundary returning parsed JSON. Lets tests inject fakes."""

    def post(self, url: str, headers: dict[str, str], body: dict[str, Any]) -> Any: ...


class UrllibTransport:
    """Default transport using the standard library. Requires an API key."""

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise ValueError("Google API key is required for live requests")
        self.api_key = api_key

    def post(self, url: str, headers: dict[str, str], body: dict[str, Any]) -> Any:
        headers = {**headers, "X-Goog-Api-Key": self.api_key}
        request = Request(
            url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST"
        )
        with urlopen(request) as response:  # noqa: S310 - fixed Google endpoint
            return json.loads(response.read().decode("utf-8"))


def _parse_duration_seconds(raw: Any) -> float | None:
    """Parse a protobuf duration like ``"123s"`` (or a number) to seconds."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    text = str(raw).strip()
    if text.endswith("s"):
        text = text[:-1]
    try:
        value = float(text)
    except ValueError:
        return None
    return value if value > 0 else None


class GoogleRoutesAdapter:
    """Fetch noisy real observations of per-edge congestion from the Routes API."""

    def __init__(
        self,
        network: RoadNetwork,
        *,
        api_key: str | None = None,
        transport: Transport | None = None,
        chunk: int = MAX_CHUNK,
    ) -> None:
        self.network = network
        self.chunk = max(1, min(chunk, MAX_CHUNK))
        if transport is not None:
            self.transport: Transport = transport
        else:
            key = api_key if api_key is not None else os.environ.get("GOOGLE_MAPS_API_KEY", "")
            self.transport = UrllibTransport(key)

    def fetch(self, *, edge_ids: list[int] | None = None, step: int = 0) -> Observation:
        """Query the API and return an :class:`Observation` for ``step``.

        Edges without known coordinates, or for which no route is found, are left
        unobserved (``mask`` False). The estimator infers them.
        """
        n = self.network.n_edges
        if edge_ids is None:
            candidates = [i for i in range(n) if self.network.has_coordinates(i)]
        else:
            candidates = [i for i in edge_ids if self.network.has_coordinates(i)]

        congestion = np.zeros(n)
        mask = np.zeros(n, dtype=bool)

        for start in range(0, len(candidates), self.chunk):
            block = candidates[start : start + self.chunk]
            elements = self._request_block(block)
            for local_index, edge_id in enumerate(block):
                element = elements.get((local_index, local_index))
                speed = self._element_speed(edge_id, element)
                if speed is not None:
                    congestion[edge_id] = speed_to_congestion(
                        speed, self.network.free_flow_speed_kmh[edge_id]
                    )
                    mask[edge_id] = True

        return Observation(step=step, congestion=congestion, mask=mask)

    def _request_block(self, block: list[int]) -> dict[tuple[int, int], dict[str, Any]]:
        origins = [self._waypoint(self.network.origin_latlon[i]) for i in block]
        destinations = [self._waypoint(self.network.dest_latlon[i]) for i in block]
        body = {
            "origins": origins,
            "destinations": destinations,
            "travelMode": "DRIVE",
            "routingPreference": "TRAFFIC_AWARE",
        }
        headers = {"Content-Type": "application/json", "X-Goog-FieldMask": FIELD_MASK}
        response = self.transport.post(ROUTE_MATRIX_URL, headers, body)
        elements: dict[tuple[int, int], dict[str, Any]] = {}
        for element in response or []:
            origin_index = element.get("originIndex")
            dest_index = element.get("destinationIndex")
            if origin_index is not None and dest_index is not None:
                elements[(origin_index, dest_index)] = element
        return elements

    def _element_speed(self, edge_id: int, element: dict[str, Any] | None) -> float | None:
        if element is None:
            return None
        condition = element.get("condition")
        if condition is not None and condition != "ROUTE_EXISTS":
            return None
        duration = _parse_duration_seconds(element.get("duration"))
        if duration is None:
            return None
        length_m = float(self.network.length_m[edge_id])
        return length_m / duration * 3.6  # m/s -> km/h

    @staticmethod
    def _waypoint(latlon: np.ndarray) -> dict[str, Any]:
        return {
            "waypoint": {
                "location": {
                    "latLng": {"latitude": float(latlon[0]), "longitude": float(latlon[1])}
                }
            }
        }
