"""Tests for observation schema and speed<->congestion conversions."""

from __future__ import annotations

import numpy as np
import pytest

from city_twin.observation.schema import (
    Observation,
    congestion_to_speed,
    speed_to_congestion,
)


def test_speed_congestion_roundtrip() -> None:
    free_flow = np.array([50.0, 60.0, 40.0])
    congestion = np.array([0.0, 0.5, 0.9])
    speed = congestion_to_speed(congestion, free_flow)
    assert np.allclose(speed, [50.0, 30.0, 4.0])
    assert np.allclose(speed_to_congestion(speed, free_flow), congestion)


def test_congestion_is_clipped() -> None:
    free_flow = np.array([50.0, 50.0])
    # speed above free flow -> negative deficit clipped to 0; near-zero speed -> ~1
    speeds = np.array([60.0, 1.0])
    congestion = speed_to_congestion(speeds, free_flow)
    assert congestion[0] == 0.0
    assert 0.0 <= congestion[1] <= 1.0


def test_observation_validation_rejects_bad_shape() -> None:
    obs = Observation(step=1, congestion=np.zeros(3), mask=np.array([True, False, True]))
    obs.validate(n_edges=3)
    assert obs.n_observed == 2
    with pytest.raises(ValueError):
        obs.validate(n_edges=4)


def test_observation_validation_rejects_non_bool_mask() -> None:
    obs = Observation(step=1, congestion=np.zeros(2), mask=np.array([1, 0]))
    with pytest.raises(ValueError):
        obs.validate(n_edges=2)
