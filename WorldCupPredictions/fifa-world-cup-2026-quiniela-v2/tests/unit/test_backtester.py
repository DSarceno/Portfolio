"""Tests for the backtesting layer."""

from __future__ import annotations

import numpy as np

from src.training.backtester import _ranked_probability_score


def test_rps_zero_for_perfect_prediction() -> None:
    proba = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert _ranked_probability_score(["H", "A"], proba) == 0.0


def test_rps_penalizes_distant_miss_more_than_near_miss() -> None:
    # True outcome is a home win. Predicting away (2 categories off) must score
    # worse than predicting a draw (1 category off).
    near = np.array([[0.0, 1.0, 0.0]])  # all mass on draw
    far = np.array([[0.0, 0.0, 1.0]])  # all mass on away
    assert _ranked_probability_score(["H"], near) < _ranked_probability_score(["H"], far)


def test_rps_ignores_unknown_labels() -> None:
    proba = np.array([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0]])
    # Second label is junk and should be skipped, leaving a perfect first row.
    assert _ranked_probability_score(["H", "?"], proba) == 0.0
