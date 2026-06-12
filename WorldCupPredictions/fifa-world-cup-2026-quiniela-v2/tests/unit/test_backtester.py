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


def test_collect_holdout_predictions_skips_when_no_history() -> None:
    import pandas as pd

    from src.training.backtester import collect_holdout_predictions

    # All matches in 2022; holding out 2022 leaves no training history -> skip.
    dates = pd.date_range("2022-01-01", periods=20, freq="5D")
    matches = pd.DataFrame(
        {
            "match_id": range(20),
            "date": dates,
            "competition": "WC",
            "season": "2022",
            "stage": "group",
            "team_a": ["Alpha", "Beta"] * 10,
            "team_b": ["Gamma", "Delta"] * 10,
            "score_a": 1,
            "score_b": 0,
            "neutral_venue": True,
            "host_country": "",
            "source": "test",
        }
    )
    proba, y_true = collect_holdout_predictions(matches, year=2022, min_train_matches=5)
    assert proba.shape == (0, 3)
    assert y_true == []


def test_collect_holdout_predictions_real_data() -> None:
    from pathlib import Path

    import pandas as pd
    import pytest

    from src.training.backtester import collect_holdout_predictions

    path = Path("data/interim/matches_unified.csv")
    if not path.exists():
        pytest.skip("canonical match table not available")
    matches = pd.read_csv(path)
    proba, y_true = collect_holdout_predictions(matches, year=2022, competition="WC")
    assert proba.shape[1] == 3
    assert len(y_true) == proba.shape[0] > 0
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert set(y_true).issubset({"H", "D", "A"})
