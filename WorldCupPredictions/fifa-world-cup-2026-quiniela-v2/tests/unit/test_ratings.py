"""Tests for rating modules."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.ratings.elo import EloRating
from src.ratings.form_rating import FormRating
from src.ratings.pi_rating import PIRating
from src.ratings.rating_ensemble import RatingEnsemble


def test_elo_expected_score_symmetric() -> None:
    elo = EloRating()
    assert abs(elo.expected_score(1500.0, 1500.0) - 0.5) < 1e-9


def test_elo_update_changes_ratings(synthetic_matches: pd.DataFrame) -> None:
    elo = EloRating().fit(synthetic_matches)
    snapshot = elo.snapshot()
    assert len(snapshot) > 0
    assert snapshot["elo"].max() != snapshot["elo"].min()


def test_pi_rating_runs(synthetic_matches: pd.DataFrame) -> None:
    pi = PIRating().fit(synthetic_matches)
    snap = pi.snapshot()
    assert "pi_combined" in snap.columns
    assert len(snap) > 0


def test_form_rating_runs(synthetic_matches: pd.DataFrame) -> None:
    form = FormRating().fit(synthetic_matches)
    snap = form.snapshot()
    assert "form_score" in snap.columns


def test_rating_ensemble_composite(synthetic_matches: pd.DataFrame) -> None:
    ensemble = RatingEnsemble().fit(synthetic_matches)
    composite = ensemble.composite_table()
    assert "composite_strength" in composite.columns
    assert len(composite) > 0
    assert composite["composite_strength"].std() >= 0


def test_elo_predict_sums_to_one() -> None:
    elo = EloRating()
    elo.ratings["A"] = 1700
    elo.ratings["B"] = 1500
    p = np.array(elo.predict("A", "B"))
    assert abs(p.sum() - 1.0) < 1e-9
    assert (p >= 0).all()
