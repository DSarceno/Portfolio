"""Tests for rating modules."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.ratings.elo import EloRating
from src.ratings.form_rating import FormRating
from src.ratings.pi_rating import PIRating
from src.ratings.rating_ensemble import RatingEnsemble
from src.ratings.shrinkage import RatingShrinker


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


def _uefa_strength_table(n: int = 10) -> tuple[dict[str, float], dict[str, int]]:
    """Build a synthetic UEFA-only strength table for elite classification."""
    uefa = [
        "Spain",
        "France",
        "Germany",
        "England",
        "Portugal",
        "Italy",
        "Croatia",
        "Poland",
        "Norway",
        "Finland",
    ][:n]
    # Strictly decreasing strength so the ranking is unambiguous.
    strength = {team: 1800.0 - 30.0 * i for i, team in enumerate(uefa)}
    counts = {team: 100 for team in uefa}
    return strength, counts


def test_classify_elite_top_fraction() -> None:
    shrinker = RatingShrinker(elite_fraction=0.30, min_matches_for_elite=20)
    strength, counts = _uefa_strength_table(10)
    elite = shrinker.classify_elite(strength, counts)
    # Top 30% of 10 teams -> the 3 strongest are elite.
    assert elite["Spain"] is True
    assert elite["France"] is True
    assert elite["Germany"] is True
    assert elite["Finland"] is False


def test_classify_elite_match_count_gate_blocks_minnow() -> None:
    shrinker = RatingShrinker(elite_fraction=0.30, min_matches_for_elite=20)
    strength, counts = _uefa_strength_table(10)
    # An inflated minnow: highest strength but only 5 matches -> must stay regular.
    strength["Gibraltar"] = 9999.0
    counts["Gibraltar"] = 5
    elite = shrinker.classify_elite(strength, counts)
    assert elite.get("Gibraltar", False) is False


def test_elite_elo_prior_above_regular() -> None:
    shrinker = RatingShrinker(mixture_prior=True)
    assert shrinker.get_elo_prior("Spain", is_elite=True) > shrinker.get_elo_prior(
        "Spain", is_elite=False
    )


def test_mixture_disabled_uses_single_prior() -> None:
    shrinker = RatingShrinker(mixture_prior=False)
    # With the mixture off, is_elite must not change the prior.
    assert shrinker.get_elo_prior("Spain", is_elite=True) == shrinker.get_elo_prior(
        "Spain", is_elite=False
    )


def _uefa_elo_table() -> tuple[dict[str, float], dict[str, int]]:
    ratings = {"France": 1700.0}
    # Lower-rated established UEFA teams so France sits in the top fraction.
    for i, team in enumerate(["Spain", "Germany", "Italy", "Croatia", "Poland", "Norway"]):
        ratings[team] = 1480.0 - i
    counts = {t: 100 for t in ratings}
    return ratings, counts


def test_mixture_shrinks_elite_toward_higher_prior_than_single() -> None:
    # Same elite team, two shrinkers: mixture must leave it higher than the
    # single-prior path, because the elite Elo prior (1740) exceeds the single
    # UEFA prior (1620).
    ratings_a, counts = _uefa_elo_table()
    elo_mix = EloRating()
    elo_mix.ratings.update(ratings_a)
    RatingShrinker(k_elo=30.0, mixture_prior=True, elite_fraction=0.30).shrink_elo(elo_mix, counts)

    elo_single = EloRating()
    elo_single.ratings.update(dict(ratings_a))
    RatingShrinker(k_elo=30.0, mixture_prior=False).shrink_elo(elo_single, counts)

    assert elo_mix.ratings["France"] > elo_single.ratings["France"]
