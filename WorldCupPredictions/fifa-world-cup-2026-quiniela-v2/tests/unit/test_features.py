"""Tests for the feature layer."""

from __future__ import annotations

import pandas as pd

from src.features.build_features import build_match_feature_matrix, select_feature_columns
from src.features.fatigue_features import compute_fatigue_features
from src.features.match_features import compute_match_features
from src.features.market_features import compute_market_features
from src.features.team_features import compute_team_features
from src.ratings.rating_ensemble import RatingEnsemble


def test_team_features_have_required_columns(synthetic_matches: pd.DataFrame) -> None:
    composite = RatingEnsemble().fit(synthetic_matches).composite_table()
    feats = compute_team_features(synthetic_matches, pd.DataFrame(), composite)
    required = {"elo_pre_match", "attack_strength", "defense_strength", "draw_rate", "host_flag"}
    assert required.issubset(set(feats.columns))


def test_match_features_outcome_label(synthetic_matches: pd.DataFrame) -> None:
    composite = RatingEnsemble().fit(synthetic_matches).composite_table()
    team_feats = compute_team_features(synthetic_matches, pd.DataFrame(), composite)
    match_feats = compute_match_features(synthetic_matches, team_feats)
    assert "outcome" in match_feats.columns
    assert match_feats["outcome"].isin(["H", "D", "A"]).all()


def test_market_and_fatigue_features(synthetic_matches: pd.DataFrame) -> None:
    composite = RatingEnsemble().fit(synthetic_matches).composite_table()
    team_feats = compute_team_features(synthetic_matches, pd.DataFrame(), composite)
    match_feats = compute_match_features(synthetic_matches, team_feats)
    market = compute_market_features(match_feats)
    assert "public_bias_proxy" in market.columns
    fatigue = compute_fatigue_features(market)
    assert "rest_diff" in fatigue.columns


def test_full_feature_matrix(synthetic_matches: pd.DataFrame) -> None:
    composite = RatingEnsemble().fit(synthetic_matches).composite_table()
    matrix = build_match_feature_matrix(
        matches=synthetic_matches,
        rankings=pd.DataFrame(),
        composite_table=composite,
    )
    assert len(matrix) == len(synthetic_matches)
    cols = select_feature_columns(matrix)
    assert "elo_diff" in cols
