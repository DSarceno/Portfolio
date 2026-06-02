"""High-level pipeline that assembles every feature group."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.features.fatigue_features import compute_fatigue_features
from src.features.market_features import compute_market_features
from src.features.match_features import compute_match_features
from src.features.team_features import compute_team_features
from src.features.tournament_features import (
    add_tournament_state_to_matches,
    compute_tournament_state_features,
)
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

NUMERIC_FEATURE_COLUMNS = [
    "elo_diff",
    "pi_diff",
    "form_diff",
    "fifa_points_diff",
    "attack_diff",
    "defense_diff",
    "stage_group",
    "stage_knockout",
    "round_of_tournament",
    "host_continent_advantage",
    "same_confederation_flag",
    "rest_days_team_a",
    "rest_days_team_b",
    "rest_diff",
    "travel_burden_proxy_team_a",
    "travel_burden_proxy_team_b",
    "fatigue_index_a",
    "fatigue_index_b",
    "reputation_a",
    "reputation_b",
    "reputation_diff",
    "fifa_overreaction_proxy",
    "public_bias_proxy",
    "wc_points_diff",
    "wc_goal_diff_diff",
]


def build_match_feature_matrix(
    matches: pd.DataFrame,
    rankings: pd.DataFrame,
    composite_table: pd.DataFrame,
    tournament_state: Optional[pd.DataFrame] = None,
    team_continent: Optional[dict[str, str]] = None,
    confederation_map: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    """Build the full feature matrix used by training and inference.

    Args:
        matches: Canonical match table.
        rankings: Pre-tournament FIFA rankings snapshot.
        composite_table: Rating ensemble composite table.
        tournament_state: Optional per-team state aggregates.
        team_continent: Optional ``team -> continent`` mapping for travel proxies.
        confederation_map: Optional ``team -> confederation`` mapping.

    Returns:
        DataFrame with engineered features and the ``outcome`` label.
    """
    team_features = compute_team_features(
        matches=matches,
        rankings=rankings,
        composite_table=composite_table,
        confederation_map=confederation_map,
    )

    match_features = compute_match_features(matches, team_features)
    if match_features.empty:
        return match_features

    match_features = compute_market_features(match_features)
    match_features = compute_fatigue_features(match_features, team_continent=team_continent)

    if tournament_state is not None:
        state_feats = compute_tournament_state_features(tournament_state)
        match_features = add_tournament_state_to_matches(match_features, state_feats)

    match_features = _add_strategy_features(match_features)
    match_features = _fillna_numeric(match_features)
    return match_features


def _add_strategy_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add aggressive-strategy auxiliary features that depend on diffs already present."""
    out = df.copy()
    elo_diff = out.get("elo_diff", pd.Series(np.zeros(len(out)))).fillna(0)
    form_diff = out.get("form_diff", pd.Series(np.zeros(len(out)))).fillna(0)
    out["volatility_index"] = np.clip(1.0 - np.abs(np.tanh(elo_diff / 250.0)), 0, 1)
    out["upset_window_score"] = np.clip(
        out["volatility_index"] - 0.3 * np.tanh(elo_diff / 250.0),
        0,
        1,
    )
    out["draw_trap_score"] = np.clip(
        1.0 - (np.abs(elo_diff) / 400.0) - (np.abs(form_diff) / 10.0), 0, 1
    )
    out["favorite_fragility_score"] = np.clip(
        0.4 - 0.3 * np.tanh(elo_diff / 300.0) + 0.5 * out["volatility_index"], 0, 1
    )
    out["underdog_live_value_score"] = np.clip(
        out["upset_window_score"] * 0.7 + out["favorite_fragility_score"] * 0.3, 0, 1
    )
    out["match_entropy"] = np.clip(1.0 - np.tanh(np.abs(elo_diff) / 250.0), 0, 1)
    out["confidence_gap"] = np.abs(np.tanh(elo_diff / 250.0))
    out["penalty_risk_score"] = out.get("stage_knockout", 0) * 0.45
    return out


def _fillna_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Fill numeric NaNs with 0.0 (after diff computation finishes)."""
    out = df.copy()
    numeric_cols = out.select_dtypes(include="number").columns
    out[numeric_cols] = out[numeric_cols].fillna(0.0)
    return out


def select_feature_columns(df: pd.DataFrame, requested: list[str] | None = None) -> list[str]:
    """Return the subset of *requested* (or :data:`NUMERIC_FEATURE_COLUMNS`) found in *df*.

    Args:
        df: Feature DataFrame.
        requested: Optional explicit list of columns.

    Returns:
        Columns that exist in *df*.
    """
    cols = requested or NUMERIC_FEATURE_COLUMNS
    return [c for c in cols if c in df.columns]
