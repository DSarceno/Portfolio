"""Tournament-state features (updated daily during the World Cup)."""

from __future__ import annotations

import pandas as pd

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def compute_tournament_state_features(
    state_table: pd.DataFrame, teams: list[str] | None = None
) -> pd.DataFrame:
    """Return a per-team tournament-state feature table.

    Args:
        state_table: Output of :meth:`TournamentUpdater.world_cup_state`.
        teams: Optional list of teams to ensure are present (with zeros).

    Returns:
        DataFrame indexed by ``team``.
    """
    expected_cols = [
        "world_cup_points_so_far",
        "world_cup_goal_diff_so_far",
        "world_cup_goals_for_so_far",
        "world_cup_goals_against_so_far",
        "world_cup_clean_sheets",
        "world_cup_form_last_1",
        "world_cup_form_last_2",
        "cards_accumulated",
        "suspension_risk_proxy",
    ]
    if state_table.empty:
        if teams is None:
            return pd.DataFrame(columns=["team", *expected_cols])
        df = pd.DataFrame({"team": teams})
        for c in expected_cols:
            df[c] = 0.0
        return df

    df = state_table.copy()
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0.0
    if teams is not None:
        missing = [t for t in teams if t not in df["team"].values]
        if missing:
            extras = pd.DataFrame({"team": missing})
            for c in expected_cols:
                extras[c] = 0.0
            df = pd.concat([df, extras], ignore_index=True)
    return df[["team", *expected_cols]]


def add_tournament_state_to_matches(
    match_features: pd.DataFrame, state_features: pd.DataFrame
) -> pd.DataFrame:
    """Join tournament-state features for both teams onto *match_features*.

    Args:
        match_features: Output of :func:`compute_match_features`.
        state_features: Output of :func:`compute_tournament_state_features`.

    Returns:
        DataFrame with ``_state_a`` and ``_state_b`` suffixed columns plus diffs.
    """
    if match_features.empty:
        return match_features
    if state_features.empty:
        return match_features

    state_features = state_features.set_index("team")
    out = match_features.copy()
    for suffix, team_col in [("state_a", "team_a"), ("state_b", "team_b")]:
        joined = state_features.reindex(out[team_col].values).reset_index(drop=True)
        joined.columns = [f"{c}_{suffix}" for c in joined.columns]
        out = pd.concat([out.reset_index(drop=True), joined], axis=1)

    out["wc_points_diff"] = out.get("world_cup_points_so_far_state_a", 0).fillna(0) - out.get(
        "world_cup_points_so_far_state_b", 0
    ).fillna(0)
    out["wc_goal_diff_diff"] = out.get("world_cup_goal_diff_so_far_state_a", 0).fillna(
        0
    ) - out.get("world_cup_goal_diff_so_far_state_b", 0).fillna(0)
    return out
