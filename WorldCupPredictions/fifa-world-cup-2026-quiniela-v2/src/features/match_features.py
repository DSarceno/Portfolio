"""Match-level feature builders."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.constants import (
    OUTCOME_AWAY_WIN,
    OUTCOME_DRAW,
    OUTCOME_HOME_WIN,
    STAGE_GROUP,
    KNOCKOUT_STAGES,
)
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def compute_match_features(
    matches: pd.DataFrame, team_features: pd.DataFrame
) -> pd.DataFrame:
    """Build the per-match feature table joining each match with both teams' features.

    Args:
        matches: Canonical match table.
        team_features: Output of :func:`compute_team_features`.

    Returns:
        DataFrame with one row per match and difference features.
    """
    if matches.empty:
        return pd.DataFrame()

    feats = matches.copy()
    feats["date"] = pd.to_datetime(feats["date"], errors="coerce")

    team_features = team_features.set_index("team")

    for suffix, team_col in [("a", "team_a"), ("b", "team_b")]:
        joined = team_features.reindex(feats[team_col].values).reset_index(drop=True)
        joined.columns = [f"{c}_{suffix}" for c in joined.columns]
        feats = pd.concat([feats.reset_index(drop=True), joined], axis=1)

    feats["elo_diff"] = feats.get("elo_pre_match_a", 0) - feats.get("elo_pre_match_b", 0)
    feats["pi_diff"] = feats.get("pi_rating_pre_match_a", 0) - feats.get(
        "pi_rating_pre_match_b", 0
    )
    feats["form_diff"] = feats.get("rolling_form_points_last_5_a", 0) - feats.get(
        "rolling_form_points_last_5_b", 0
    )
    feats["fifa_points_diff"] = feats.get("fifa_rank_points_a", 0) - feats.get(
        "fifa_rank_points_b", 0
    )
    feats["attack_diff"] = feats.get("attack_strength_a", 0) - feats.get("defense_strength_b", 0)
    feats["defense_diff"] = feats.get("defense_strength_a", 0) - feats.get("attack_strength_b", 0)

    stage_str = feats["stage"].astype(str).str.lower()
    feats["stage_group"] = stage_str.str.contains("group").astype(int)
    feats["stage_knockout"] = stage_str.apply(
        lambda s: int(any(k in s for k in KNOCKOUT_STAGES) or "round" in s or "final" in s)
    )
    feats["round_of_tournament"] = feats["stage_group"] + 2 * feats["stage_knockout"]

    feats["host_continent_advantage"] = feats.get("host_flag_a", 0) - feats.get("host_flag_b", 0)
    feats["same_confederation_flag"] = (
        feats.get("confederation_strength_index_a", 0)
        == feats.get("confederation_strength_index_b", 0)
    ).astype(int)

    feats["outcome"] = _outcome_label(feats)
    return feats


def _outcome_label(df: pd.DataFrame) -> pd.Series:
    """Compute the H/D/A label given ``score_a`` and ``score_b``.

    Args:
        df: DataFrame with score columns.

    Returns:
        Series of outcome strings; NaN when scores are missing.
    """
    score_a = pd.to_numeric(df.get("score_a"), errors="coerce")
    score_b = pd.to_numeric(df.get("score_b"), errors="coerce")
    out = pd.Series([np.nan] * len(df), index=df.index, dtype=object)
    mask = score_a.notna() & score_b.notna()
    out.loc[mask & (score_a > score_b)] = OUTCOME_HOME_WIN
    out.loc[mask & (score_a == score_b)] = OUTCOME_DRAW
    out.loc[mask & (score_a < score_b)] = OUTCOME_AWAY_WIN
    return out
