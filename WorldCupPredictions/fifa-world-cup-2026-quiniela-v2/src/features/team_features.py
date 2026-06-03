"""Team-level feature builders."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.constants import (
    CONFEDERATION_STRENGTH_PRIOR,
    WORLD_CUP_2026_HOSTS,
)
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def compute_team_features(
    matches: pd.DataFrame,
    rankings: pd.DataFrame,
    composite_table: pd.DataFrame,
    confederation_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Build the per-team feature table from historical matches and rankings.

    Args:
        matches: Canonical match table.
        rankings: FIFA rankings snapshot for the cut-off date.
        composite_table: Output of :meth:`RatingEnsemble.composite_table`.
        confederation_map: Optional mapping ``team -> confederation``.

    Returns:
        DataFrame indexed by ``team`` with strength / experience features.
    """
    confederation_map = confederation_map or {}
    team_a = matches.get("team_a", pd.Series(dtype=str)).dropna().astype(str)
    team_b = matches.get("team_b", pd.Series(dtype=str)).dropna().astype(str)
    teams = pd.Index(sorted(set(team_a).union(team_b)))
    feats = pd.DataFrame(index=teams)
    feats.index.name = "team"

    if not rankings.empty and "team" in rankings.columns:
        rank_lookup = rankings.set_index("team")
        feats["fifa_rank_points"] = feats.index.map(rank_lookup["points"].to_dict()).astype(float)
        feats["fifa_rank_position"] = feats.index.map(rank_lookup["rank"].to_dict()).astype(float)
    else:
        feats["fifa_rank_points"] = np.nan
        feats["fifa_rank_position"] = np.nan

    composite_lookup = composite_table.set_index("team")
    feats["elo_pre_match"] = feats.index.map(composite_lookup["elo"].to_dict()).astype(float)
    feats["pi_rating_pre_match"] = feats.index.map(
        composite_lookup["pi_combined"].to_dict()
    ).astype(float)
    feats["rolling_form_points_last_5"] = feats.index.map(
        composite_lookup["form_score"].to_dict()
    ).astype(float)

    if "rolling_form_points_last_5" not in feats.columns:
        feats["rolling_form_points_last_5"] = 0.0
    feats["rolling_goal_diff_last_5"] = 0.0

    feats["attack_strength"] = _attack_strength(matches)
    feats["defense_strength"] = _defense_strength(matches)
    feats["clean_sheet_rate"] = _clean_sheet_rate(matches)
    feats["concede_rate"] = _concede_rate(matches)
    feats["draw_rate"] = _draw_rate(matches)
    feats["first_goal_rate"] = 0.5
    feats["comeback_rate"] = 0.1
    feats["penalty_shootout_history"] = 0.0
    feats["tournament_experience"] = _tournament_experience(matches)
    feats["world_cup_experience"] = _world_cup_experience(matches)
    feats["host_flag"] = feats.index.isin(WORLD_CUP_2026_HOSTS).astype(int)
    feats["same_confederation_flag"] = 0
    feats["confederation_strength_index"] = feats.index.map(
        lambda t: CONFEDERATION_STRENGTH_PRIOR.get(confederation_map.get(t, ""), 0.75)
    )

    feats = feats.fillna({
        "fifa_rank_points": feats["fifa_rank_points"].median()
        if feats["fifa_rank_points"].notna().any()
        else 800.0,
        "fifa_rank_position": feats["fifa_rank_position"].median()
        if feats["fifa_rank_position"].notna().any()
        else 100.0,
        "elo_pre_match": 1500.0,
        "pi_rating_pre_match": 0.0,
    }).fillna(0.0)

    return feats.reset_index()


def _team_aggregates(matches: pd.DataFrame) -> pd.DataFrame:
    """Aggregate goals scored and conceded per team."""
    if matches.empty:
        return pd.DataFrame(columns=["team", "matches", "gf", "ga", "wins", "draws", "losses"])
    a = matches.rename(columns={"team_a": "team", "score_a": "gf", "score_b": "ga"})[
        ["team", "gf", "ga"]
    ]
    b = matches.rename(columns={"team_b": "team", "score_b": "gf", "score_a": "ga"})[
        ["team", "gf", "ga"]
    ]
    stacked = pd.concat([a, b], ignore_index=True).dropna(subset=["team"])
    stacked["gf"] = pd.to_numeric(stacked["gf"], errors="coerce")
    stacked["ga"] = pd.to_numeric(stacked["ga"], errors="coerce")
    stacked = stacked.dropna()
    grouped = stacked.groupby("team").agg(
        matches=("gf", "size"),
        gf=("gf", "sum"),
        ga=("ga", "sum"),
    )
    grouped["wins"] = stacked.assign(win=(stacked["gf"] > stacked["ga"]).astype(int)).groupby(
        "team"
    )["win"].sum()
    grouped["draws"] = stacked.assign(draw=(stacked["gf"] == stacked["ga"]).astype(int)).groupby(
        "team"
    )["draw"].sum()
    grouped["losses"] = grouped["matches"] - grouped["wins"] - grouped["draws"]
    return grouped.reset_index()


def _attack_strength(matches: pd.DataFrame) -> pd.Series:
    agg = _team_aggregates(matches)
    if agg.empty:
        return pd.Series(dtype=float)
    avg = (agg["gf"] / agg["matches"]).rename("attack_strength")
    avg.index = agg["team"]
    return avg


def _defense_strength(matches: pd.DataFrame) -> pd.Series:
    agg = _team_aggregates(matches)
    if agg.empty:
        return pd.Series(dtype=float)
    avg_against = agg["ga"] / agg["matches"]
    score = 1.0 / (1.0 + avg_against)
    score.index = agg["team"]
    score.name = "defense_strength"
    return score


def _clean_sheet_rate(matches: pd.DataFrame) -> pd.Series:
    if matches.empty:
        return pd.Series(dtype=float)
    a = matches[["team_a", "score_b"]].rename(columns={"team_a": "team", "score_b": "ga"})
    b = matches[["team_b", "score_a"]].rename(columns={"team_b": "team", "score_a": "ga"})
    stacked = pd.concat([a, b], ignore_index=True).dropna()
    stacked["ga"] = pd.to_numeric(stacked["ga"], errors="coerce")
    stacked = stacked.dropna()
    rate = stacked.groupby("team")["ga"].apply(lambda s: float((s == 0).mean()))
    rate.name = "clean_sheet_rate"
    return rate


def _concede_rate(matches: pd.DataFrame) -> pd.Series:
    rate = _clean_sheet_rate(matches)
    if rate.empty:
        return rate
    return (1.0 - rate).rename("concede_rate")


def _draw_rate(matches: pd.DataFrame) -> pd.Series:
    if matches.empty:
        return pd.Series(dtype=float)
    a = matches[["team_a", "score_a", "score_b"]].rename(columns={"team_a": "team"})
    b = matches[["team_b", "score_a", "score_b"]].rename(columns={"team_b": "team"})
    stacked = pd.concat([a, b], ignore_index=True).dropna()
    stacked["draw"] = (stacked["score_a"] == stacked["score_b"]).astype(int)
    rate = stacked.groupby("team")["draw"].mean()
    rate.name = "draw_rate"
    return rate


def _tournament_experience(matches: pd.DataFrame) -> pd.Series:
    if matches.empty:
        return pd.Series(dtype=float)
    tournaments = {"WC", "EURO", "COPA", "AFCON", "ASIAN", "GOLD"}
    mask = matches["competition"].astype(str).str.upper().isin(tournaments)
    sub = matches[mask]
    teams = pd.concat([sub["team_a"], sub["team_b"]]).value_counts()
    teams.name = "tournament_experience"
    return teams


def _world_cup_experience(matches: pd.DataFrame) -> pd.Series:
    if matches.empty:
        return pd.Series(dtype=float)
    sub = matches[matches["competition"].astype(str).str.upper() == "WC"]
    teams = pd.concat([sub["team_a"], sub["team_b"]]).value_counts()
    teams.name = "world_cup_experience"
    return teams
