"""Fatigue, rest and travel proxy features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

CONTINENT_DISTANCE_PROXY = {
    ("Europe", "Europe"): 0.1,
    ("Europe", "South America"): 0.8,
    ("Europe", "North America"): 0.7,
    ("Europe", "Asia"): 0.9,
    ("Europe", "Africa"): 0.5,
    ("Europe", "Oceania"): 1.0,
    ("South America", "South America"): 0.2,
    ("South America", "North America"): 0.5,
    ("South America", "Asia"): 1.0,
    ("South America", "Africa"): 0.9,
    ("North America", "North America"): 0.1,
    ("Asia", "Asia"): 0.2,
    ("Africa", "Africa"): 0.2,
}


def compute_fatigue_features(
    match_features: pd.DataFrame, team_continent: dict[str, str] | None = None
) -> pd.DataFrame:
    """Add rest / travel proxies to a match feature table.

    Args:
        match_features: Per-match feature DataFrame, must include ``date``,
            ``team_a`` and ``team_b``.
        team_continent: Mapping ``team -> continent``.

    Returns:
        DataFrame with ``rest_days_team_a``, ``rest_days_team_b``,
        ``rest_diff``, ``travel_burden_proxy_team_a`` and
        ``travel_burden_proxy_team_b``.
    """
    if match_features.empty:
        return match_features

    df = match_features.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)

    last_played: dict[str, pd.Timestamp] = {}
    rest_a: list[float] = []
    rest_b: list[float] = []
    for _, row in df.iterrows():
        match_date = row["date"]
        ra = (match_date - last_played[row["team_a"]]).days if row["team_a"] in last_played else 10
        rb = (match_date - last_played[row["team_b"]]).days if row["team_b"] in last_played else 10
        rest_a.append(float(ra))
        rest_b.append(float(rb))
        last_played[row["team_a"]] = match_date
        last_played[row["team_b"]] = match_date
    df["rest_days_team_a"] = rest_a
    df["rest_days_team_b"] = rest_b
    df["rest_diff"] = df["rest_days_team_a"] - df["rest_days_team_b"]

    if team_continent:
        df["travel_burden_proxy_team_a"] = df.apply(
            lambda r: _travel_proxy(r.get("team_a"), r.get("host_country"), team_continent),
            axis=1,
        )
        df["travel_burden_proxy_team_b"] = df.apply(
            lambda r: _travel_proxy(r.get("team_b"), r.get("host_country"), team_continent),
            axis=1,
        )
    else:
        df["travel_burden_proxy_team_a"] = 0.3
        df["travel_burden_proxy_team_b"] = 0.3

    df["fatigue_index_a"] = np.clip(1.0 / (1.0 + df["rest_days_team_a"]) * 10, 0, 1.5)
    df["fatigue_index_b"] = np.clip(1.0 / (1.0 + df["rest_days_team_b"]) * 10, 0, 1.5)
    return df


def _travel_proxy(team: str | None, host: str | None, mapping: dict[str, str]) -> float:
    """Return a coarse travel-burden proxy for *team* playing in *host*."""
    if not team or not host or not mapping:
        return 0.3
    team_continent = mapping.get(team, "")
    host_continent = mapping.get(host, "")
    if not team_continent or not host_continent:
        return 0.3
    return float(
        CONTINENT_DISTANCE_PROXY.get(
            (team_continent, host_continent),
            CONTINENT_DISTANCE_PROXY.get((host_continent, team_continent), 0.5),
        )
    )
