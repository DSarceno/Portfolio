"""Market / public-bias proxy features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

BLUE_BLOOD_REPUTATION = {
    "Brazil": 1.0,
    "Argentina": 0.95,
    "Germany": 0.95,
    "France": 0.95,
    "Italy": 0.90,
    "Spain": 0.90,
    "England": 0.88,
    "Netherlands": 0.86,
    "Portugal": 0.84,
    "Uruguay": 0.80,
    "Belgium": 0.78,
    "Mexico": 0.70,
    "United States": 0.65,
}


def compute_market_features(match_features: pd.DataFrame) -> pd.DataFrame:
    """Add market/public-bias proxy columns to *match_features*.

    Args:
        match_features: Per-match feature DataFrame.

    Returns:
        DataFrame with the new ``public_bias_proxy``, ``reputation_diff`` and
        ``fifa_overreaction_proxy`` columns.
    """
    if match_features.empty:
        return match_features

    df = match_features.copy()
    rep_a = df["team_a"].map(BLUE_BLOOD_REPUTATION).fillna(0.4)
    rep_b = df["team_b"].map(BLUE_BLOOD_REPUTATION).fillna(0.4)
    df["reputation_a"] = rep_a
    df["reputation_b"] = rep_b
    df["reputation_diff"] = rep_a - rep_b

    elo_diff = df.get("elo_diff", pd.Series(np.zeros(len(df)))).fillna(0)
    df["fifa_overreaction_proxy"] = np.tanh(elo_diff / 200.0) - df["reputation_diff"]
    df["public_bias_proxy"] = 0.6 * df["reputation_diff"] + 0.4 * np.tanh(elo_diff / 200.0)
    return df
