"""Helper that rebuilds the feature matrix on the fly for prediction scripts."""

from __future__ import annotations

from typing import Optional

import pandas as pd

from src.data.data_loader import DataLoader
from src.data.fifa_rankings_client import FifaRankingsClient
from src.data.tournament_updater import TournamentUpdater
from src.features.build_features import build_match_feature_matrix
from src.ratings.rating_ensemble import RatingEnsemble
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def build_inference_feature_matrix(
    tournament_year: int = 2026,
    matches: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build the full feature matrix used for inference.

    The matrix contains every match in the canonical table (historical +
    upcoming). Upcoming matches have ``outcome=NaN``.

    Args:
        tournament_year: Tournament year used to compute the state features.
        matches: Optional pre-loaded canonical match table.

    Returns:
        DataFrame with the engineered feature columns and metadata. Empty if
        no matches are available.
    """
    if matches is None:
        matches = DataLoader().load_matches()
    if matches.empty:
        logger.warning("Canonical match table is empty; cannot build features")
        return pd.DataFrame()

    ensemble = RatingEnsemble().fit(matches)
    composite = ensemble.composite_table()
    rankings = FifaRankingsClient().latest()
    state = TournamentUpdater().world_cup_state(tournament_year=tournament_year)

    feature_matrix = build_match_feature_matrix(
        matches=matches,
        rankings=rankings,
        composite_table=composite,
        tournament_state=state,
    )
    logger.info("Built inference feature matrix: %d rows", len(feature_matrix))
    return feature_matrix


def filter_upcoming(
    feature_matrix: pd.DataFrame,
    stage_substr: Optional[str] = None,
) -> pd.DataFrame:
    """Filter the feature matrix to upcoming (unplayed) matches.

    Args:
        feature_matrix: Output of :func:`build_inference_feature_matrix`.
        stage_substr: Optional substring to require in the ``stage`` column
            (case-insensitive). E.g. ``"group"`` for group-stage fixtures.

    Returns:
        DataFrame slice containing only unplayed matches matching the filter.
    """
    if feature_matrix.empty:
        return feature_matrix

    upcoming_mask = feature_matrix["outcome"].isna()
    sub = feature_matrix[upcoming_mask]

    if stage_substr is not None:
        stage_mask = (
            sub["stage"].astype(str).str.lower().str.contains(stage_substr.lower(), na=False)
        )
        sub_stage = sub[stage_mask]
        if not sub_stage.empty:
            return sub_stage.copy()
        logger.warning(
            "No upcoming matches matched stage filter '%s'; returning all upcoming",
            stage_substr,
        )
    return sub.copy()
