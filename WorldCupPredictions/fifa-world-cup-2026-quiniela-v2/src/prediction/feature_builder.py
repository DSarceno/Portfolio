"""Helper that rebuilds the feature matrix on the fly for prediction scripts."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src.data.data_loader import DataLoader
from src.data.fifa_rankings_client import FifaRankingsClient
from src.data.squad_value_client import SquadValueClient
from src.data.tournament_updater import TournamentUpdater
from src.features.build_features import build_match_feature_matrix
from src.ratings.rating_ensemble import RatingEnsemble
from src.utils.config import load_config
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

    squad_values = _load_squad_values()

    feature_matrix = build_match_feature_matrix(
        matches=matches,
        rankings=rankings,
        composite_table=composite,
        tournament_state=state,
        squad_values=squad_values,
    )
    logger.info("Built inference feature matrix: %d rows", len(feature_matrix))
    return feature_matrix


def build_pairwise_feature_matrix(
    teams: Sequence[str],
    matches: Optional[pd.DataFrame] = None,
    tournament_year: int = 2026,
    squad_values: Optional[pd.DataFrame] = None,
    as_of_date: Optional[str] = None,
) -> pd.DataFrame:
    """Build a full feature row for every ordered pair of *teams*.

    The Monte-Carlo simulator predicts hypothetical knockout pairings that do
    not exist as scheduled fixtures, so historically it fell back to ratings +
    Poisson only — ignoring squad value and the other engineered features. This
    builds synthetic neutral-venue fixtures for all ``n*(n-1)`` ordered pairs and
    runs them through the **same** feature pipeline (ratings fit on real history,
    squad value as-of join, neutral match context), so the simulator can score
    them with the full blended model exactly like a real match.

    Args:
        teams: Teams competing in the tournament.
        matches: Canonical match table (loaded if ``None``). Ratings and team
            aggregates come from this real history only.
        tournament_year: Year used for ratings/state and the squad-value
            snapshot.
        squad_values: Optional squad-value snapshots; loaded from config when
            ``None``.
        as_of_date: Synthetic fixture date (defaults to ``<year>-07-01``, i.e.
            the knockout window, so the current squad-value snapshot applies).

    Returns:
        Feature matrix with one row per ordered pair, carrying every column in
        :data:`~src.features.build_features.NUMERIC_FEATURE_COLUMNS`. Empty if no
        matches are available.
    """
    if matches is None:
        matches = DataLoader().load_matches()
    if matches.empty:
        logger.warning("Canonical match table is empty; cannot build pairwise features")
        return pd.DataFrame()

    team_list = sorted({str(t) for t in teams})
    as_of_date = as_of_date or f"{tournament_year}-07-01"
    synthetic = pd.DataFrame(
        [
            {
                "date": as_of_date,
                "competition": "WC",
                "stage": "SIMPAIR",
                "group": "",
                "team_a": a,
                "team_b": b,
                "score_a": np.nan,
                "score_b": np.nan,
                "neutral_venue": True,
            }
            for a in team_list
            for b in team_list
            if a != b
        ]
    )
    if synthetic.empty:
        return pd.DataFrame()

    # Ratings/aggregates from real history only; synthetic rows (no scores) do
    # not affect them but receive the resulting team-strength diffs.
    ensemble = RatingEnsemble().fit(matches)
    composite = ensemble.composite_table()
    rankings = FifaRankingsClient().latest()
    state = TournamentUpdater().world_cup_state(tournament_year=tournament_year)
    if squad_values is None:
        squad_values = _load_squad_values()

    combined = pd.concat([matches, synthetic], ignore_index=True)
    feature_matrix = build_match_feature_matrix(
        matches=combined,
        rankings=rankings,
        composite_table=composite,
        tournament_state=state,
        squad_values=squad_values,
    )
    if feature_matrix.empty:
        return feature_matrix

    feature_matrix["date"] = pd.to_datetime(feature_matrix["date"], errors="coerce")
    pairs = feature_matrix[feature_matrix["date"] == pd.Timestamp(as_of_date)].copy()
    logger.info("Built pairwise feature matrix: %d ordered pairs", len(pairs))
    return pairs.reset_index(drop=True)


def _load_squad_values() -> Optional[pd.DataFrame]:
    """Load squad-value snapshots when the source is enabled and present."""
    config = load_config()
    if not bool(config.get("data.sources.squad_value.enabled", True)):
        logger.info("Squad-value source disabled in config")
        return None
    client = SquadValueClient()
    if not client.is_available():
        logger.info("Squad-value table not found; features will skip squad value")
        return None
    return client.load()


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
