"""Run the feature pipeline on the existing canonical match table.

This script is **idempotent**: it does not re-collect data from any external
source, so running it never overwrites ``data/interim/matches_unified.csv``.
Use ``scripts/bootstrap_historical_data.py`` (one-off) or
``scripts/update_after_matchday.py`` (daily) to refresh the canonical table.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_loader import DataLoader
from src.data.data_validator import validate_match_dataframe
from src.data.fifa_rankings_client import FifaRankingsClient
from src.data.tournament_updater import TournamentUpdater
from src.features.build_features import build_match_feature_matrix
from src.ratings.rating_ensemble import RatingEnsemble
from src.utils.config import load_config
from src.utils.io import save_csv
from src.utils.logging_config import get_logger, setup_logging


def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(description="Build the feature matrix")
    parser.add_argument(
        "--collect",
        action="store_true",
        help="Also run ResultsCollector before building features (slow, may overwrite).",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=None,
        help="Restrict the feature matrix to matches from this year onward.",
    )
    return parser.parse_args()


def main() -> int:
    """Build the feature matrix."""
    setup_logging(log_file="logs/pipeline/run_pipeline.log")
    logger = get_logger(__name__)
    config = load_config()
    args = parse_args()

    if args.collect:
        from src.data.results_collector import ResultsCollector

        logger.info("--collect flag set: refreshing the canonical table first")
        ResultsCollector().collect_all(
            competitions=["WC", "EC", "CA"],
            start_year=args.start_year or config.get("data.historical_start_year", 2014),
            end_year=config.get("data.historical_end_year", 2026),
            merge_with_existing=True,
        )

    loader = DataLoader()
    matches = loader.load_matches()

    if matches.empty:
        logger.error(
            "Canonical match table is empty. Run scripts/bootstrap_historical_data.py "
            "or scripts/import_kaggle_history.py first."
        )
        return 1

    if args.start_year is not None:
        import pandas as pd

        matches = matches[pd.to_datetime(matches["date"], errors="coerce").dt.year >= args.start_year]
        logger.info("Filtered to matches from %d onwards: %d rows", args.start_year, len(matches))

    report = validate_match_dataframe(matches)
    if not report.is_valid:
        logger.warning("Validation issues found: %s", report)

    ensemble = RatingEnsemble().fit(matches)
    composite = ensemble.composite_table()

    rankings = FifaRankingsClient().latest()
    state = TournamentUpdater().world_cup_state(
        tournament_year=config.get("tournament.year", 2026)
    )
    feature_matrix = build_match_feature_matrix(
        matches=matches,
        rankings=rankings,
        composite_table=composite,
        tournament_state=state,
    )

    if feature_matrix.empty:
        logger.error(
            "Feature matrix is empty after the build step. "
            "Inspect data/interim/matches_unified.csv for data issues."
        )
        return 1

    save_csv(feature_matrix, "data/processed/feature_matrix.csv")
    logger.info(
        "Pipeline finished: %d feature rows, %d columns",
        len(feature_matrix),
        len(feature_matrix.columns),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
