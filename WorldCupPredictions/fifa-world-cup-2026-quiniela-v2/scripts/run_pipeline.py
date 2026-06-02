"""Run the full ETL + feature pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_loader import DataLoader
from src.data.data_validator import validate_match_dataframe
from src.data.results_collector import ResultsCollector
from src.data.tournament_updater import TournamentUpdater
from src.features.build_features import build_match_feature_matrix
from src.ratings.rating_ensemble import RatingEnsemble
from src.data.fifa_rankings_client import FifaRankingsClient
from src.utils.io import save_csv
from src.utils.logging_config import get_logger, setup_logging


def main() -> int:
    """Run the pipeline end-to-end."""
    setup_logging(log_file="logs/pipeline/run_pipeline.log")
    logger = get_logger(__name__)

    collector = ResultsCollector()
    matches = collector.collect_all(["WC", "EC", "CA"])
    logger.info("Collected %d matches", len(matches))

    report = validate_match_dataframe(matches)
    if not report.is_valid:
        logger.warning("Validation issues found: %s", report)

    loader = DataLoader()
    matches = loader.load_matches()
    ensemble = RatingEnsemble().fit(matches)
    composite = ensemble.composite_table()

    rankings = FifaRankingsClient().latest()
    state = TournamentUpdater().world_cup_state(tournament_year=2026)
    feature_matrix = build_match_feature_matrix(
        matches=matches,
        rankings=rankings,
        composite_table=composite,
        tournament_state=state,
    )
    save_csv(feature_matrix, "data/processed/feature_matrix.csv")
    logger.info("Pipeline finished: %d feature rows", len(feature_matrix))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
