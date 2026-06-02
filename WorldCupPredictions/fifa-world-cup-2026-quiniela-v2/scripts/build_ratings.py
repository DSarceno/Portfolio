"""Build pre-tournament team ratings (Elo + PI + form) and persist them."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_loader import DataLoader
from src.ratings.rating_ensemble import RatingEnsemble
from src.utils.io import save_csv, save_pickle
from src.utils.logging_config import get_logger, setup_logging


def main() -> int:
    """Run the rating build."""
    setup_logging(log_file="logs/pipeline/build_ratings.log")
    logger = get_logger(__name__)

    loader = DataLoader()
    matches = loader.load_matches()
    if matches.empty:
        logger.warning("Canonical match table is empty; ratings will be uniform")

    ensemble = RatingEnsemble().fit(matches)
    composite = ensemble.composite_table()
    save_csv(composite, "outputs/diagnostics/composite_ratings.csv")
    save_pickle(ensemble, "models/rating_ensemble.pkl")
    logger.info("Wrote composite_ratings.csv (%d teams)", len(composite))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
