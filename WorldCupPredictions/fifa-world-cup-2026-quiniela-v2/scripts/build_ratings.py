"""Build pre-tournament team ratings (Elo + PI + form) with optional shrinkage."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_loader import DataLoader
from src.ratings.elo import EloConfig, EloRating
from src.ratings.pi_rating import PIConfig, PIRating
from src.ratings.rating_ensemble import RatingEnsemble
from src.ratings.shrinkage import RatingShrinker
from src.utils.config import load_config
from src.utils.io import save_csv, save_pickle
from src.utils.logging_config import get_logger, setup_logging


def main() -> int:
    """Run the rating build."""
    setup_logging(log_file="logs/pipeline/build_ratings.log")
    logger = get_logger(__name__)
    config = load_config()

    loader = DataLoader()
    matches = loader.load_matches()
    if matches.empty:
        logger.warning("Canonical match table is empty; ratings will be uniform")

    shrinker = None
    if bool(config.get("ratings.shrinkage.enabled", True)):
        shrinker = RatingShrinker(
            k_elo=float(config.get("ratings.shrinkage.k_elo", 30.0)),
            k_pi=float(config.get("ratings.shrinkage.k_pi", 30.0)),
            k_poisson=float(config.get("ratings.shrinkage.k_poisson", 20.0)),
        )
        logger.info(
            "Shrinkage enabled (K_elo=%s, K_pi=%s, K_poisson=%s)",
            shrinker.k_elo,
            shrinker.k_pi,
            shrinker.k_poisson,
        )
    else:
        logger.info("Shrinkage disabled in config")

    decay_enabled = bool(config.get("ratings.time_decay.enabled", False))
    xi_elo = float(config.get("ratings.time_decay.xi_elo", 0.0)) if decay_enabled else 0.0
    xi_pi = float(config.get("ratings.time_decay.xi_pi", 0.0)) if decay_enabled else 0.0
    if decay_enabled:
        logger.info("Time decay enabled (xi_elo=%.4f, xi_pi=%.4f)", xi_elo, xi_pi)

    elo = EloRating(config=EloConfig(time_decay_xi=xi_elo))
    pi = PIRating(config=PIConfig(time_decay_xi=xi_pi))

    ensemble = RatingEnsemble(elo=elo, pi=pi, shrinker=shrinker).fit(matches)
    composite = ensemble.composite_table()
    save_csv(composite, "outputs/diagnostics/composite_ratings.csv")
    save_pickle(ensemble, "models/rating_ensemble.pkl")
    logger.info("Wrote composite_ratings.csv (%d teams)", len(composite))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
