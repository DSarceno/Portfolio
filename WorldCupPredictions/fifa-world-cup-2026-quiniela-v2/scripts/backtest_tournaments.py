"""Cross-tournament backtest of the full prediction stack (A.8).

Trains on every match before a target tournament year and evaluates on that
tournament, with no leakage. Quantifies whether modelling changes (mixture
prior, blend weights, K factors) improve held-out log-loss / Brier / RPS.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_loader import DataLoader
from src.ensemble.blender import BlendWeights
from src.ratings.shrinkage import RatingShrinker
from src.training.backtester import run_tournament_backtest
from src.utils.config import load_config
from src.utils.io import save_csv
from src.utils.logging_config import get_logger, setup_logging


def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(description="Cross-tournament backtest")
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=[2018, 2022],
        help="Tournament years to hold out and predict (default: 2018 2022).",
    )
    parser.add_argument(
        "--competition",
        type=str,
        default="WC",
        help="Competition code identifying the target tournament (default: WC).",
    )
    return parser.parse_args()


def _build_shrinker(config) -> RatingShrinker | None:
    """Construct a shrinker from config, or ``None`` when disabled."""
    if not bool(config.get("ratings.shrinkage.enabled", True)):
        return None
    return RatingShrinker(
        k_elo=float(config.get("ratings.shrinkage.k_elo", 30.0)),
        k_pi=float(config.get("ratings.shrinkage.k_pi", 30.0)),
        k_poisson=float(config.get("ratings.shrinkage.k_poisson", 20.0)),
        mixture_prior=bool(config.get("ratings.shrinkage.mixture_prior.enabled", True)),
        elite_fraction=float(config.get("ratings.shrinkage.mixture_prior.elite_fraction", 0.30)),
        min_matches_for_elite=int(config.get("ratings.shrinkage.mixture_prior.min_matches", 20)),
    )


def main() -> int:
    """Run the cross-tournament backtest and persist the metrics."""
    setup_logging(log_file="logs/training/backtest_tournaments.log")
    logger = get_logger(__name__)
    config = load_config()
    args = parse_args()

    matches = DataLoader().load_matches()
    if matches.empty:
        logger.error("Canonical match table is empty; cannot backtest")
        return 1

    blend_weights = BlendWeights.from_config(config)
    lasso_enabled = bool(config.get("training.lasso.enabled", True))
    lasso_C = float(config.get("training.lasso.C", 0.1))

    result = run_tournament_backtest(
        matches=matches,
        target_years=args.years,
        competition=args.competition,
        shrinker_factory=lambda: _build_shrinker(config),
        blend_weights=blend_weights,
        hyperparameters=config.model_params.get("models"),
        lasso_select=lasso_enabled,
        lasso_C=lasso_C,
    )

    if result.per_fold.empty:
        logger.error("Backtest produced no scored folds; check year coverage")
        return 1

    suffix = "_".join(str(y) for y in sorted(args.years))
    out_path = f"outputs/diagnostics/backtest_{args.competition}_{suffix}.csv"
    save_csv(result.per_fold, out_path)
    logger.info("Wrote %s", out_path)
    logger.info("Aggregate metrics: %s", result.aggregate.to_dict())
    print(result.per_fold.to_string(index=False))
    print("\nAggregate:")
    print(result.aggregate.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
