"""Train every outcome and scoreline model and persist artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.data_splitter import temporal_split
from src.ratings.shrinkage import RatingShrinker
from src.training.trainer import Trainer
from src.utils.config import load_config
from src.utils.io import load_csv, save_csv
from src.utils.logging_config import get_logger, setup_logging


def main() -> int:
    """Train models from ``data/processed/feature_matrix.csv``."""
    setup_logging(log_file="logs/training/train_models.log")
    logger = get_logger(__name__)
    config = load_config()

    feature_path = "data/processed/feature_matrix.csv"
    if not Path(feature_path).exists():
        logger.error("Feature matrix %s not found; run scripts/run_pipeline.py first", feature_path)
        return 1
    df = load_csv(feature_path)
    df = df.dropna(subset=["outcome"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    split = temporal_split(
        df, validation_year=config.get("training.validation_year", 2025)
    )

    shrinker = None
    if bool(config.get("ratings.shrinkage.enabled", True)):
        shrinker = RatingShrinker(
            k_elo=float(config.get("ratings.shrinkage.k_elo", 30.0)),
            k_pi=float(config.get("ratings.shrinkage.k_pi", 30.0)),
            k_poisson=float(config.get("ratings.shrinkage.k_poisson", 20.0)),
            mixture_prior=bool(config.get("ratings.shrinkage.mixture_prior.enabled", True)),
            elite_fraction=float(config.get("ratings.shrinkage.mixture_prior.elite_fraction", 0.30)),
            min_matches_for_elite=int(config.get("ratings.shrinkage.mixture_prior.min_matches", 20)),
        )
        logger.info(
            "Poisson shrinkage enabled (K=%s, mixture=%s)",
            shrinker.k_poisson,
            shrinker.mixture_prior,
        )

    lasso_enabled = bool(config.get("training.lasso.enabled", True))
    lasso_C = float(config.get("training.lasso.C", 0.1))
    if lasso_enabled:
        logger.info("LASSO feature selection enabled (C=%.3f)", lasso_C)

    trainer = Trainer(
        hyperparameters={
            "multinomial": config.model_params.get("models", {}).get("multinomial", {}).get("hyperparameters", {}),
            "xgboost": config.model_params.get("models", {}).get("xgboost", {}).get("hyperparameters", {}),
            "poisson": config.model_params.get("models", {}).get("poisson", {}).get("hyperparameters", {}),
        },
        shrinker=shrinker,
        lasso_select=lasso_enabled,
        lasso_C=lasso_C,
    )
    outputs = trainer.fit(train_features=split.train, val_features=split.validation)
    logger.info("Training complete on %d rows", len(split.train))
    save_csv(
        outputs.xgboost.feature_importance(),
        "outputs/diagnostics/feature_importance.csv",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
