"""Train every outcome and scoreline model and persist artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.data_splitter import temporal_split
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

    trainer = Trainer(
        hyperparameters={
            "multinomial": config.model_params.get("models", {}).get("multinomial", {}).get("hyperparameters", {}),
            "xgboost": config.model_params.get("models", {}).get("xgboost", {}).get("hyperparameters", {}),
            "poisson": config.model_params.get("models", {}).get("poisson", {}).get("hyperparameters", {}),
        }
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
