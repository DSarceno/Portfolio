"""Fit outcome and scoreline models on the canonical feature matrix."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.features.build_features import select_feature_columns
from src.models.calibration import CalibrationStrategy, ProbabilityCalibrator
from src.models.model_factory import build_model
from src.models.multinomial_model import MultinomialOutcomeModel
from src.models.poisson_model import PoissonScoreModel
from src.models.xgboost_model import XGBoostOutcomeModel
from src.utils.io import ensure_dir, save_pickle
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class TrainerOutputs:
    """Container with every fitted artifact produced by :class:`Trainer`."""

    multinomial: MultinomialOutcomeModel
    xgboost: XGBoostOutcomeModel
    poisson: PoissonScoreModel
    calibrator: Optional[ProbabilityCalibrator]
    feature_columns: list[str]


class Trainer:
    """High-level trainer that fits the full model stack."""

    def __init__(
        self,
        models_dir: str | Path = "models",
        hyperparameters: Optional[dict[str, Any]] = None,
        calibration_strategy: CalibrationStrategy = CalibrationStrategy.ISOTONIC,
    ) -> None:
        """Initialize the trainer.

        Args:
            models_dir: Directory used to persist artifacts.
            hyperparameters: Mapping ``model_name -> hyperparameter dict``.
            calibration_strategy: Calibration strategy.
        """
        self.models_dir = ensure_dir(models_dir)
        self.hyperparameters = hyperparameters or {}
        self.calibration_strategy = calibration_strategy

    def fit(
        self,
        train_features: pd.DataFrame,
        val_features: Optional[pd.DataFrame] = None,
        feature_columns: Optional[list[str]] = None,
        label_column: str = "outcome",
    ) -> TrainerOutputs:
        """Fit the entire model stack.

        Args:
            train_features: Training feature matrix.
            val_features: Optional validation feature matrix used for calibration.
            feature_columns: Optional explicit feature column list.
            label_column: Outcome label column.

        Returns:
            :class:`TrainerOutputs`.

        Raises:
            ValueError: If *train_features* is empty.
        """
        if train_features.empty:
            raise ValueError("Training feature matrix is empty")

        cols = feature_columns or select_feature_columns(train_features)
        X_train = train_features[cols].copy()
        y_train = train_features[label_column].astype(str)

        mn = build_model("multinomial", feature_columns=cols, hyperparameters=self.hyperparameters.get("multinomial"))
        xgb = build_model("xgboost", feature_columns=cols, hyperparameters=self.hyperparameters.get("xgboost"))
        poisson = build_model("poisson", hyperparameters=self.hyperparameters.get("poisson"))

        mn.fit(X_train, y_train)
        xgb.fit(X_train, y_train)

        poisson_train = train_features[["team_a", "team_b", "score_a", "score_b"]].dropna()
        poisson.fit(poisson_train, poisson_train.get("outcome", pd.Series(dtype=str)))

        calibrator: Optional[ProbabilityCalibrator] = None
        if val_features is not None and not val_features.empty:
            X_val = val_features[cols]
            y_val = val_features[label_column].astype(str)
            xgb_val_proba = xgb.predict_proba(X_val)
            calibrator = ProbabilityCalibrator(strategy=self.calibration_strategy)
            calibrator.fit(xgb_val_proba, y_val)
            logger.info(
                "Calibrator fitted with %s strategy on %d validation samples",
                self.calibration_strategy.value,
                len(val_features),
            )

        save_pickle(mn, self.models_dir / "multinomial_model.pkl")
        save_pickle(xgb, self.models_dir / "xgboost_model.pkl")
        save_pickle(poisson, self.models_dir / "poisson_model.pkl")
        if calibrator is not None:
            save_pickle(calibrator, self.models_dir / "calibrator.pkl")
        save_pickle(cols, self.models_dir / "feature_columns.pkl")

        return TrainerOutputs(
            multinomial=mn,
            xgboost=xgb,
            poisson=poisson,
            calibrator=calibrator,
            feature_columns=cols,
        )
