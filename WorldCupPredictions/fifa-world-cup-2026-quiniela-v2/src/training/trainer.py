"""Fit outcome and scoreline models on the canonical feature matrix."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.features.build_features import select_feature_columns
from src.models.calibration import CalibrationStrategy, ProbabilityCalibrator
from src.models.model_factory import build_model
from src.models.multinomial_model import MultinomialOutcomeModel
from src.models.poisson_model import PoissonScoreModel
from src.models.xgboost_model import XGBoostOutcomeModel
from src.ratings.shrinkage import RatingShrinker, count_matches_per_team
from src.utils.io import ensure_dir, save_pickle
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def _lasso_select_features(
    X: pd.DataFrame,
    y: pd.Series,
    candidate_columns: list[str],
    C: float = 0.1,
    min_features: int = 5,
) -> list[str]:
    """Run an L1-penalized multinomial logistic on candidate features.

    Following Groll, Schauberger & Tutz (2015), we use LASSO to prune
    engineered features that don't carry independent predictive signal.
    Features whose coefficient is zero across all classes are dropped.

    Args:
        X: Training feature matrix (rows aligned with *y*).
        y: Outcome labels (``"H"``, ``"D"``, ``"A"``).
        candidate_columns: Features to consider for selection.
        C: Inverse regularization strength. Lower ``C`` = more pruning.
        min_features: Floor on the number of retained features. If LASSO
            picks fewer than this, we keep the top-``min_features`` by
            absolute coefficient magnitude across classes.

    Returns:
        Sorted list of retained feature names. Falls back to *candidate_columns*
        unchanged if anything goes wrong.
    """
    cols = [c for c in candidate_columns if c in X.columns]
    if len(cols) <= min_features:
        return cols
    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X[cols].fillna(0.0))
        lasso = LogisticRegression(
            penalty="l1",
            solver="saga",
            C=C,
            max_iter=2000,
            multi_class="multinomial",
            n_jobs=-1,
        )
        lasso.fit(X_scaled, y)
        coef_mag = np.abs(lasso.coef_).max(axis=0)
        nonzero_mask = coef_mag > 1e-8
        kept = [c for c, keep in zip(cols, nonzero_mask) if keep]
        if len(kept) < min_features:
            order = np.argsort(-coef_mag)
            kept = [cols[i] for i in order[:min_features]]
        dropped = sorted(set(cols) - set(kept))
        logger.info(
            "LASSO kept %d/%d features (C=%.2f). Dropped: %s",
            len(kept),
            len(cols),
            C,
            dropped,
        )
        return sorted(kept)
    except (ValueError, RuntimeError) as exc:
        logger.warning("LASSO selection failed (%s); falling back to full set", exc)
        return cols


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
        shrinker: Optional[RatingShrinker] = None,
        lasso_select: bool = True,
        lasso_C: float = 0.1,
    ) -> None:
        """Initialize the trainer.

        Args:
            models_dir: Directory used to persist artifacts.
            hyperparameters: Mapping ``model_name -> hyperparameter dict``.
            calibration_strategy: Calibration strategy.
            shrinker: Optional :class:`RatingShrinker`. When provided, the
                Poisson attack/defense parameters are pulled toward
                confederation priors after fit, reducing inflation for teams
                with weak strength-of-schedule.
            lasso_select: If ``True`` (default), run an L1-penalized
                multinomial logistic to prune engineered features that don't
                carry independent predictive signal (Groll et al. 2015).
            lasso_C: Inverse regularization strength for the LASSO selection
                step. Lower values prune more aggressively.
        """
        self.models_dir = ensure_dir(models_dir)
        self.hyperparameters = hyperparameters or {}
        self.calibration_strategy = calibration_strategy
        self.shrinker = shrinker
        self.lasso_select = lasso_select
        self.lasso_C = float(lasso_C)

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

        candidate_cols = feature_columns or select_feature_columns(train_features)
        y_train = train_features[label_column].astype(str)
        if self.lasso_select and feature_columns is None:
            cols = _lasso_select_features(
                train_features, y_train, candidate_cols, C=self.lasso_C
            )
        else:
            cols = candidate_cols
        X_train = train_features[cols].copy()

        mn = build_model("multinomial", feature_columns=cols, hyperparameters=self.hyperparameters.get("multinomial"))
        xgb = build_model("xgboost", feature_columns=cols, hyperparameters=self.hyperparameters.get("xgboost"))
        poisson = build_model("poisson", hyperparameters=self.hyperparameters.get("poisson"))

        mn.fit(X_train, y_train)
        xgb.fit(X_train, y_train)

        poisson_cols = ["team_a", "team_b", "score_a", "score_b"]
        if "date" in train_features.columns:
            poisson_cols = poisson_cols + ["date"]
        poisson_train = train_features[poisson_cols].dropna(
            subset=["team_a", "team_b", "score_a", "score_b"]
        )
        poisson.fit(poisson_train, poisson_train.get("outcome", pd.Series(dtype=str)))

        if self.shrinker is not None:
            match_counts = count_matches_per_team(poisson_train)
            self.shrinker.shrink_poisson(poisson, match_counts)

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
