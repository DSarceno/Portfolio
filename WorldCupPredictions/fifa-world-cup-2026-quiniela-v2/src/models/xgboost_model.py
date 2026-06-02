"""XGBoost multi-class outcome model."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
import xgboost as xgb

from src.models.base_model import BaseOutcomeModel
from src.utils.constants import OUTCOME_TO_INDEX
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class XGBoostOutcomeModel(BaseOutcomeModel):
    """XGBoost classifier with sane defaults for H/D/A prediction."""

    def __init__(
        self,
        feature_columns: Optional[list[str]] = None,
        hyperparameters: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize the model.

        Args:
            feature_columns: Optional explicit list of feature columns.
            hyperparameters: Optional dict overriding default XGBoost params.
        """
        super().__init__(name="xgboost", feature_columns=feature_columns)
        hp = hyperparameters or {}
        self.params = {
            "objective": hp.get("objective", "multi:softprob"),
            "num_class": hp.get("num_class", 3),
            "max_depth": hp.get("max_depth", 5),
            "learning_rate": hp.get("learning_rate", 0.05),
            "n_estimators": hp.get("n_estimators", 400),
            "subsample": hp.get("subsample", 0.85),
            "colsample_bytree": hp.get("colsample_bytree", 0.85),
            "reg_alpha": hp.get("reg_alpha", 0.0),
            "reg_lambda": hp.get("reg_lambda", 1.0),
            "random_state": hp.get("random_state", 42),
            "n_jobs": hp.get("n_jobs", -1),
            "tree_method": hp.get("tree_method", "hist"),
        }
        self.classifier: Optional[xgb.XGBClassifier] = None

    def _to_index(self, y: pd.Series) -> np.ndarray:
        return np.asarray([OUTCOME_TO_INDEX[str(v)] for v in y], dtype=int)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBoostOutcomeModel":
        """Fit the model.

        Args:
            X: Feature DataFrame.
            y: H/D/A outcome labels.

        Returns:
            ``self``.
        """
        cols = self.feature_columns or list(X.columns)
        self.feature_columns = cols
        y_idx = self._to_index(y)
        self.classifier = xgb.XGBClassifier(**self.params, use_label_encoder=False, eval_metric="mlogloss")
        self.classifier.fit(X[cols].values, y_idx)
        self.is_fitted = True
        logger.info("XGBoost model fitted on %d samples / %d features", len(X), len(cols))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict per-class probabilities."""
        self._check_fitted()
        proba = self.classifier.predict_proba(X[self.feature_columns].values)
        proba = np.asarray(proba, dtype=float)
        proba = proba / np.clip(proba.sum(axis=1, keepdims=True), 1e-9, None)
        return proba

    def feature_importance(self) -> pd.DataFrame:
        """Return a sorted DataFrame of feature importances."""
        self._check_fitted()
        importances = self.classifier.feature_importances_
        return (
            pd.DataFrame(
                {"feature": self.feature_columns, "importance": importances}
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
