"""Multinomial logistic-regression outcome model."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.models.base_model import BaseOutcomeModel
from src.utils.constants import OUTCOME_TO_INDEX
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class MultinomialOutcomeModel(BaseOutcomeModel):
    """Standardized multinomial logistic regression on H/D/A outcomes."""

    def __init__(
        self,
        feature_columns: Optional[list[str]] = None,
        hyperparameters: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize the model.

        Args:
            feature_columns: Optional explicit list of feature columns.
            hyperparameters: Optional dict overriding scikit-learn defaults.
        """
        super().__init__(name="multinomial", feature_columns=feature_columns)
        hp = hyperparameters or {}
        self.scaler = StandardScaler()
        self.classifier = LogisticRegression(
            C=hp.get("C", 1.0),
            max_iter=hp.get("max_iter", 1000),
            class_weight=hp.get("class_weight", "balanced"),
            multi_class="multinomial",
            solver="lbfgs",
            n_jobs=hp.get("n_jobs", -1),
            random_state=hp.get("random_state", 42),
        )

    def _to_index(self, y: pd.Series) -> np.ndarray:
        return np.asarray([OUTCOME_TO_INDEX[str(v)] for v in y], dtype=int)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MultinomialOutcomeModel":
        """Fit the model.

        Args:
            X: Feature DataFrame.
            y: H/D/A outcome labels.

        Returns:
            ``self``.
        """
        cols = self.feature_columns or list(X.columns)
        self.feature_columns = cols
        X_scaled = self.scaler.fit_transform(X[cols].values)
        y_idx = self._to_index(y)
        self.classifier.fit(X_scaled, y_idx)
        self.is_fitted = True
        logger.info("Multinomial model fitted on %d samples / %d features", len(X), len(cols))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict per-class probabilities."""
        self._check_fitted()
        X_scaled = self.scaler.transform(X[self.feature_columns].values)
        proba = self.classifier.predict_proba(X_scaled)
        proba = np.asarray(proba, dtype=float)
        proba = proba / np.clip(proba.sum(axis=1, keepdims=True), 1e-9, None)
        return proba
