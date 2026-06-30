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
        near_constant_frac: float = 0.99,
        max_abs_scaled: float = 8.0,
    ) -> None:
        """Initialize the model.

        Args:
            feature_columns: Optional explicit list of feature columns.
            hyperparameters: Optional dict overriding scikit-learn defaults.
            near_constant_frac: A feature whose single most-frequent training
                value covers at least this fraction of rows is dropped before
                fitting. Such a feature carries no learnable variance, and —
                because this model standardizes — its tiny ``scale_`` turns any
                differing value at inference (the classic train/serve skew of a
                tournament-stage indicator that is constant in historical data
                but 1 for every live knockout fixture) into a ~100-SD lever that
                dominates the linear logit and saturates the softmax. See
                ``docs/AUDIT_2026-06-29.md``.
            max_abs_scaled: Standardized features are clipped to ±this many SDs
                before the logit, as defense-in-depth so no residual out-of-
                distribution value can dominate the prediction.
        """
        super().__init__(name="multinomial", feature_columns=feature_columns)
        hp = hyperparameters or {}
        self.near_constant_frac = float(near_constant_frac)
        self.max_abs_scaled = float(max_abs_scaled)
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

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore from a pickle, backfilling fields added after it was saved.

        Models pickled before the near-constant guard / clip were introduced have
        neither attribute; without this the live prediction path (which calls
        ``predict_proba`` on the model regardless of its blend weight) would raise
        ``AttributeError``. Defaults also retro-apply the ±SD clip at predict time,
        which tames — without retraining — an older degenerate artifact.
        """
        self.__dict__.update(state)
        self.__dict__.setdefault("near_constant_frac", 0.99)
        self.__dict__.setdefault("max_abs_scaled", 8.0)

    def _to_index(self, y: pd.Series) -> np.ndarray:
        return np.asarray([OUTCOME_TO_INDEX[str(v)] for v in y], dtype=int)

    def _drop_near_constant(self, X: pd.DataFrame, cols: list[str]) -> list[str]:
        """Return *cols* minus features that are (near-)constant in training.

        A column is near-constant when its most-frequent value covers at least
        ``near_constant_frac`` of the rows (this also catches exact constants).
        At least one column is always retained so the model can still fit.
        """
        n = len(X)
        if n == 0:
            return cols
        kept: list[str] = []
        dropped: list[str] = []
        for c in cols:
            top_frac = X[c].value_counts(dropna=False).iloc[0] / n if n else 1.0
            (kept if top_frac < self.near_constant_frac else dropped).append(c)
        if not kept:  # degenerate edge case: keep everything rather than nothing
            return cols
        if dropped:
            logger.info(
                "Multinomial dropped %d near-constant feature(s) (>=%.0f%% one value): %s",
                len(dropped),
                100 * self.near_constant_frac,
                dropped,
            )
        return kept

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MultinomialOutcomeModel":
        """Fit the model.

        Args:
            X: Feature DataFrame.
            y: H/D/A outcome labels.

        Returns:
            ``self``.
        """
        cols = self._drop_near_constant(X, self.feature_columns or list(X.columns))
        self.feature_columns = cols
        X_scaled = self.scaler.fit_transform(X[cols].values)
        X_scaled = np.clip(X_scaled, -self.max_abs_scaled, self.max_abs_scaled)
        y_idx = self._to_index(y)
        self.classifier.fit(X_scaled, y_idx)
        self.is_fitted = True
        logger.info("Multinomial model fitted on %d samples / %d features", len(X), len(cols))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict per-class probabilities."""
        self._check_fitted()
        X_scaled = self.scaler.transform(X[self.feature_columns].values)
        X_scaled = np.clip(X_scaled, -self.max_abs_scaled, self.max_abs_scaled)
        proba = self.classifier.predict_proba(X_scaled)
        proba = np.asarray(proba, dtype=float)
        proba = proba / np.clip(proba.sum(axis=1, keepdims=True), 1e-9, None)
        return proba
