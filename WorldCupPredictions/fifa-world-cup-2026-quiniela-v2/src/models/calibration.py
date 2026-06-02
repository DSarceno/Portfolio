"""Probability calibration for multi-class outcome models."""

from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.utils.constants import OUTCOME_TO_INDEX
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class CalibrationStrategy(str, Enum):
    """Supported calibration strategies."""

    ISOTONIC = "isotonic"
    SIGMOID = "sigmoid"


class ProbabilityCalibrator:
    """Per-class probability calibrator using one-vs-rest mapping."""

    def __init__(self, strategy: CalibrationStrategy = CalibrationStrategy.ISOTONIC) -> None:
        """Initialize the calibrator.

        Args:
            strategy: Calibration strategy.
        """
        self.strategy = strategy
        self.calibrators: list = []
        self.is_fitted = False

    def fit(self, proba: np.ndarray, y: pd.Series | np.ndarray) -> "ProbabilityCalibrator":
        """Fit one per-class calibrator on validation probabilities.

        Args:
            proba: Probability matrix ``(n_samples, n_classes)``.
            y: True labels.

        Returns:
            ``self``.
        """
        y_idx = self._labels_to_indices(y)
        n_classes = proba.shape[1]
        self.calibrators = []
        for cls in range(n_classes):
            mask = ~np.isnan(proba[:, cls])
            target = (y_idx == cls).astype(int)[mask]
            if self.strategy == CalibrationStrategy.ISOTONIC:
                cal = IsotonicRegression(out_of_bounds="clip")
                cal.fit(proba[mask, cls], target)
            else:
                cal = LogisticRegression(max_iter=1000)
                cal.fit(proba[mask, cls].reshape(-1, 1), target)
            self.calibrators.append(cal)
        self.is_fitted = True
        logger.info("Calibration (%s) fitted on %d samples", self.strategy.value, len(y_idx))
        return self

    def transform(self, proba: np.ndarray) -> np.ndarray:
        """Apply the calibration to *proba*.

        Args:
            proba: Probability matrix.

        Returns:
            Calibrated probability matrix that sums to 1 row-wise.
        """
        if not self.is_fitted:
            return proba

        out = np.zeros_like(proba, dtype=float)
        for cls, cal in enumerate(self.calibrators):
            if isinstance(cal, IsotonicRegression):
                out[:, cls] = cal.predict(proba[:, cls])
            else:
                out[:, cls] = cal.predict_proba(proba[:, cls].reshape(-1, 1))[:, 1]
        out = np.clip(out, 1e-6, 1 - 1e-6)
        total = out.sum(axis=1, keepdims=True)
        return out / np.clip(total, 1e-9, None)

    @staticmethod
    def _labels_to_indices(y: pd.Series | np.ndarray) -> np.ndarray:
        arr = np.asarray(y)
        if arr.dtype.kind in {"U", "O"}:
            return np.asarray([OUTCOME_TO_INDEX[str(v)] for v in arr], dtype=int)
        return arr.astype(int)
