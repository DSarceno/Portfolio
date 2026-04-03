"""Baseline F1 race position predictor using grid position heuristics."""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.models.base_model import BaseModel

logger = logging.getLogger(__name__)


class BaselineModel(BaseModel):
    """Baseline model that predicts race position from grid position.

    Uses historical grid-to-finish deltas to adjust raw grid position.
    Serves as a lower bound for more sophisticated models.
    """

    GRID_POSITION_COL = "GridPosition"
    DRIVER_COL = "Abbreviation"

    def __init__(self, config: Dict) -> None:
        """Initialize BaselineModel.

        Args:
            config: Model configuration dictionary.
        """
        super().__init__("baseline", config)
        params = config.get("models", {}).get("baseline", {}).get(
            "parameters", {}
        )
        self._use_historical_delta = bool(
            params.get("use_historical_delta", True)
        )
        self._dnf_threshold = float(params.get("dnf_threshold", 0.2))
        self._max_position_change = int(params.get("max_position_change", 5))
        self._avg_delta: float = 0.0
        logger.info("BaselineModel initialized")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        """Fit baseline by computing mean grid-to-finish delta.

        Args:
            X_train: Training features (first column assumed to be GridPosition).
            y_train: True race positions.
            X_val: Unused.
            y_val: Unused.
        """
        if self._use_historical_delta and X_train.shape[1] > 0:
            grid_positions = X_train[:, 0].astype(float)
            deltas = grid_positions - y_train.astype(float)
            self._avg_delta = float(np.nanmean(deltas))
            logger.info("Baseline avg delta: %.3f", self._avg_delta)
        else:
            self._avg_delta = 0.0

        self._is_trained = True
        logger.info("BaselineModel trained on %d samples", len(y_train))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict positions as grid_position minus avg_delta.

        Args:
            X: Feature matrix (first column = GridPosition).

        Returns:
            Array of integer position predictions, clipped to [1, 20].

        Raises:
            RuntimeError: If model has not been trained.
        """
        if not self._is_trained:
            raise RuntimeError("BaselineModel must be trained before predict()")

        grid_positions = X[:, 0].astype(float)
        predictions = grid_positions - self._avg_delta
        predictions = np.clip(np.round(predictions), 1, 20).astype(int)
        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate soft probability matrix from position estimates.

        Uses a Gaussian distribution centred on the predicted position.

        Args:
            X: Feature matrix.

        Returns:
            Probability matrix of shape (n_samples, 20).
        """
        positions = self.predict(X)
        n = len(positions)
        proba = np.zeros((n, 20))
        for i, pos in enumerate(positions):
            # Gaussian spread around predicted position
            for j in range(20):
                dist = abs(j + 1 - pos)
                proba[i, j] = np.exp(-0.5 * (dist / 2.0) ** 2)
            row_sum = proba[i].sum()
            if row_sum > 0:
                proba[i] /= row_sum
        return proba

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Baseline has no feature importance.

        Returns:
            None always.
        """
        return None

    def _compute_historical_delta(
        self, data: pd.DataFrame
    ) -> Dict[str, float]:
        """Compute per-driver historical grid-to-finish deltas.

        Args:
            data: Historical race DataFrame with GridPosition and Position.

        Returns:
            Dict mapping driver abbreviation to avg delta.
        """
        if (
            self.GRID_POSITION_COL not in data.columns
            or "Position" not in data.columns
        ):
            return {}
        data = data.copy()
        data["delta"] = (
            pd.to_numeric(data[self.GRID_POSITION_COL], errors="coerce")
            - pd.to_numeric(data["Position"], errors="coerce")
        )
        if self.DRIVER_COL in data.columns:
            return data.groupby(self.DRIVER_COL)["delta"].mean().to_dict()
        return {"global": float(data["delta"].mean())}
