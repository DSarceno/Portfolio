"""Abstract base class for all F1 race prediction models."""

import logging
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base for all prediction models.

    Subclasses must implement train, predict, predict_proba,
    and get_feature_importance.
    """

    def __init__(self, name: str, config: Dict) -> None:
        """Initialize BaseModel.

        Args:
            name: Human-readable model name.
            config: Model configuration dictionary.
        """
        self._name = name
        self._config = config
        self._model = None
        self._is_trained: bool = False
        logger.info("Model '%s' created", name)

    @property
    def name(self) -> str:
        """Model identifier string."""
        return self._name

    @property
    def is_trained(self) -> bool:
        """Whether the model has been fitted."""
        return self._is_trained

    @abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        """Fit the model on training data.

        Args:
            X_train: Training feature matrix.
            y_train: Training target array (positions 1-20).
            X_val: Optional validation feature matrix.
            y_val: Optional validation target array.
        """

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict race positions.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Array of predicted positions (1-indexed integers).
        """

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Probability matrix of shape (n_samples, n_classes).
        """

    @abstractmethod
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Return feature importance scores if available.

        Returns:
            Dict mapping feature name to importance, or None.
        """

    def evaluate(
        self, X: np.ndarray, y_true: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate model on provided data.

        Args:
            X: Feature matrix.
            y_true: True position labels.

        Returns:
            Dictionary with mae, rmse, exact_accuracy metrics.

        Raises:
            RuntimeError: If model has not been trained.
        """
        if not self._is_trained:
            raise RuntimeError(f"Model '{self._name}' is not trained")

        y_pred = self.predict(X)
        errors = np.abs(y_true - y_pred)
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        mae = float(np.mean(errors))
        exact = float(np.mean(y_true == y_pred))
        top3 = float(np.mean(errors < 3))
        top5 = float(np.mean(errors < 5))

        metrics = {
            "mae": mae,
            "rmse": rmse,
            "exact_accuracy": exact,
            "top_3_accuracy": top3,
            "top_5_accuracy": top5,
        }
        logger.info("Evaluation for '%s': MAE=%.3f, Top3=%.3f", self._name, mae, top3)
        return metrics

    def save(self, path: str) -> None:
        """Serialize model to disk using pickle.

        Args:
            path: Destination file path.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Model '%s' saved to %s", self._name, path)

    @classmethod
    def load(cls, path: str) -> "BaseModel":
        """Load a serialized model from disk.

        Args:
            path: Source file path.

        Returns:
            Loaded model instance.

        Raises:
            FileNotFoundError: If file does not exist.
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info("Loaded model from %s", path)
        return model
