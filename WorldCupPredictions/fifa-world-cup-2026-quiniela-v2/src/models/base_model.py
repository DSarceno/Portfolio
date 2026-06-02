"""Abstract base class for outcome models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.io import load_pickle, save_pickle
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class BaseOutcomeModel(ABC):
    """Interface for any 1X2 outcome model."""

    def __init__(self, name: str, feature_columns: Optional[list[str]] = None) -> None:
        """Initialize the model.

        Args:
            name: Human-readable model name.
            feature_columns: Optional explicit list of feature columns.
        """
        self.name = name
        self.feature_columns = feature_columns or []
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseOutcomeModel":
        """Fit the model.

        Args:
            X: Feature DataFrame.
            y: Outcome labels.

        Returns:
            ``self``.
        """

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return outcome probabilities with shape ``(n_samples, 3)``.

        Args:
            X: Feature DataFrame.

        Returns:
            Probability matrix.
        """

    def save(self, path: str | Path) -> Path:
        """Persist the model to disk.

        Args:
            path: Destination pickle path.

        Returns:
            The output :class:`Path`.
        """
        return save_pickle(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "BaseOutcomeModel":
        """Load a previously saved model.

        Args:
            path: Pickle path.

        Returns:
            The deserialized model.
        """
        obj = load_pickle(path)
        if not isinstance(obj, BaseOutcomeModel):
            raise TypeError(f"Loaded object is not a BaseOutcomeModel: {type(obj)}")
        return obj

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(f"Model '{self.name}' is not fitted")
