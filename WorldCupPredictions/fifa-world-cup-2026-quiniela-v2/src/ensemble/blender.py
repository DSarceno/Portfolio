"""Blend probabilities from multiple outcome / scoreline models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class BlendWeights:
    """Blend weights for each contributing model."""

    ratings: float = 0.25
    multinomial: float = 0.20
    xgboost: float = 0.30
    poisson: float = 0.25


@dataclass
class ProbabilityBlender:
    """Weighted blend of per-model probability matrices."""

    weights: BlendWeights = field(default_factory=BlendWeights)

    def blend(
        self,
        ratings: Optional[np.ndarray] = None,
        multinomial: Optional[np.ndarray] = None,
        xgboost: Optional[np.ndarray] = None,
        poisson: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Combine available probability matrices using the configured weights.

        Args:
            ratings: Probabilities from the rating ensemble.
            multinomial: Probabilities from the multinomial model.
            xgboost: Probabilities from XGBoost.
            poisson: Probabilities derived from the Poisson scoreline grid.

        Returns:
            Blended probability matrix that sums to 1 row-wise.

        Raises:
            ValueError: When no probability matrix is provided.
        """
        provided = {
            "ratings": (ratings, self.weights.ratings),
            "multinomial": (multinomial, self.weights.multinomial),
            "xgboost": (xgboost, self.weights.xgboost),
            "poisson": (poisson, self.weights.poisson),
        }
        active = [(name, arr, w) for name, (arr, w) in provided.items() if arr is not None and w > 0]
        if not active:
            raise ValueError("No probability matrices supplied to blend()")

        total_weight = sum(w for _, _, w in active)
        accumulator = np.zeros_like(active[0][1], dtype=float)
        for name, arr, w in active:
            accumulator += (w / total_weight) * np.asarray(arr, dtype=float)
            logger.debug("Blender included %s with effective weight %.3f", name, w / total_weight)

        accumulator = np.clip(accumulator, 1e-6, 1 - 1e-6)
        accumulator /= accumulator.sum(axis=1, keepdims=True)
        return accumulator
