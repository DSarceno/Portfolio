"""Blend probabilities from multiple outcome / scoreline models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping, Optional

import numpy as np

from src.utils.logging_config import get_logger

if TYPE_CHECKING:
    from src.utils.config import Config

logger = get_logger(__name__)


@dataclass
class BlendWeights:
    """Blend weights for each contributing model."""

    ratings: float = 0.25
    multinomial: float = 0.20
    xgboost: float = 0.30
    poisson: float = 0.25

    @classmethod
    def from_model_params(cls, model_params: Optional[Mapping[str, Any]]) -> "BlendWeights":
        """Build weights from a ``model_params.yaml`` mapping.

        Reads ``ensemble.weights.{ratings,multinomial,xgboost,poisson}`` and
        falls back to the class defaults for any missing key. Unknown or
        non-numeric values fall back silently so a malformed config never
        breaks inference.

        Args:
            model_params: The parsed ``model_params.yaml`` tree (e.g.
                ``Config.model_params``). ``None`` yields the defaults.

        Returns:
            A populated :class:`BlendWeights`.
        """
        defaults = cls()
        if not isinstance(model_params, Mapping):
            return defaults
        ensemble = model_params.get("ensemble")
        weights = ensemble.get("weights") if isinstance(ensemble, Mapping) else None
        if not isinstance(weights, Mapping):
            return defaults

        def _read(key: str, fallback: float) -> float:
            try:
                return float(weights[key])
            except (KeyError, TypeError, ValueError):
                return fallback

        return cls(
            ratings=_read("ratings", defaults.ratings),
            multinomial=_read("multinomial", defaults.multinomial),
            xgboost=_read("xgboost", defaults.xgboost),
            poisson=_read("poisson", defaults.poisson),
        )

    @classmethod
    def from_config(cls, config: "Config") -> "BlendWeights":
        """Build weights from a loaded :class:`~src.utils.config.Config`."""
        return cls.from_model_params(getattr(config, "model_params", None))


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
        active = [
            (name, arr, w) for name, (arr, w) in provided.items() if arr is not None and w > 0
        ]
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
