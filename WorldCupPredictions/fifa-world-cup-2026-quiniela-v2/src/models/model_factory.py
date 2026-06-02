"""Factory function for outcome models."""

from __future__ import annotations

from typing import Any, Optional

from src.models.base_model import BaseOutcomeModel
from src.models.multinomial_model import MultinomialOutcomeModel
from src.models.poisson_model import PoissonScoreModel
from src.models.xgboost_model import XGBoostOutcomeModel


def build_model(
    kind: str,
    feature_columns: Optional[list[str]] = None,
    hyperparameters: Optional[dict[str, Any]] = None,
) -> BaseOutcomeModel:
    """Return an unfitted model instance.

    Args:
        kind: Model family identifier (``"multinomial"``, ``"xgboost"``, ``"poisson"``).
        feature_columns: Optional list of feature columns.
        hyperparameters: Optional hyperparameter dictionary.

    Returns:
        An unfitted :class:`BaseOutcomeModel`.

    Raises:
        ValueError: When ``kind`` is unknown.
    """
    kind = kind.lower()
    if kind == "multinomial":
        return MultinomialOutcomeModel(feature_columns=feature_columns, hyperparameters=hyperparameters)
    if kind == "xgboost":
        return XGBoostOutcomeModel(feature_columns=feature_columns, hyperparameters=hyperparameters)
    if kind == "poisson":
        return PoissonScoreModel(hyperparameters=hyperparameters)
    raise ValueError(f"Unknown model kind: {kind}")
