"""Rolling-origin backtester."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from src.features.build_features import select_feature_columns
from src.models.multinomial_model import MultinomialOutcomeModel
from src.models.xgboost_model import XGBoostOutcomeModel
from src.training.cross_validation import rolling_origin_splits
from src.training.evaluator import Evaluator
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class BacktestResult:
    """Container with per-fold metrics."""

    per_fold: pd.DataFrame
    aggregate: pd.Series


class Backtester:
    """Rolling-origin backtester for outcome models."""

    def __init__(self, n_folds: int = 5, min_train_size: int = 200) -> None:
        """Initialize the backtester.

        Args:
            n_folds: Number of rolling-origin folds.
            min_train_size: Minimum training rows per fold.
        """
        self.n_folds = n_folds
        self.min_train_size = min_train_size

    def run(
        self,
        features: pd.DataFrame,
        feature_columns: Optional[list[str]] = None,
        label_column: str = "outcome",
        date_column: str = "date",
    ) -> BacktestResult:
        """Run the backtest.

        Args:
            features: Feature DataFrame with labels.
            feature_columns: Optional explicit feature column list.
            label_column: Outcome label column.
            date_column: Date column used to order rows.

        Returns:
            :class:`BacktestResult`.

        Raises:
            ValueError: If *features* is empty.
        """
        if features.empty:
            raise ValueError("No features supplied to backtester")

        features = features.copy()
        features[date_column] = pd.to_datetime(features[date_column], errors="coerce")
        features = features.dropna(subset=[date_column, label_column]).sort_values(date_column)

        cols = feature_columns or select_feature_columns(features)
        evaluator = Evaluator()
        records: list[dict[str, float]] = []
        for fold_idx, (train_idx, test_idx) in enumerate(
            rolling_origin_splits(features, n_splits=self.n_folds, min_train_size=self.min_train_size)
        ):
            train = features.iloc[train_idx]
            test = features.iloc[test_idx]
            if train.empty or test.empty:
                continue

            model = XGBoostOutcomeModel(feature_columns=cols)
            try:
                model.fit(train[cols], train[label_column].astype(str))
            except (ValueError, RuntimeError) as exc:
                logger.warning("XGBoost failed on fold %d (%s); falling back to multinomial", fold_idx, exc)
                model = MultinomialOutcomeModel(feature_columns=cols)
                model.fit(train[cols], train[label_column].astype(str))

            proba = model.predict_proba(test[cols])
            report = evaluator.evaluate(
                y_true=test[label_column].astype(str).tolist(),
                proba=proba,
            )
            records.append({"fold": fold_idx, **report.to_dict()})

        per_fold = pd.DataFrame(records)
        if per_fold.empty:
            aggregate = pd.Series(dtype=float)
        else:
            aggregate = per_fold.drop(columns=["fold"]).mean()
        logger.info("Backtest aggregate: %s", aggregate.to_dict())
        return BacktestResult(per_fold=per_fold, aggregate=aggregate)
