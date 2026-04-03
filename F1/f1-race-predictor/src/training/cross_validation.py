"""Temporal cross-validation for F1 race prediction models."""

import logging
from typing import Dict, Generator, List, Tuple

import numpy as np
import pandas as pd

from src.models.base_model import BaseModel

logger = logging.getLogger(__name__)


class TemporalCrossValidator:
    """Cross-validator that respects the temporal ordering of F1 data.

    Splits data by season year to prevent future data leaking into
    training folds. Each fold uses earlier seasons as training data.
    """

    def __init__(self, config: Dict, n_splits: int = 5) -> None:
        """Initialize TemporalCrossValidator.

        Args:
            config: Project configuration dictionary.
            n_splits: Number of cross-validation folds.
        """
        self.config = config
        self.n_splits = n_splits
        logger.info(
            "TemporalCrossValidator initialized with %d splits", n_splits
        )

    def split(
        self, data: pd.DataFrame
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Yield (train_idx, val_idx) pairs using temporal ordering.

        Args:
            data: DataFrame with 'Year' column for temporal splitting.

        Yields:
            Tuples of (train_indices, val_indices).

        Raises:
            ValueError: If 'Year' column is missing.
        """
        if "Year" not in data.columns:
            raise ValueError("Data must contain 'Year' column")

        folds = self._get_temporal_folds(data)
        for train_idx, val_idx in folds:
            yield np.array(train_idx), np.array(val_idx)

    def cross_validate(
        self,
        model: BaseModel,
        X: np.ndarray,
        y: np.ndarray,
        data: pd.DataFrame,
    ) -> Dict[str, List[float]]:
        """Run temporal cross-validation on a model.

        Args:
            model: Model instance to validate (re-trained each fold).
            X: Feature matrix aligned with data rows.
            y: Target labels.
            data: DataFrame with 'Year' column for temporal splits.

        Returns:
            Dict of metric_name -> list of per-fold scores.
        """
        cv_results: Dict[str, List[float]] = {
            "mae": [],
            "rmse": [],
            "top_3_accuracy": [],
            "top_5_accuracy": [],
            "exact_accuracy": [],
        }

        for fold_num, (train_idx, val_idx) in enumerate(
            self.split(data), start=1
        ):
            if len(train_idx) == 0 or len(val_idx) == 0:
                logger.warning("Fold %d has empty split, skipping", fold_num)
                continue

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Re-create model for each fold to avoid state leakage
            from src.models.model_factory import ModelFactory
            fold_model = ModelFactory.create(model.name, self.config)

            try:
                fold_model.train(X_train, y_train, X_val, y_val)
                metrics = fold_model.evaluate(X_val, y_val)
                for key in cv_results:
                    cv_results[key].append(metrics.get(key, float("nan")))
                logger.info(
                    "Fold %d — top3=%.3f mae=%.3f",
                    fold_num,
                    metrics.get("top_3_accuracy", float("nan")),
                    metrics.get("mae", float("nan")),
                )
            except Exception as e:
                logger.warning("Fold %d failed: %s", fold_num, e)

        return cv_results

    def get_cv_summary(
        self, cv_results: Dict[str, List[float]]
    ) -> Dict[str, Dict[str, float]]:
        """Compute summary statistics from CV results.

        Args:
            cv_results: Dict of metric_name -> list of fold scores.

        Returns:
            Dict of metric_name -> {mean, std, min, max}.
        """
        summary: Dict[str, Dict[str, float]] = {}
        for metric, scores in cv_results.items():
            clean = [s for s in scores if not np.isnan(s)]
            if not clean:
                summary[metric] = {"mean": float("nan"), "std": float("nan"),
                                   "min": float("nan"), "max": float("nan")}
                continue
            arr = np.array(clean)
            summary[metric] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }
        logger.info("CV summary computed for %d metrics", len(summary))
        return summary

    def _get_temporal_folds(
        self, data: pd.DataFrame
    ) -> List[Tuple[List[int], List[int]]]:
        """Create temporal fold index pairs from year ordering.

        Args:
            data: DataFrame with 'Year' column.

        Returns:
            List of (train_indices, val_indices) tuples.
        """
        years = sorted(data["Year"].unique().tolist())
        n = len(years)
        if n < 2:
            logger.warning("Not enough years for CV splits")
            return []

        n_splits = min(self.n_splits, n - 1)
        folds: List[Tuple[List[int], List[int]]] = []

        # Expanding window: each fold adds one more year to training
        step = max(1, (n - 1) // n_splits)
        for i in range(1, n_splits + 1):
            split_point = min(i * step, n - 1)
            train_years = years[:split_point]
            val_years = [years[split_point]]

            train_idx = data[data["Year"].isin(train_years)].index.tolist()
            val_idx = data[data["Year"].isin(val_years)].index.tolist()

            if train_idx and val_idx:
                folds.append((train_idx, val_idx))

        logger.info("Created %d temporal folds", len(folds))
        return folds
