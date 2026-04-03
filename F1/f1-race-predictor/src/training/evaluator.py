"""Model evaluation utilities for F1 race prediction."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from src.models.base_model import BaseModel

logger = logging.getLogger(__name__)


class Evaluator:
    """Computes and reports evaluation metrics for F1 prediction models."""

    def __init__(self, config: Dict) -> None:
        """Initialize Evaluator.

        Args:
            config: Project configuration with evaluation metrics list.
        """
        self.config = config
        eval_cfg = config.get("evaluation", {})
        self._metrics = eval_cfg.get(
            "metrics",
            [
                "mae",
                "rmse",
                "top_3_accuracy",
                "top_5_accuracy",
                "spearman_corr",
                "exact_accuracy",
            ],
        )
        logger.info("Evaluator initialized with metrics: %s", self._metrics)

    def evaluate(
        self, model: BaseModel, X: np.ndarray, y_true: np.ndarray
    ) -> Dict[str, float]:
        """Run full evaluation suite on a model.

        Args:
            model: Trained model to evaluate.
            X: Feature matrix.
            y_true: True position labels.

        Returns:
            Dictionary of metric_name -> score.
        """
        y_pred = model.predict(X)
        errors = np.abs(y_true - y_pred)
        metrics: Dict[str, float] = {}

        if "mae" in self._metrics:
            metrics["mae"] = self._compute_mae(y_true, y_pred)
        if "rmse" in self._metrics:
            metrics["rmse"] = self._compute_rmse(y_true, y_pred)
        if "top_3_accuracy" in self._metrics:
            metrics["top_3_accuracy"] = self._compute_top_k_accuracy(
                y_true, y_pred, k=3
            )
        if "top_5_accuracy" in self._metrics:
            metrics["top_5_accuracy"] = self._compute_top_k_accuracy(
                y_true, y_pred, k=5
            )
        if "spearman_corr" in self._metrics:
            metrics["spearman_corr"] = self._compute_spearman(y_true, y_pred)
        if "exact_accuracy" in self._metrics:
            metrics["exact_accuracy"] = float(np.mean(y_true == y_pred))

        logger.info(
            "Evaluation for '%s': %s",
            model.name,
            {k: f"{v:.4f}" for k, v in metrics.items()},
        )
        return metrics

    def evaluate_all_models(
        self,
        models_dict: Dict[str, BaseModel],
        X: np.ndarray,
        y_true: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate all provided models.

        Args:
            models_dict: Dict of model_name -> BaseModel.
            X: Feature matrix.
            y_true: True labels.

        Returns:
            Dict of model_name -> metrics dict.
        """
        results: Dict[str, Dict[str, float]] = {}
        for name, model in models_dict.items():
            try:
                results[name] = self.evaluate(model, X, y_true)
            except Exception as e:
                logger.warning("Evaluation failed for %s: %s", name, e)
                results[name] = {}
        return results

    def compare_models(
        self, results: Dict[str, Dict[str, float]]
    ) -> pd.DataFrame:
        """Create a comparison DataFrame from evaluation results.

        Args:
            results: Dict of model_name -> metrics dict.

        Returns:
            DataFrame with models as rows and metrics as columns.
        """
        if not results:
            return pd.DataFrame()
        df = pd.DataFrame(results).T
        df.index.name = "model"
        logger.info("Model comparison table created with %d rows", len(df))
        return df

    def _compute_mae(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> float:
        """Compute mean absolute error.

        Args:
            y_true: True values.
            y_pred: Predicted values.

        Returns:
            MAE float.
        """
        return float(np.mean(np.abs(y_true - y_pred)))

    def _compute_rmse(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> float:
        """Compute root mean squared error.

        Args:
            y_true: True values.
            y_pred: Predicted values.

        Returns:
            RMSE float.
        """
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    def _compute_top_k_accuracy(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        k: int,
    ) -> float:
        """Compute top-k accuracy.

        Args:
            y_true: True positions.
            y_pred: Predicted positions.
            k: Tolerance window.

        Returns:
            Fraction of predictions within k positions.
        """
        return float(np.mean(np.abs(y_true - y_pred) < k))

    def _compute_spearman(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> float:
        """Compute Spearman rank correlation.

        Args:
            y_true: True values.
            y_pred: Predicted values.

        Returns:
            Correlation coefficient.
        """
        corr, _ = stats.spearmanr(y_true, y_pred)
        return float(corr)

    def generate_evaluation_report(
        self,
        results: Dict[str, Dict[str, float]],
        output_path: str,
    ) -> None:
        """Save evaluation results to a JSON file.

        Args:
            results: Dict of model_name -> metrics dict.
            output_path: Destination file path.
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info("Evaluation report saved to %s", output_path)
