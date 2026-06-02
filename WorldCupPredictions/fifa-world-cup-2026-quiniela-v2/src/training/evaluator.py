"""Evaluation helpers: prints classical metrics and quiniela-specific metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from src.utils.logging_config import get_logger
from src.utils.metrics import (
    compute_accuracy,
    compute_brier,
    compute_log_loss,
    compute_macro_f1,
    expected_calibration_error,
    quiniela_score,
)

logger = get_logger(__name__)


@dataclass
class EvaluationReport:
    """Container with computed metrics."""

    log_loss: float
    brier: float
    accuracy: float
    macro_f1: float
    calibration_error: float
    quiniela_metrics: dict[str, float]

    def to_dict(self) -> dict[str, float]:
        out = {
            "log_loss": self.log_loss,
            "brier": self.brier,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "calibration_error": self.calibration_error,
        }
        out.update({f"quiniela_{k}": v for k, v in self.quiniela_metrics.items()})
        return out


class Evaluator:
    """Compute predictive metrics, including custom quiniela scoring."""

    def evaluate(
        self,
        y_true: Sequence[str],
        proba: np.ndarray,
        picks: Sequence[str] | None = None,
        actual_outcomes: Sequence[str] | None = None,
        pick_was_underdog: Sequence[bool] | None = None,
    ) -> EvaluationReport:
        """Compute the full metric suite.

        Args:
            y_true: True labels.
            proba: Probability matrix.
            picks: Optional picks (defaults to argmax).
            actual_outcomes: Optional actual outcomes for the picks.
            pick_was_underdog: Optional flag indicating an underdog pick.

        Returns:
            :class:`EvaluationReport`.
        """
        report = EvaluationReport(
            log_loss=compute_log_loss(y_true, proba),
            brier=compute_brier(y_true, proba),
            accuracy=compute_accuracy(y_true, proba),
            macro_f1=compute_macro_f1(y_true, proba),
            calibration_error=expected_calibration_error(y_true, proba),
            quiniela_metrics={},
        )
        if picks is not None and actual_outcomes is not None:
            report.quiniela_metrics = quiniela_score(
                picks=picks,
                actual_outcomes=actual_outcomes,
                pick_was_underdog=pick_was_underdog,
            )
        logger.info("Evaluation: %s", report.to_dict())
        return report

    def evaluate_dataframe(
        self,
        df: pd.DataFrame,
        outcome_col: str = "outcome",
        proba_cols: tuple[str, str, str] = ("p_home", "p_draw", "p_away"),
        pick_col: str | None = None,
    ) -> EvaluationReport:
        """Evaluate a DataFrame with stacked outcome + probability columns.

        Args:
            df: DataFrame with labels, probabilities and optional picks.
            outcome_col: Outcome column name.
            proba_cols: Column names of the three probabilities.
            pick_col: Optional pick column name.

        Returns:
            :class:`EvaluationReport`.
        """
        proba = df[list(proba_cols)].to_numpy(dtype=float)
        y_true = df[outcome_col].astype(str).tolist()
        picks = df[pick_col].tolist() if pick_col and pick_col in df.columns else None
        return self.evaluate(y_true=y_true, proba=proba, picks=picks, actual_outcomes=y_true)
