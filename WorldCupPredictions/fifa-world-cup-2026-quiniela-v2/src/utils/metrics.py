"""Evaluation metrics: classification, calibration, and quiniela-specific scoring."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
)

from src.utils.constants import INDEX_TO_OUTCOME, OUTCOMES, OUTCOME_TO_INDEX


def _as_index(labels: Sequence[str | int]) -> np.ndarray:
    arr = np.asarray(labels)
    if arr.dtype.kind in {"U", "O"}:
        return np.asarray([OUTCOME_TO_INDEX[str(x)] for x in arr], dtype=int)
    return arr.astype(int)


def compute_log_loss(y_true: Sequence[str | int], proba: np.ndarray) -> float:
    """Multi-class log loss for the canonical H/D/A label space.

    Args:
        y_true: True labels (strings ``"H"``/``"D"``/``"A"`` or indices).
        proba: Probability matrix with shape ``(n_samples, 3)``.

    Returns:
        Log loss value.
    """
    y_idx = _as_index(y_true)
    return float(log_loss(y_idx, np.clip(proba, 1e-9, 1 - 1e-9), labels=[0, 1, 2]))


def compute_brier(y_true: Sequence[str | int], proba: np.ndarray) -> float:
    """Macro-averaged multi-class Brier score (mean over the three outcomes).

    Args:
        y_true: True labels.
        proba: Probability matrix ``(n_samples, 3)``.

    Returns:
        Brier score.
    """
    y_idx = _as_index(y_true)
    scores: list[float] = []
    for cls in range(3):
        scores.append(brier_score_loss((y_idx == cls).astype(int), proba[:, cls]))
    return float(np.mean(scores))


def compute_accuracy(y_true: Sequence[str | int], proba: np.ndarray) -> float:
    """Top-1 accuracy of an argmax pick over a probability matrix.

    Args:
        y_true: True labels.
        proba: Probability matrix ``(n_samples, 3)``.

    Returns:
        Accuracy in ``[0, 1]``.
    """
    y_idx = _as_index(y_true)
    preds = proba.argmax(axis=1)
    return float(accuracy_score(y_idx, preds))


def compute_macro_f1(y_true: Sequence[str | int], proba: np.ndarray) -> float:
    """Macro-F1 of the argmax classification.

    Args:
        y_true: True labels.
        proba: Probability matrix.

    Returns:
        Macro-averaged F1 score.
    """
    y_idx = _as_index(y_true)
    preds = proba.argmax(axis=1)
    return float(f1_score(y_idx, preds, average="macro", zero_division=0))


def expected_calibration_error(
    y_true: Sequence[str | int],
    proba: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute the expected calibration error (ECE) of a multi-class predictor.

    Args:
        y_true: True labels.
        proba: Probability matrix.
        n_bins: Number of confidence bins.

    Returns:
        ECE in ``[0, 1]``.
    """
    y_idx = _as_index(y_true)
    confidences = proba.max(axis=1)
    predictions = proba.argmax(axis=1)
    accuracies = (predictions == y_idx).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if not np.any(mask):
            continue
        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.mean()) * abs(bin_acc - bin_conf)
    return float(ece)


def quiniela_score(
    picks: Sequence[str],
    actual_outcomes: Sequence[str],
    pick_scorelines: Sequence[tuple[int, int]] | None = None,
    actual_scorelines: Sequence[tuple[int, int]] | None = None,
    correct_1x2: float = 1.0,
    exact_score_bonus: float = 2.0,
    upset_bonus: float = 1.0,
    pick_was_underdog: Sequence[bool] | None = None,
) -> dict[str, float]:
    """Compute total quiniela score given picks and actual results.

    Args:
        picks: List of picked outcomes.
        actual_outcomes: List of actual outcomes.
        pick_scorelines: Optional list of picked scorelines.
        actual_scorelines: Optional list of actual scorelines.
        correct_1x2: Points awarded for a correct outcome.
        exact_score_bonus: Bonus points for a correct exact score.
        upset_bonus: Bonus points when the correct pick was an underdog.
        pick_was_underdog: Flags indicating whether each pick targeted an underdog.

    Returns:
        Dictionary with ``total``, ``correct_outcomes``, ``correct_scores`` and ``upsets_hit``.
    """
    total = 0.0
    correct_outcomes = 0
    correct_scores = 0
    upsets_hit = 0

    for i, (p, a) in enumerate(zip(picks, actual_outcomes)):
        if p == a:
            total += correct_1x2
            correct_outcomes += 1
            if pick_was_underdog is not None and pick_was_underdog[i]:
                total += upset_bonus
                upsets_hit += 1
            if pick_scorelines is not None and actual_scorelines is not None:
                if tuple(pick_scorelines[i]) == tuple(actual_scorelines[i]):
                    total += exact_score_bonus
                    correct_scores += 1

    return {
        "total": float(total),
        "correct_outcomes": float(correct_outcomes),
        "correct_scores": float(correct_scores),
        "upsets_hit": float(upsets_hit),
    }


def reliability_table(
    y_true: Sequence[str | int],
    proba: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Build a reliability diagram dataframe (confidence vs. accuracy per bin).

    Args:
        y_true: True labels.
        proba: Probability matrix.
        n_bins: Bin count.

    Returns:
        DataFrame with ``bin_low``, ``bin_high``, ``mean_confidence``, ``accuracy``, ``count``.
    """
    y_idx = _as_index(y_true)
    confidences = proba.max(axis=1)
    predictions = proba.argmax(axis=1)
    accuracies = (predictions == y_idx).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    records: list[dict[str, float]] = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        count = int(mask.sum())
        records.append(
            {
                "bin_low": float(lo),
                "bin_high": float(hi),
                "mean_confidence": float(confidences[mask].mean()) if count else float("nan"),
                "accuracy": float(accuracies[mask].mean()) if count else float("nan"),
                "count": count,
            }
        )
    return pd.DataFrame(records)


def outcome_label_from_index(idx: int) -> str:
    """Map a class index to the canonical outcome label.

    Args:
        idx: Integer in ``{0, 1, 2}``.

    Returns:
        ``"H"``, ``"D"`` or ``"A"``.
    """
    return INDEX_TO_OUTCOME[idx]


def all_outcomes() -> tuple[str, ...]:
    """Return the immutable tuple of canonical outcomes."""
    return OUTCOMES
