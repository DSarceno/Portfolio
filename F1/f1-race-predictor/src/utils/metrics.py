"""Custom evaluation metrics for F1 race prediction."""

import logging
from typing import Dict

import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    ndcg_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


def top_k_accuracy(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """Compute top-K accuracy for position predictions.

    A prediction is correct if the true position is within k positions
    of the predicted position.

    Args:
        y_true: Array of true positions (1-indexed).
        y_pred: Array of predicted positions (1-indexed).
        k: Tolerance window.

    Returns:
        Fraction of predictions within k positions of truth.

    Raises:
        ValueError: If arrays have different lengths or k < 1.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if k < 1:
        raise ValueError("k must be >= 1")

    correct = np.sum(np.abs(y_true - y_pred) < k)
    accuracy = float(correct) / len(y_true)
    logger.debug("Top-%d accuracy: %.4f", k, accuracy)
    return accuracy


def mean_absolute_error_positions(
    y_true: np.ndarray, y_pred: np.ndarray
) -> float:
    """Compute mean absolute error for position predictions.

    Args:
        y_true: Array of true positions.
        y_pred: Array of predicted positions.

    Returns:
        Mean absolute error in positions.

    Raises:
        ValueError: If arrays have different lengths.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    mae = float(np.mean(np.abs(y_true - y_pred)))
    logger.debug("MAE positions: %.4f", mae)
    return mae


def spearman_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Spearman rank correlation between true and predicted positions.

    Args:
        y_true: Array of true positions.
        y_pred: Array of predicted positions.

    Returns:
        Spearman correlation coefficient in [-1, 1].

    Raises:
        ValueError: If arrays have different lengths.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    correlation, _ = stats.spearmanr(y_true, y_pred)
    logger.debug("Spearman correlation: %.4f", correlation)
    return float(correlation)


def ndcg_score_custom(
    y_true: np.ndarray, y_pred: np.ndarray, k: int = 10
) -> float:
    """Compute Normalized Discounted Cumulative Gain for rankings.

    Args:
        y_true: Array of true positions (lower = better).
        y_pred: Array of predicted positions (lower = better).
        k: Number of top positions to consider.

    Returns:
        NDCG score in [0, 1], higher is better.
    """
    # Convert positions to relevance scores (inverted, higher = more relevant)
    max_pos = max(y_true.max(), y_pred.max()) + 1
    relevance_true = (max_pos - y_true).reshape(1, -1).astype(float)
    relevance_pred = (max_pos - y_pred).reshape(1, -1).astype(float)

    try:
        score = ndcg_score(relevance_true, relevance_pred, k=k)
    except Exception as e:
        logger.warning("NDCG computation failed: %s", e)
        return 0.0

    logger.debug("NDCG@%d: %.4f", k, score)
    return float(score)


def dnf_classification_metrics(
    y_true_dnf: np.ndarray, y_pred_dnf: np.ndarray
) -> Dict[str, float]:
    """Compute classification metrics for DNF prediction.

    Args:
        y_true_dnf: Binary array, 1 = DNF, 0 = finished.
        y_pred_dnf: Binary array of predicted DNFs.

    Returns:
        Dictionary with accuracy, precision, recall, f1_score.

    Raises:
        ValueError: If arrays have different lengths.
    """
    if len(y_true_dnf) != len(y_pred_dnf):
        raise ValueError("Arrays must have the same length")

    try:
        metrics = {
            "accuracy": float(accuracy_score(y_true_dnf, y_pred_dnf)),
            "precision": float(
                precision_score(y_true_dnf, y_pred_dnf, zero_division=0)
            ),
            "recall": float(
                recall_score(y_true_dnf, y_pred_dnf, zero_division=0)
            ),
            "f1_score": float(
                f1_score(y_true_dnf, y_pred_dnf, zero_division=0)
            ),
        }
    except Exception as e:
        logger.error("Failed to compute DNF metrics: %s", e)
        raise

    logger.debug("DNF metrics: %s", metrics)
    return metrics
