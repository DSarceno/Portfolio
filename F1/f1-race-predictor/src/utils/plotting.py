"""Visualization utilities for F1 Race Predictor."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid")


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    title: str = "Feature Importance",
    save_path: Optional[str] = None,
) -> None:
    """Plot horizontal bar chart of feature importances.

    Args:
        feature_names: List of feature name strings.
        importances: Array of importance scores.
        title: Plot title.
        save_path: If provided, save figure to this path.
    """
    sorted_idx = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(10, max(6, len(feature_names) * 0.4)))
    ax.barh(
        [feature_names[i] for i in sorted_idx],
        importances[sorted_idx],
        color="steelblue",
    )
    ax.set_xlabel("Importance Score")
    ax.set_title(title)
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_prediction_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predicted vs Actual Positions",
    save_path: Optional[str] = None,
) -> None:
    """Scatter plot of predicted vs actual positions.

    Args:
        y_true: Array of true positions.
        y_pred: Array of predicted positions.
        title: Plot title.
        save_path: If provided, save figure to this path.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true, y_pred, alpha=0.5, color="steelblue", s=20)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect")
    ax.set_xlabel("Actual Position")
    ax.set_ylabel("Predicted Position")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_training_history(
    history_dict: Dict[str, List[float]],
    save_path: Optional[str] = None,
) -> None:
    """Plot training and validation loss curves.

    Args:
        history_dict: Dict with keys like 'train_loss', 'val_loss'.
        save_path: If provided, save figure to this path.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    for key, values in history_dict.items():
        ax.plot(values, label=key)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training History")
    ax.legend()
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_race_results(
    predictions_df: pd.DataFrame,
    actual_df: Optional[pd.DataFrame],
    race_name: str,
    save_path: Optional[str] = None,
) -> None:
    """Visualize predicted vs actual race results.

    Args:
        predictions_df: DataFrame with 'driver' and 'predicted_position'.
        actual_df: Optional DataFrame with 'driver' and 'position'.
        race_name: Name of the race for the title.
        save_path: If provided, save figure to this path.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    drivers = predictions_df["driver"].tolist()
    pred_pos = predictions_df["predicted_position"].tolist()

    x = np.arange(len(drivers))
    width = 0.35

    ax.bar(x - width / 2, pred_pos, width, label="Predicted", color="steelblue")
    if actual_df is not None:
        actual_df = actual_df.set_index("driver").reindex(drivers)
        ax.bar(
            x + width / 2,
            actual_df["position"].tolist(),
            width,
            label="Actual",
            color="coral",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(drivers, rotation=45, ha="right")
    ax.set_ylabel("Position")
    ax.set_title(f"Race Results: {race_name}")
    ax.invert_yaxis()
    ax.legend()
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_model_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
) -> None:
    """Bar chart comparing metrics across models.

    Args:
        metrics_dict: Dict of model_name -> {metric_name: value}.
        save_path: If provided, save figure to this path.
    """
    models = list(metrics_dict.keys())
    if not models:
        logger.warning("No models to compare")
        return

    metrics = list(next(iter(metrics_dict.values())).keys())
    x = np.arange(len(metrics))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, model in enumerate(models):
        values = [metrics_dict[model].get(m, 0) for m in metrics]
        ax.bar(x + i * width, values, width, label=model)

    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(metrics, rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.legend()
    plt.tight_layout()
    _save_or_show(fig, save_path)


def _save_or_show(fig: plt.Figure, save_path: Optional[str]) -> None:
    """Save figure to file or display it.

    Args:
        fig: Matplotlib figure to save or show.
        save_path: File path to save to, or None to display.
    """
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved plot to %s", save_path)
    else:
        plt.show()
    plt.close(fig)
