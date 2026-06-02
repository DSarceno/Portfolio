"""Plotting helpers for diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.io import ensure_dir
from src.utils.metrics import reliability_table


def plot_reliability(
    y_true: Sequence[str | int],
    proba: np.ndarray,
    path: str | Path,
    title: str = "Reliability diagram",
) -> Path:
    """Save a reliability diagram to *path*.

    Args:
        y_true: True labels.
        proba: Probability matrix ``(n_samples, 3)``.
        path: Output PNG path.
        title: Plot title.

    Returns:
        The saved :class:`Path`.
    """
    table = reliability_table(y_true, proba)
    out = Path(path)
    ensure_dir(out.parent)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfect calibration")
    ax.plot(
        table["mean_confidence"],
        table["accuracy"],
        marker="o",
        linewidth=2,
        label="Model",
    )
    ax.set_xlabel("Mean confidence per bin")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_probability_bars(
    df: pd.DataFrame,
    home_col: str = "p_home",
    draw_col: str = "p_draw",
    away_col: str = "p_away",
    label_col: str = "label",
    path: str | Path = "outputs/diagnostics/probabilities.png",
    title: str = "Match probabilities",
) -> Path:
    """Plot stacked match probability bars.

    Args:
        df: DataFrame with probability columns.
        home_col: Column name for home probabilities.
        draw_col: Column name for draw probabilities.
        away_col: Column name for away probabilities.
        label_col: Column used to label the bars.
        path: Output PNG path.
        title: Plot title.

    Returns:
        The saved :class:`Path`.
    """
    out = Path(path)
    ensure_dir(out.parent)

    labels = df[label_col].astype(str).tolist()
    fig, ax = plt.subplots(figsize=(max(8, 0.4 * len(labels)), 5))
    bottom = np.zeros(len(df))
    for col, color in zip([home_col, draw_col, away_col], ["#1f77b4", "#bbbbbb", "#d62728"]):
        ax.bar(labels, df[col].values, bottom=bottom, color=color, label=col)
        bottom += df[col].values

    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.set_title(title)
    ax.legend(loc="upper right")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
