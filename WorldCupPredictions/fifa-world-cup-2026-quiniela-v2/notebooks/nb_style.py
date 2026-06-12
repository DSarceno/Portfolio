"""Shared styling and helpers for the analysis notebooks.

Import once at the top of each notebook::

    import nb_style as nbs
    nbs.apply_theme()

Provides a single seaborn theme, a confederation colour map and small plotting
helpers so every notebook looks consistent and presentable.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import seaborn as sns

# Make ``src`` importable from within notebooks/ (cwd = notebooks/) and run from
# the project root so the relative data/model/output paths used by DataLoader,
# the feature builder and the simulation outputs all resolve — regardless of
# whether the kernel was started in notebooks/ (e.g. via ``nbconvert``) or root.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

from src.ratings.shrinkage import TEAM_CONFEDERATION  # noqa: E402

# Qualitative palette (color-blind friendly) reused across notebooks.
PALETTE = sns.color_palette("colorblind")

CONFEDERATION_COLORS: dict[str, str] = {
    "UEFA": "#1f77b4",
    "CONMEBOL": "#2ca02c",
    "CONCACAF": "#ff7f0e",
    "AFC": "#d62728",
    "CAF": "#9467bd",
    "OFC": "#8c564b",
    "UNKNOWN": "#7f7f7f",
}


def apply_theme() -> None:
    """Apply the shared seaborn/matplotlib theme."""
    sns.set_theme(style="whitegrid", context="notebook", palette="colorblind")
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 110,
            "figure.titlesize": 15,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.edgecolor": "#444444",
            "font.size": 11,
            "legend.frameon": False,
        }
    )


def team_confederation(team: str) -> str:
    """Return a team's confederation (``"UNKNOWN"`` if unmapped)."""
    return TEAM_CONFEDERATION.get(str(team), "UNKNOWN")


def confederation_color(team: str) -> str:
    """Return the plotting colour for a team's confederation."""
    return CONFEDERATION_COLORS[team_confederation(team)]


def kpi_header(items: Sequence[tuple[str, str]], title: str | None = None) -> None:
    """Render a row of KPI cards.

    Args:
        items: Sequence of ``(label, value)`` pairs.
        title: Optional figure title.
    """
    n = len(items)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 1.6))
    if n == 1:
        axes = [axes]
    for ax, (label, value) in zip(axes, items):
        ax.axis("off")
        ax.text(
            0.5,
            0.62,
            str(value),
            ha="center",
            va="center",
            fontsize=20,
            fontweight="bold",
            color="#1f77b4",
        )
        ax.text(0.5, 0.18, label, ha="center", va="center", fontsize=10, color="#444444")
    if title:
        fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    plt.show()


def annotate_barh(
    ax: plt.Axes, values: Iterable[float], fmt: str = "{:.0f}", pad: float = 0.0
) -> None:
    """Write value labels at the end of horizontal bars."""
    for i, v in enumerate(values):
        ax.text(float(v) + pad, i, fmt.format(v), va="center", fontsize=8)
