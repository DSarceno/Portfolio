"""Combine Elo, PI and rolling form into a single team-strength signal."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.ratings.elo import EloRating
from src.ratings.form_rating import FormRating
from src.ratings.pi_rating import PIRating
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class EnsembleWeights:
    """Weights applied to each rating system before averaging."""

    elo: float = 0.55
    pi: float = 0.30
    form: float = 0.15


class RatingEnsemble:
    """Container that runs Elo + PI + Form rating on a common match table."""

    def __init__(
        self,
        elo: Optional[EloRating] = None,
        pi: Optional[PIRating] = None,
        form: Optional[FormRating] = None,
        weights: Optional[EnsembleWeights] = None,
    ) -> None:
        """Initialize the ensemble.

        Args:
            elo: Optional pre-configured :class:`EloRating`.
            pi: Optional pre-configured :class:`PIRating`.
            form: Optional pre-configured :class:`FormRating`.
            weights: Optional blend weights.
        """
        self.elo = elo or EloRating()
        self.pi = pi or PIRating()
        self.form = form or FormRating()
        self.weights = weights or EnsembleWeights()

    def fit(self, matches: pd.DataFrame) -> "RatingEnsemble":
        """Fit all three rating systems on *matches*.

        Args:
            matches: Canonical match table.

        Returns:
            ``self``.
        """
        self.elo.fit(matches)
        self.pi.fit(matches)
        self.form.fit(matches)
        logger.info("Rating ensemble fit completed")
        return self

    def composite_table(self) -> pd.DataFrame:
        """Build a per-team table with a z-scored composite strength index.

        Returns:
            DataFrame with columns ``team``, ``elo``, ``pi_combined``,
            ``form_score``, and ``composite_strength``.
        """
        elo_df = self.elo.snapshot()
        pi_df = self.pi.snapshot()[["team", "pi_combined"]]
        form_df = self.form.snapshot()[["team", "form_score"]]

        merged = elo_df.merge(pi_df, on="team", how="outer").merge(
            form_df, on="team", how="outer"
        )
        for col in ["elo", "pi_combined", "form_score"]:
            if col not in merged.columns:
                merged[col] = 0.0
            merged[col] = merged[col].fillna(merged[col].median() if not merged[col].isna().all() else 0.0)

        merged["elo_z"] = self._zscore(merged["elo"])
        merged["pi_z"] = self._zscore(merged["pi_combined"])
        merged["form_z"] = self._zscore(merged["form_score"])

        w = self.weights
        total = w.elo + w.pi + w.form
        merged["composite_strength"] = (
            w.elo * merged["elo_z"] + w.pi * merged["pi_z"] + w.form * merged["form_z"]
        ) / total
        return merged.sort_values("composite_strength", ascending=False).reset_index(drop=True)

    @staticmethod
    def _zscore(series: pd.Series) -> pd.Series:
        """Return the z-score of *series*; returns zeros when std is zero."""
        std = series.std()
        if std == 0 or np.isnan(std):
            return pd.Series([0.0] * len(series), index=series.index)
        return (series - series.mean()) / std
