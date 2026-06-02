"""Recommend exact scorelines from the Poisson model."""

from __future__ import annotations

from typing import Optional

import pandas as pd

from src.models.poisson_model import PoissonScoreModel
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class ScorePredictor:
    """Exact-score recommender backed by :class:`PoissonScoreModel`."""

    def __init__(self, poisson_model: PoissonScoreModel) -> None:
        """Initialize the recommender.

        Args:
            poisson_model: Fitted Poisson scoreline model.
        """
        self.poisson_model = poisson_model

    def recommend(self, team_a: str, team_b: str, k: int = 5) -> pd.DataFrame:
        """Return the top-*k* scorelines for a match.

        Args:
            team_a: First team.
            team_b: Second team.
            k: Number of scorelines to return.

        Returns:
            DataFrame with columns ``score_a``, ``score_b``, ``probability``.
        """
        if not self.poisson_model.is_fitted:
            logger.warning("Poisson model is not fitted; returning empty recommendations")
            return pd.DataFrame(columns=["score_a", "score_b", "probability"])
        top = self.poisson_model.top_k_scorelines(team_a, team_b, k=k)
        records = [
            {"score_a": score[0], "score_b": score[1], "probability": prob}
            for score, prob in top
        ]
        return pd.DataFrame(records)

    def recommend_for_fixtures(self, fixtures: pd.DataFrame, k: int = 5) -> pd.DataFrame:
        """Recommend scorelines for every fixture in a slate.

        Args:
            fixtures: DataFrame with ``team_a`` and ``team_b`` columns.
            k: Number of scorelines per match.

        Returns:
            DataFrame with one row per (match, scoreline).
        """
        if fixtures.empty:
            return pd.DataFrame(columns=["team_a", "team_b", "score_a", "score_b", "probability"])
        rows: list[dict[str, object]] = []
        for _, fixture in fixtures.iterrows():
            for _, rec in self.recommend(fixture["team_a"], fixture["team_b"], k=k).iterrows():
                rows.append(
                    {
                        "team_a": fixture["team_a"],
                        "team_b": fixture["team_b"],
                        "score_a": int(rec["score_a"]),
                        "score_b": int(rec["score_b"]),
                        "probability": float(rec["probability"]),
                    }
                )
        return pd.DataFrame(rows)

    def most_likely(self, team_a: str, team_b: str) -> Optional[tuple[int, int]]:
        """Return the mode of the scoreline distribution (or ``None`` if unfitted)."""
        if not self.poisson_model.is_fitted:
            return None
        score, _ = self.poisson_model.most_likely_scoreline(team_a, team_b)
        return score
