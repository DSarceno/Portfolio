"""Quiniela-oriented pick generation across multiple risk profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from src.ensemble.pick_optimizer import PickOptimizer
from src.prediction.score_predictor import ScorePredictor
from src.utils.constants import RISK_PROFILES
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class QuinielaPick:
    """Pick payload for a single match under a single profile."""

    team_a: str
    team_b: str
    profile: str
    pick: str
    confidence: float
    rationale: str
    recommended_scoreline: Optional[tuple[int, int]] = None


class QuinielaStrategy:
    """Generate per-profile quiniela picks from a calibrated probability table."""

    def __init__(
        self,
        optimizer: Optional[PickOptimizer] = None,
        score_predictor: Optional[ScorePredictor] = None,
    ) -> None:
        """Initialize the strategy.

        Args:
            optimizer: Optional pre-built :class:`PickOptimizer`.
            score_predictor: Optional :class:`ScorePredictor` for scorelines.
        """
        self.optimizer = optimizer or PickOptimizer()
        self.score_predictor = score_predictor

    def generate(
        self,
        probabilities: pd.DataFrame,
        profile: str = "balanced",
        include_scoreline: bool = True,
    ) -> pd.DataFrame:
        """Generate a single-profile pick sheet.

        Args:
            probabilities: DataFrame with ``team_a``, ``team_b``, ``p_home``,
                ``p_draw``, ``p_away`` (and optional strategy features).
            profile: Risk profile name.
            include_scoreline: When ``True`` append a recommended scoreline.

        Returns:
            DataFrame with one row per match and the picked outcome.
        """
        if profile not in RISK_PROFILES:
            raise ValueError(f"Unknown profile '{profile}'")

        picks = self.optimizer.pick_dataframe(probabilities, risk_profile=profile)
        out = probabilities[["team_a", "team_b", "p_home", "p_draw", "p_away"]].reset_index(drop=True)
        out["profile"] = profile
        out["pick"] = picks["pick"].values
        out["confidence"] = picks["confidence"].values
        out["rationale"] = picks["rationale"].values
        if include_scoreline and self.score_predictor is not None:
            out["scoreline"] = [
                self.score_predictor.most_likely(r["team_a"], r["team_b"]) for _, r in out.iterrows()
            ]
        return out

    def generate_all(
        self,
        probabilities: pd.DataFrame,
        include_scoreline: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """Generate every risk profile in a single pass.

        Args:
            probabilities: Probability DataFrame (see :meth:`generate`).
            include_scoreline: When ``True`` append a recommended scoreline.

        Returns:
            Mapping ``profile -> DataFrame``.
        """
        return {
            profile: self.generate(probabilities, profile=profile, include_scoreline=include_scoreline)
            for profile in RISK_PROFILES
        }
