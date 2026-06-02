"""Match-level outcome predictor that wires together all model layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.ensemble.blender import BlendWeights, ProbabilityBlender
from src.models.base_model import BaseOutcomeModel
from src.models.calibration import ProbabilityCalibrator
from src.models.poisson_model import PoissonScoreModel
from src.ratings.elo import EloRating
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class MatchPrediction:
    """Single-match prediction payload."""

    team_a: str
    team_b: str
    p_home: float
    p_draw: float
    p_away: float
    most_likely_outcome: str
    confidence: float

    def as_dict(self) -> dict[str, object]:
        return {
            "team_a": self.team_a,
            "team_b": self.team_b,
            "p_home": self.p_home,
            "p_draw": self.p_draw,
            "p_away": self.p_away,
            "most_likely_outcome": self.most_likely_outcome,
            "confidence": self.confidence,
        }


class MatchPredictor:
    """Top-level inference helper consuming the rating + ML + Poisson stack."""

    def __init__(
        self,
        outcome_models: Optional[dict[str, BaseOutcomeModel]] = None,
        poisson_model: Optional[PoissonScoreModel] = None,
        elo: Optional[EloRating] = None,
        calibrator: Optional[ProbabilityCalibrator] = None,
        blender: Optional[ProbabilityBlender] = None,
    ) -> None:
        """Initialize the predictor.

        Args:
            outcome_models: Optional dictionary of model name -> outcome model.
            poisson_model: Optional Poisson scoreline model.
            elo: Optional Elo rating system used as a ratings backbone.
            calibrator: Optional calibrator applied to the blended output.
            blender: Optional :class:`ProbabilityBlender`.
        """
        self.outcome_models = outcome_models or {}
        self.poisson_model = poisson_model
        self.elo = elo
        self.calibrator = calibrator
        self.blender = blender or ProbabilityBlender(weights=BlendWeights())

    def _ratings_probabilities(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        if self.elo is None:
            return None
        rows = np.zeros((len(X), 3))
        for i, (_, r) in enumerate(X[["team_a", "team_b"]].iterrows()):
            p_home, p_draw, p_away = self.elo.predict(r["team_a"], r["team_b"], neutral=True)
            rows[i] = [p_home, p_draw, p_away]
        return rows

    def _poisson_probabilities(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        if self.poisson_model is None or not self.poisson_model.is_fitted:
            return None
        return self.poisson_model.predict_proba(X[["team_a", "team_b"]])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities for an arbitrary set of matches.

        Args:
            X: Feature DataFrame with at least ``team_a`` and ``team_b`` columns.

        Returns:
            Probability matrix ``(n_samples, 3)`` summing to 1 row-wise.
        """
        mn = self.outcome_models.get("multinomial")
        xgb = self.outcome_models.get("xgboost")
        ratings_p = self._ratings_probabilities(X)
        poisson_p = self._poisson_probabilities(X)
        mn_p = mn.predict_proba(X) if mn is not None and mn.is_fitted else None
        xgb_p = xgb.predict_proba(X) if xgb is not None and xgb.is_fitted else None

        if not any(p is not None for p in (ratings_p, poisson_p, mn_p, xgb_p)):
            logger.warning("No fitted models available; returning uniform probabilities")
            return np.full((len(X), 3), 1 / 3)

        blended = self.blender.blend(
            ratings=ratings_p,
            multinomial=mn_p,
            xgboost=xgb_p,
            poisson=poisson_p,
        )
        if self.calibrator is not None and self.calibrator.is_fitted:
            blended = self.calibrator.transform(blended)
        return blended

    def predict_matches(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict outcomes for *X* and return a tidy DataFrame.

        Args:
            X: Feature DataFrame.

        Returns:
            DataFrame with ``team_a``, ``team_b``, ``p_home``, ``p_draw``,
            ``p_away``, ``most_likely_outcome`` and ``confidence``.
        """
        proba = self.predict_proba(X)
        out = X[["team_a", "team_b"]].copy().reset_index(drop=True)
        out["p_home"] = proba[:, 0]
        out["p_draw"] = proba[:, 1]
        out["p_away"] = proba[:, 2]
        outcomes = np.array(["H", "D", "A"])[proba.argmax(axis=1)]
        out["most_likely_outcome"] = outcomes
        out["confidence"] = proba.max(axis=1)
        return out

    def predict_single(self, team_a: str, team_b: str) -> MatchPrediction:
        """Convenience helper for a single fixture.

        Args:
            team_a: First team.
            team_b: Second team.

        Returns:
            :class:`MatchPrediction`.
        """
        X = pd.DataFrame({"team_a": [team_a], "team_b": [team_b]})
        result = self.predict_matches(X).iloc[0]
        return MatchPrediction(
            team_a=team_a,
            team_b=team_b,
            p_home=float(result["p_home"]),
            p_draw=float(result["p_draw"]),
            p_away=float(result["p_away"]),
            most_likely_outcome=str(result["most_likely_outcome"]),
            confidence=float(result["confidence"]),
        )
