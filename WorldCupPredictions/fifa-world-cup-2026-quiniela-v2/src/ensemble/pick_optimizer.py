"""Risk-profile-based pick optimization for quinielas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.constants import (
    OUTCOME_AWAY_WIN,
    OUTCOME_DRAW,
    OUTCOME_HOME_WIN,
    RISK_PROFILES,
)
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class RiskProfileConfig:
    """Tunable parameters of a single risk profile."""

    favorite_threshold: float
    upset_tolerance: float
    draw_bias: float


DEFAULT_PROFILES: dict[str, RiskProfileConfig] = {
    "safe": RiskProfileConfig(favorite_threshold=0.58, upset_tolerance=0.05, draw_bias=0.90),
    "balanced": RiskProfileConfig(favorite_threshold=0.50, upset_tolerance=0.10, draw_bias=1.00),
    "aggressive": RiskProfileConfig(favorite_threshold=0.43, upset_tolerance=0.18, draw_bias=1.10),
    "contrarian": RiskProfileConfig(
        favorite_threshold=0.38, upset_tolerance=0.25, draw_bias=1.15
    ),
}


class PickOptimizer:
    """Translate calibrated probabilities into risk-profile-aware picks."""

    def __init__(
        self,
        profiles: Optional[dict[str, RiskProfileConfig]] = None,
    ) -> None:
        """Initialize the optimizer.

        Args:
            profiles: Optional override of the default profile dictionary.
        """
        self.profiles = profiles or DEFAULT_PROFILES

    def pick(
        self,
        proba: np.ndarray,
        risk_profile: str = "balanced",
        upset_window_score: float = 0.0,
        favorite_fragility_score: float = 0.0,
        public_bias_proxy: float = 0.0,
    ) -> tuple[str, float, str]:
        """Pick an outcome under a risk profile.

        Args:
            proba: ``[p_home, p_draw, p_away]``.
            risk_profile: One of :data:`RISK_PROFILES`.
            upset_window_score: Upset-window feature ``[0, 1]``.
            favorite_fragility_score: Favorite-fragility feature ``[0, 1]``.
            public_bias_proxy: Reputation/public-bias proxy (sign indicates the
                team biased upward).

        Returns:
            Tuple ``(pick_label, confidence, rationale)``.
        """
        if risk_profile not in self.profiles:
            raise ValueError(
                f"Unknown profile '{risk_profile}'. Available: {list(self.profiles)}"
            )
        profile = self.profiles[risk_profile]

        p = np.asarray(proba, dtype=float).copy()
        p[1] *= profile.draw_bias
        p = p / p.sum()

        outcomes = [OUTCOME_HOME_WIN, OUTCOME_DRAW, OUTCOME_AWAY_WIN]
        best_idx = int(np.argmax(p))
        favored = p[best_idx]
        rationale_parts = [f"profile={risk_profile}", f"p={p.round(3).tolist()}"]

        if risk_profile == "safe":
            label = outcomes[best_idx]
            return label, float(favored), ", ".join(rationale_parts + ["safe favourite"])

        if risk_profile == "balanced":
            label = outcomes[best_idx]
            if favored < profile.favorite_threshold and p[1] > 0.27:
                label = OUTCOME_DRAW
                rationale_parts.append("balanced -> draw fallback")
            return label, float(favored if label != OUTCOME_DRAW else p[1]), ", ".join(rationale_parts)

        if risk_profile == "aggressive":
            underdog_idx = 0 if best_idx == 2 else 2
            if (
                upset_window_score >= 0.5
                and favorite_fragility_score >= 0.5
                and p[underdog_idx] >= profile.upset_tolerance
            ):
                rationale_parts.append("aggressive -> upset trigger")
                return outcomes[underdog_idx], float(p[underdog_idx]), ", ".join(rationale_parts)
            if p[1] >= 0.27 and abs(p[0] - p[2]) < 0.05:
                rationale_parts.append("aggressive -> draw value")
                return OUTCOME_DRAW, float(p[1]), ", ".join(rationale_parts)
            return outcomes[best_idx], float(favored), ", ".join(rationale_parts)

        # contrarian
        underdog_idx = 0 if best_idx == 2 else 2
        bias_sign = 1.0 if public_bias_proxy >= 0 else -1.0
        fade_idx = 2 if bias_sign > 0 else 0
        rationale_parts.append("contrarian -> fade public bias")
        if p[fade_idx] >= profile.upset_tolerance:
            return outcomes[fade_idx], float(p[fade_idx]), ", ".join(rationale_parts)
        return outcomes[underdog_idx], float(p[underdog_idx]), ", ".join(rationale_parts)

    def all_profiles(
        self,
        proba: np.ndarray,
        upset_window_score: float = 0.0,
        favorite_fragility_score: float = 0.0,
        public_bias_proxy: float = 0.0,
    ) -> dict[str, dict]:
        """Run :meth:`pick` for every profile in :data:`RISK_PROFILES`."""
        return {
            profile: {
                "pick": (
                    res := self.pick(
                        proba,
                        risk_profile=profile,
                        upset_window_score=upset_window_score,
                        favorite_fragility_score=favorite_fragility_score,
                        public_bias_proxy=public_bias_proxy,
                    )
                )[0],
                "confidence": res[1],
                "rationale": res[2],
            }
            for profile in RISK_PROFILES
        }

    def pick_dataframe(
        self,
        probs: pd.DataFrame,
        risk_profile: str = "balanced",
    ) -> pd.DataFrame:
        """Vectorized application of :meth:`pick` over a probability table.

        Args:
            probs: DataFrame containing columns ``p_home``, ``p_draw``,
                ``p_away`` and optional strategy features.
            risk_profile: Profile name.

        Returns:
            DataFrame with ``pick``, ``confidence`` and ``rationale`` columns.
        """
        records: list[dict[str, object]] = []
        for _, row in probs.iterrows():
            label, conf, rationale = self.pick(
                proba=np.array([row["p_home"], row["p_draw"], row["p_away"]]),
                risk_profile=risk_profile,
                upset_window_score=float(row.get("upset_window_score", 0.0)),
                favorite_fragility_score=float(row.get("favorite_fragility_score", 0.0)),
                public_bias_proxy=float(row.get("public_bias_proxy", 0.0)),
            )
            records.append({"pick": label, "confidence": conf, "rationale": rationale})
        return pd.DataFrame(records, index=probs.index)
