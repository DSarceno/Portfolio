"""Poisson / Dixon-Coles-inspired scoreline model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.stats import poisson, skellam

from src.models.base_model import BaseOutcomeModel
from src.utils.constants import OUTCOME_TO_INDEX
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class PoissonFit:
    """Container holding the fitted Poisson parameters."""

    intercept: float
    home_advantage: float
    attack: dict[str, float]
    defense: dict[str, float]


class PoissonScoreModel(BaseOutcomeModel):
    """Goal-rate model with a Dixon-Coles low-score adjustment."""

    def __init__(
        self,
        max_goals: int = 8,
        rho: float = -0.10,
        hyperparameters: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize the model.

        Args:
            max_goals: Maximum number of goals to consider per side in the
                scoreline grid.
            rho: Dixon-Coles correlation parameter (used only for the exact
                scoreline grid when ``use_dixon_coles_adjustment`` is true).
            hyperparameters: Optional configuration dict. Recognized keys:
                ``max_goals``, ``rho``, ``use_dixon_coles_adjustment``,
                ``use_skellam`` (default True; if true, H/D/A probabilities are
                derived from the Skellam distribution which natively handles
                the goal-difference distribution and yields better-calibrated
                draws than the Dixon-Coles tau correction), ``time_decay_xi``
                (Dixon-Coles 1997 style exponential decay applied to historical
                matches at fit time; default 0 = no decay).
        """
        super().__init__(name="poisson", feature_columns=["team_a", "team_b"])
        hp = hyperparameters or {}
        self.max_goals = int(hp.get("max_goals", max_goals))
        self.rho = float(hp.get("rho", rho))
        self.use_dixon_coles = bool(hp.get("use_dixon_coles_adjustment", True))
        self.use_skellam = bool(hp.get("use_skellam", True))
        self.time_decay_xi = float(hp.get("time_decay_xi", 0.0))
        self.fit_result: Optional[PoissonFit] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PoissonScoreModel":
        """Estimate attack/defense strengths via weighted averaging.

        When ``self.time_decay_xi > 0`` and *X* has a ``date`` column, each
        match contributes a weight ``exp(-xi * days_since_latest)`` so that
        recent results dominate the attack/defense estimates (Dixon-Coles
        1997).

        Args:
            X: DataFrame with ``team_a``, ``team_b``, ``score_a``, ``score_b``
                and optional ``date`` columns.
            y: Outcome labels (unused — kept for API compatibility).

        Returns:
            ``self``.
        """
        keep_cols = ["team_a", "team_b", "score_a", "score_b"]
        if "date" in X.columns:
            keep_cols = keep_cols + ["date"]
        df = X[keep_cols].dropna(subset=["team_a", "team_b", "score_a", "score_b"]).copy()
        df["score_a"] = pd.to_numeric(df["score_a"], errors="coerce")
        df["score_b"] = pd.to_numeric(df["score_b"], errors="coerce")
        df = df.dropna(subset=["score_a", "score_b"])
        if df.empty:
            self.fit_result = PoissonFit(0.0, 0.20, {}, {})
            self.is_fitted = True
            return self

        if self.time_decay_xi > 0 and "date" in df.columns:
            dates = pd.to_datetime(df["date"], errors="coerce")
            ref_date = dates.max()
            days = (ref_date - dates).dt.days.clip(lower=0).fillna(0).astype(float)
            df["weight"] = np.exp(-self.time_decay_xi * days)
        else:
            df["weight"] = 1.0

        w = df["weight"]
        avg_goals = float(
            ((df["score_a"] * w).sum() + (df["score_b"] * w).sum()) / (2.0 * w.sum())
        )
        avg_goals = max(avg_goals, 0.3)

        gf = pd.concat(
            [
                df[["team_a", "score_a", "weight"]].rename(
                    columns={"team_a": "team", "score_a": "goals"}
                ),
                df[["team_b", "score_b", "weight"]].rename(
                    columns={"team_b": "team", "score_b": "goals"}
                ),
            ]
        )
        ga = pd.concat(
            [
                df[["team_a", "score_b", "weight"]].rename(
                    columns={"team_a": "team", "score_b": "goals"}
                ),
                df[["team_b", "score_a", "weight"]].rename(
                    columns={"team_b": "team", "score_a": "goals"}
                ),
            ]
        )

        def _weighted_mean(group: pd.DataFrame) -> float:
            ws = group["weight"].sum()
            if ws <= 0:
                return 1.0
            return float((group["goals"] * group["weight"]).sum() / ws)

        attack = {
            team: _weighted_mean(g) / avg_goals
            for team, g in gf.groupby("team")
        }
        defense = {
            team: _weighted_mean(g) / avg_goals
            for team, g in ga.groupby("team")
        }

        ha = float(
            ((df["score_a"] - df["score_b"]) * w).sum()
            / (w.sum() * max(avg_goals, 0.1))
        )

        self.fit_result = PoissonFit(
            intercept=float(np.log(avg_goals)),
            home_advantage=ha,
            attack=attack,
            defense=defense,
        )
        self.is_fitted = True
        logger.info(
            "Poisson model fitted on %d matches (xi=%.4f, skellam=%s)",
            len(df),
            self.time_decay_xi,
            self.use_skellam,
        )
        return self

    def expected_goals(self, team_a: str, team_b: str, neutral: bool = True) -> tuple[float, float]:
        """Compute expected goal rates for *team_a* and *team_b*.

        Args:
            team_a: First team.
            team_b: Second team.
            neutral: ``True`` if neutral venue.

        Returns:
            Tuple ``(lambda_a, lambda_b)``.
        """
        self._check_fitted()
        fit = self.fit_result
        attack_a = fit.attack.get(team_a, 1.0)
        defense_a = fit.defense.get(team_a, 1.0)
        attack_b = fit.attack.get(team_b, 1.0)
        defense_b = fit.defense.get(team_b, 1.0)
        base = float(np.exp(fit.intercept))
        ha = 0.0 if neutral else fit.home_advantage
        lam_a = base * attack_a * defense_b * np.exp(ha / 2)
        lam_b = base * attack_b * defense_a * np.exp(-ha / 2)
        return float(max(lam_a, 0.05)), float(max(lam_b, 0.05))

    def score_matrix(
        self, team_a: str, team_b: str, neutral: bool = True
    ) -> np.ndarray:
        """Return the joint scoreline probability matrix.

        Args:
            team_a: First team.
            team_b: Second team.
            neutral: ``True`` if neutral venue.

        Returns:
            ``(max_goals+1, max_goals+1)`` matrix that sums to ~1.
        """
        lam_a, lam_b = self.expected_goals(team_a, team_b, neutral=neutral)
        goals = np.arange(self.max_goals + 1)
        pa = poisson.pmf(goals, lam_a)
        pb = poisson.pmf(goals, lam_b)
        matrix = np.outer(pa, pb)
        if self.use_dixon_coles:
            matrix = self._apply_dixon_coles(matrix, lam_a, lam_b)
        total = matrix.sum()
        if total > 0:
            matrix = matrix / total
        return matrix

    def _apply_dixon_coles(
        self, matrix: np.ndarray, lam_a: float, lam_b: float
    ) -> np.ndarray:
        """Apply the Dixon-Coles low-score correlation adjustment."""
        rho = self.rho
        adj = matrix.copy()
        adj[0, 0] *= 1 - lam_a * lam_b * rho
        if matrix.shape[0] > 1 and matrix.shape[1] > 1:
            adj[0, 1] *= 1 + lam_a * rho
            adj[1, 0] *= 1 + lam_b * rho
            adj[1, 1] *= 1 - rho
        return np.clip(adj, 1e-12, None)

    def outcome_probabilities(
        self, team_a: str, team_b: str, neutral: bool = True
    ) -> np.ndarray:
        """Return per-outcome probabilities for the H/D/A market.

        Two engines are available, controlled by ``self.use_skellam``:

        * **Skellam** (default, Karlis & Ntzoufras 2009): models the goal
          difference ``D = G_a - G_b`` directly as a Skellam distribution with
          parameters ``(lambda_a, lambda_b)``. ``p_home = P(D > 0)``,
          ``p_draw = P(D = 0)``, ``p_away = P(D < 0)``. Better calibrated for
          draws than the Dixon-Coles tau correction on the score grid.
        * **Score-grid + Dixon-Coles** (legacy): integrates the scoreline
          matrix where ``matrix[i, j] = P(team_a scores i, team_b scores j)``;
          team_a wins for ``i > j`` (lower triangle), team_b for ``j > i``
          (upper triangle), draws on the diagonal.
        """
        if self.use_skellam:
            lam_a, lam_b = self.expected_goals(team_a, team_b, neutral=neutral)
            p_draw = float(skellam.pmf(0, lam_a, lam_b))
            p_home = float(1.0 - skellam.cdf(0, lam_a, lam_b))
            p_away = float(skellam.cdf(-1, lam_a, lam_b))
        else:
            matrix = self.score_matrix(team_a, team_b, neutral=neutral)
            p_home = float(np.tril(matrix, k=-1).sum())
            p_draw = float(np.trace(matrix))
            p_away = float(np.triu(matrix, k=1).sum())
        total = p_home + p_draw + p_away
        if total <= 0:
            return np.array([1 / 3, 1 / 3, 1 / 3])
        return np.array([p_home / total, p_draw / total, p_away / total])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Vectorized scoreline-derived outcome probabilities."""
        self._check_fitted()
        out = np.zeros((len(X), 3))
        neutral_col = X.get("neutral_venue", pd.Series([True] * len(X)))
        for i, (_, row) in enumerate(X[["team_a", "team_b"]].iterrows()):
            out[i] = self.outcome_probabilities(
                row["team_a"],
                row["team_b"],
                neutral=bool(neutral_col.iloc[i]) if hasattr(neutral_col, "iloc") else True,
            )
        return out

    def most_likely_scoreline(
        self, team_a: str, team_b: str, neutral: bool = True
    ) -> tuple[tuple[int, int], float]:
        """Return the mode of the scoreline distribution and its probability."""
        matrix = self.score_matrix(team_a, team_b, neutral=neutral)
        idx = np.unravel_index(np.argmax(matrix), matrix.shape)
        return (int(idx[0]), int(idx[1])), float(matrix[idx])

    def top_k_scorelines(
        self, team_a: str, team_b: str, k: int = 5, neutral: bool = True
    ) -> list[tuple[tuple[int, int], float]]:
        """Return the top-*k* most probable scorelines."""
        matrix = self.score_matrix(team_a, team_b, neutral=neutral)
        flat = matrix.flatten()
        top = flat.argsort()[::-1][:k]
        results: list[tuple[tuple[int, int], float]] = []
        for idx in top:
            i, j = divmod(int(idx), matrix.shape[1])
            results.append(((i, j), float(matrix[i, j])))
        return results
