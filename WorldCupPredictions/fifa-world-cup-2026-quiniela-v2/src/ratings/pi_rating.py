"""Pi-Rating implementation (Constantinou & Fenton, 2013)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class PIConfig:
    """Configuration for :class:`PIRating`."""

    lambda_factor: float = 0.054
    gamma_factor: float = 0.79
    base: float = 0.0
    log_scale: float = 3.0
    time_decay_xi: float = 0.0


class PIRating:
    """Pi-Rating: maintains separate home and away ratings per team."""

    def __init__(self, config: PIConfig | None = None) -> None:
        """Initialize the rater.

        Args:
            config: Optional :class:`PIConfig`. Defaults are used when ``None``.
        """
        self.config = config or PIConfig()
        self.home_rating: dict[str, float] = defaultdict(lambda: self.config.base)
        self.away_rating: dict[str, float] = defaultdict(lambda: self.config.base)

    @staticmethod
    def _expected_goals(diff: float, log_scale: float) -> float:
        """Convert a rating difference into an expected goal difference (signed)."""
        return (10 ** (abs(diff) / log_scale) - 1) * (1 if diff >= 0 else -1)

    def update_match(
        self,
        team_a: str,
        team_b: str,
        score_a: int,
        score_b: int,
        weight: float = 1.0,
    ) -> None:
        """Update ratings for one match.

        Args:
            team_a: First team.
            team_b: Second team.
            score_a: Goals scored by *team_a*.
            score_b: Goals scored by *team_b*.
            weight: Multiplier on the learning step (``< 1`` for old matches).
        """
        cfg = self.config
        rha = self.home_rating[team_a]
        raway = self.away_rating[team_b]
        expected_diff = self._expected_goals(rha - raway, cfg.log_scale)
        observed_diff = float(score_a) - float(score_b)
        error = weight * (observed_diff - expected_diff)

        self.home_rating[team_a] = rha + cfg.lambda_factor * error
        self.away_rating[team_a] = (
            self.away_rating[team_a] + cfg.gamma_factor * cfg.lambda_factor * error
        )
        self.away_rating[team_b] = raway - cfg.lambda_factor * error
        self.home_rating[team_b] = (
            self.home_rating[team_b] - cfg.gamma_factor * cfg.lambda_factor * error
        )

    def fit(self, matches: pd.DataFrame) -> "PIRating":
        """Replay matches in chronological order.

        Args:
            matches: DataFrame with ``date``, ``team_a``, ``team_b``, ``score_a``, ``score_b``.

        Returns:
            ``self`` for chaining.
        """
        if matches.empty:
            return self
        ordered = matches.sort_values("date").reset_index(drop=True)
        xi = float(self.config.time_decay_xi)
        ref_date: pd.Timestamp | None = None
        if xi > 0:
            dates = pd.to_datetime(ordered["date"], errors="coerce")
            ref_date = dates.max()
        for _, row in ordered.iterrows():
            try:
                score_a = int(row["score_a"])
                score_b = int(row["score_b"])
            except (TypeError, ValueError):
                continue
            weight = 1.0
            if xi > 0 and ref_date is not None:
                match_date = pd.to_datetime(row["date"], errors="coerce")
                if pd.notna(match_date):
                    days = max(float((ref_date - match_date).days), 0.0)
                    weight = float(np.exp(-xi * days))
            self.update_match(
                row["team_a"], row["team_b"], score_a, score_b, weight=weight
            )
        logger.info("PI Rating fit over %d matches (xi=%.4f)", len(ordered), xi)
        return self

    def combined(self, team: str) -> float:
        """Return the average of home and away ratings for *team*."""
        return 0.5 * (self.home_rating[team] + self.away_rating[team])

    def snapshot(self) -> pd.DataFrame:
        """Return the rating table sorted descending by the combined rating."""
        teams = sorted(set(self.home_rating.keys()) | set(self.away_rating.keys()))
        records = [
            {
                "team": t,
                "pi_home": self.home_rating[t],
                "pi_away": self.away_rating[t],
                "pi_combined": self.combined(t),
            }
            for t in teams
        ]
        return pd.DataFrame(records).sort_values("pi_combined", ascending=False).reset_index(
            drop=True
        )

    def __getstate__(self) -> dict:
        """Return picklable state (defaultdicts converted to plain dicts)."""
        return {
            "config": self.config,
            "home_rating": dict(self.home_rating),
            "away_rating": dict(self.away_rating),
        }

    def __setstate__(self, state: dict) -> None:
        """Rebuild the defaultdicts on unpickle."""
        self.config = state["config"]
        base = self.config.base
        self.home_rating = defaultdict(lambda: base, state["home_rating"])
        self.away_rating = defaultdict(lambda: base, state["away_rating"])
