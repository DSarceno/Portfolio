"""Elo rating implementation tailored to international football."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class EloConfig:
    """Configuration for :class:`EloRating`."""

    base: float = 1500.0
    k_default: float = 24.0
    k_world_cup: float = 60.0
    k_continental: float = 40.0
    k_qualifier: float = 30.0
    k_friendly: float = 18.0
    home_advantage: float = 65.0
    goal_diff_multiplier: float = 1.0


COMPETITION_K_MAP = {
    "WC": "k_world_cup",
    "EURO": "k_continental",
    "COPA": "k_continental",
    "ASIAN": "k_continental",
    "AFCON": "k_continental",
    "QUAL": "k_qualifier",
    "WCQ": "k_qualifier",
    "FRIENDLY": "k_friendly",
}


class EloRating:
    """Stateful Elo rating system with competition-aware K and goal-margin scaling."""

    def __init__(self, config: EloConfig | None = None) -> None:
        """Initialize the rater.

        Args:
            config: Optional :class:`EloConfig`. Defaults are used when ``None``.
        """
        self.config = config or EloConfig()
        self.ratings: dict[str, float] = defaultdict(lambda: self.config.base)

    def expected_score(self, rating_a: float, rating_b: float, home_advantage: float = 0.0) -> float:
        """Standard Elo expected score for *a* against *b*.

        Args:
            rating_a: Rating of team A.
            rating_b: Rating of team B.
            home_advantage: Elo points added to *a*'s rating.

        Returns:
            Expected score in ``[0, 1]``.
        """
        diff = (rating_a + home_advantage) - rating_b
        return 1.0 / (1.0 + 10 ** (-diff / 400.0))

    def k_for_competition(self, competition: str | None) -> float:
        """Return the K factor for a competition code.

        Args:
            competition: Competition tag (e.g. ``"WC"``, ``"FRIENDLY"``).

        Returns:
            K value.
        """
        if not competition:
            return self.config.k_default
        key = COMPETITION_K_MAP.get(str(competition).upper(), None)
        if key is None:
            return self.config.k_default
        return float(getattr(self.config, key))

    def goal_margin_multiplier(self, goal_diff: int) -> float:
        """Margin-of-victory multiplier used by 538-style Elo systems.

        Args:
            goal_diff: ``|score_a - score_b|``.

        Returns:
            Scaling factor.
        """
        g = max(int(abs(goal_diff)), 1)
        if g == 1:
            return 1.0
        if g == 2:
            return 1.5
        return ((11 + g) / 8.0) * self.config.goal_diff_multiplier

    def update_match(
        self,
        team_a: str,
        team_b: str,
        score_a: int,
        score_b: int,
        competition: str | None = None,
        neutral: bool = True,
    ) -> tuple[float, float]:
        """Update ratings for one match.

        Args:
            team_a: Home (or first) team name.
            team_b: Away (or second) team name.
            score_a: Goals scored by *team_a*.
            score_b: Goals scored by *team_b*.
            competition: Competition tag.
            neutral: ``True`` for neutral-venue matches (no home advantage).

        Returns:
            Tuple ``(new_rating_a, new_rating_b)``.
        """
        ra = self.ratings[team_a]
        rb = self.ratings[team_b]
        ha = 0.0 if neutral else self.config.home_advantage
        expected_a = self.expected_score(ra, rb, ha)

        if score_a > score_b:
            actual_a = 1.0
        elif score_a < score_b:
            actual_a = 0.0
        else:
            actual_a = 0.5

        k = self.k_for_competition(competition)
        margin = self.goal_margin_multiplier(int(score_a) - int(score_b))
        delta = k * margin * (actual_a - expected_a)

        self.ratings[team_a] = ra + delta
        self.ratings[team_b] = rb - delta
        return self.ratings[team_a], self.ratings[team_b]

    def fit(self, matches: pd.DataFrame) -> "EloRating":
        """Replay every match in chronological order.

        Args:
            matches: DataFrame with columns ``date``, ``team_a``, ``team_b``,
                ``score_a``, ``score_b``, optional ``competition`` and
                ``neutral_venue``.

        Returns:
            ``self`` for chaining.
        """
        if matches.empty:
            return self
        ordered = matches.sort_values("date").reset_index(drop=True)
        for _, row in ordered.iterrows():
            try:
                score_a = int(row["score_a"])
                score_b = int(row["score_b"])
            except (TypeError, ValueError):
                continue
            self.update_match(
                team_a=row["team_a"],
                team_b=row["team_b"],
                score_a=score_a,
                score_b=score_b,
                competition=row.get("competition"),
                neutral=bool(row.get("neutral_venue", True)),
            )
        logger.info("Elo fit over %d matches; %d teams rated", len(ordered), len(self.ratings))
        return self

    def snapshot(self) -> pd.DataFrame:
        """Return the current rating table sorted descending.

        Returns:
            DataFrame with columns ``team`` and ``elo``.
        """
        df = pd.DataFrame(
            sorted(self.ratings.items(), key=lambda kv: -kv[1]),
            columns=["team", "elo"],
        )
        return df

    def predict(
        self, team_a: str, team_b: str, neutral: bool = True
    ) -> tuple[float, float, float]:
        """Predict outcome probabilities from current Elo ratings.

        Args:
            team_a: First team name.
            team_b: Second team name.
            neutral: ``True`` for neutral venue.

        Returns:
            Tuple ``(p_home, p_draw, p_away)``.
        """
        ha = 0.0 if neutral else self.config.home_advantage
        expected_a = self.expected_score(self.ratings[team_a], self.ratings[team_b], ha)
        p_draw_base = 0.26
        excess = abs(expected_a - 0.5)
        p_draw = max(0.10, p_draw_base - 0.18 * excess)
        remainder = 1.0 - p_draw
        p_home = remainder * expected_a
        p_away = remainder * (1.0 - expected_a)
        return p_home, p_draw, p_away

    def to_series(self, teams: Iterable[str]) -> pd.Series:
        """Return Elo ratings for the given iterable of team names."""
        return pd.Series({t: self.ratings[t] for t in teams}, name="elo")

    def __getstate__(self) -> dict:
        """Return picklable state (defaultdict converted to plain dict)."""
        return {"config": self.config, "ratings": dict(self.ratings)}

    def __setstate__(self, state: dict) -> None:
        """Rebuild the defaultdict on unpickle."""
        self.config = state["config"]
        base = self.config.base
        self.ratings = defaultdict(lambda: base, state["ratings"])
