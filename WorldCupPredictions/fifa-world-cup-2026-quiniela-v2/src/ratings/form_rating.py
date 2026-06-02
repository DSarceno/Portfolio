"""Rolling-form rating based on recent matches."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import pandas as pd

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class FormConfig:
    """Configuration for :class:`FormRating`."""

    window: int = 5
    win_value: float = 3.0
    draw_value: float = 1.0
    loss_value: float = 0.0
    goal_diff_weight: float = 0.25


class FormRating:
    """Rolling-form rating combining a results window and a goal-difference window."""

    def __init__(self, config: FormConfig | None = None) -> None:
        """Initialize the rater.

        Args:
            config: Optional :class:`FormConfig`. Defaults are used when ``None``.
        """
        self.config = config or FormConfig()
        self.results: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=self.config.window)
        )
        self.goal_diffs: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=self.config.window)
        )

    def update_match(self, team_a: str, team_b: str, score_a: int, score_b: int) -> None:
        """Record a match for both teams.

        Args:
            team_a: First team.
            team_b: Second team.
            score_a: Goals scored by *team_a*.
            score_b: Goals scored by *team_b*.
        """
        if score_a > score_b:
            self.results[team_a].append(self.config.win_value)
            self.results[team_b].append(self.config.loss_value)
        elif score_a < score_b:
            self.results[team_a].append(self.config.loss_value)
            self.results[team_b].append(self.config.win_value)
        else:
            self.results[team_a].append(self.config.draw_value)
            self.results[team_b].append(self.config.draw_value)
        diff = float(score_a) - float(score_b)
        self.goal_diffs[team_a].append(diff)
        self.goal_diffs[team_b].append(-diff)

    def fit(self, matches: pd.DataFrame) -> "FormRating":
        """Replay matches chronologically.

        Args:
            matches: DataFrame with ``date``, ``team_a``, ``team_b``, ``score_a``, ``score_b``.

        Returns:
            ``self`` for chaining.
        """
        if matches.empty:
            return self
        ordered = matches.sort_values("date")
        for _, row in ordered.iterrows():
            try:
                score_a = int(row["score_a"])
                score_b = int(row["score_b"])
            except (TypeError, ValueError):
                continue
            self.update_match(row["team_a"], row["team_b"], score_a, score_b)
        logger.info("Form rating fit over %d matches", len(ordered))
        return self

    def form_score(self, team: str) -> float:
        """Return the team's current rolling-form score.

        Args:
            team: Team name.

        Returns:
            Weighted points + goal-diff component.
        """
        points = sum(self.results[team]) if self.results[team] else 0.0
        gd = sum(self.goal_diffs[team]) if self.goal_diffs[team] else 0.0
        return points + self.config.goal_diff_weight * gd

    def points_last_n(self, team: str) -> float:
        """Return the raw sum of points across the rolling window."""
        return float(sum(self.results[team])) if self.results[team] else 0.0

    def goal_diff_last_n(self, team: str) -> float:
        """Return the rolling goal-difference sum."""
        return float(sum(self.goal_diffs[team])) if self.goal_diffs[team] else 0.0

    def snapshot(self) -> pd.DataFrame:
        """Return the rolling-form table sorted descending by ``form_score``."""
        teams = sorted(set(self.results.keys()))
        records = [
            {
                "team": t,
                "rolling_form_points_last_5": self.points_last_n(t),
                "rolling_goal_diff_last_5": self.goal_diff_last_n(t),
                "form_score": self.form_score(t),
            }
            for t in teams
        ]
        return pd.DataFrame(records).sort_values("form_score", ascending=False).reset_index(
            drop=True
        )
