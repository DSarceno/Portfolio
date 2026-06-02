"""Group-stage simulation with FIFA tiebreak ordering and best-third selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from src.utils.constants import POINTS_DRAW, POINTS_LOSS, POINTS_WIN
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class TeamGroupRow:
    """Per-team standings row inside a group."""

    team: str
    group: str
    matches: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    gf: int = 0
    ga: int = 0

    @property
    def points(self) -> int:
        return self.wins * POINTS_WIN + self.draws * POINTS_DRAW + self.losses * POINTS_LOSS

    @property
    def goal_diff(self) -> int:
        return self.gf - self.ga

    def to_dict(self) -> dict[str, int | str]:
        return {
            "team": self.team,
            "group": self.group,
            "matches": self.matches,
            "wins": self.wins,
            "draws": self.draws,
            "losses": self.losses,
            "gf": self.gf,
            "ga": self.ga,
            "points": self.points,
            "goal_diff": self.goal_diff,
        }


@dataclass
class GroupStageResult:
    """Outcome of a single group-stage simulation."""

    standings: pd.DataFrame
    qualified_top_two: list[str]
    best_thirds: list[str]
    raw_match_log: list[dict] = field(default_factory=list)


def _ordered_standings(group: list[TeamGroupRow]) -> list[TeamGroupRow]:
    """Sort *group* by the standard FIFA tiebreaks (points, GD, GF, random)."""
    rng = np.random.default_rng()
    return sorted(
        group,
        key=lambda r: (
            -r.points,
            -r.goal_diff,
            -r.gf,
            rng.random(),
        ),
    )


def simulate_group_stage(
    fixtures: pd.DataFrame,
    predict_fn: Callable[[str, str], np.ndarray],
    rng: np.random.Generator | None = None,
) -> GroupStageResult:
    """Simulate a full group stage.

    Args:
        fixtures: DataFrame with ``group``, ``team_a`` and ``team_b`` columns
            for every group-stage fixture.
        predict_fn: Callable returning ``[p_home, p_draw, p_away]`` for a pair.
        rng: Optional random generator (for reproducibility).

    Returns:
        :class:`GroupStageResult` with the standings, the top-two list, the
        best-third list, and the per-match log.
    """
    rng = rng or np.random.default_rng()
    groups: dict[str, dict[str, TeamGroupRow]] = {}
    log: list[dict] = []

    for _, row in fixtures.iterrows():
        group = str(row["group"])
        team_a, team_b = row["team_a"], row["team_b"]
        groups.setdefault(group, {})
        for team in (team_a, team_b):
            groups[group].setdefault(team, TeamGroupRow(team=team, group=group))

        probs = predict_fn(team_a, team_b)
        idx = int(rng.choice(3, p=probs / probs.sum()))
        ga_score = max(0, int(rng.poisson(1.2)))
        gb_score = max(0, int(rng.poisson(1.0)))
        if idx == 0 and gb_score >= ga_score:
            ga_score = gb_score + 1
        elif idx == 2 and ga_score >= gb_score:
            gb_score = ga_score + 1
        elif idx == 1:
            gb_score = ga_score

        a = groups[group][team_a]
        b = groups[group][team_b]
        a.matches += 1
        b.matches += 1
        a.gf += ga_score
        a.ga += gb_score
        b.gf += gb_score
        b.ga += ga_score
        if ga_score > gb_score:
            a.wins += 1
            b.losses += 1
        elif ga_score == gb_score:
            a.draws += 1
            b.draws += 1
        else:
            a.losses += 1
            b.wins += 1
        log.append(
            {
                "group": group,
                "team_a": team_a,
                "team_b": team_b,
                "score_a": ga_score,
                "score_b": gb_score,
            }
        )

    standings_rows: list[dict] = []
    qualified_top_two: list[str] = []
    third_candidates: list[TeamGroupRow] = []
    for group, teams in groups.items():
        ordered = _ordered_standings(list(teams.values()))
        for rank, team in enumerate(ordered, start=1):
            d = team.to_dict()
            d["rank"] = rank
            standings_rows.append(d)
        qualified_top_two.extend([t.team for t in ordered[:2]])
        if len(ordered) >= 3:
            third_candidates.append(ordered[2])

    third_sorted = sorted(
        third_candidates,
        key=lambda r: (-r.points, -r.goal_diff, -r.gf, rng.random()),
    )
    best_thirds = [t.team for t in third_sorted[:8]]

    return GroupStageResult(
        standings=pd.DataFrame(standings_rows),
        qualified_top_two=qualified_top_two,
        best_thirds=best_thirds,
        raw_match_log=log,
    )
