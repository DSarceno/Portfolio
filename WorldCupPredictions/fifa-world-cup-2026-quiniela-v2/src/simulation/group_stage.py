"""Group-stage simulation with FIFA tiebreak ordering and best-third selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from src.utils.constants import POINTS_DRAW, POINTS_LOSS, POINTS_WIN
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_HOME_GOAL_RATE = 1.30
DEFAULT_AWAY_GOAL_RATE = 1.10


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
    """Simulate a full group stage (scalar reference implementation).

    Kept for backwards compatibility, the API, and unit tests. For Monte-Carlo
    workloads use :class:`VectorizedGroupStageEngine` instead — it is one to
    two orders of magnitude faster.

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
        ga_score = max(0, int(rng.poisson(DEFAULT_HOME_GOAL_RATE)))
        gb_score = max(0, int(rng.poisson(DEFAULT_AWAY_GOAL_RATE)))
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


class VectorizedGroupStageEngine:
    """Pre-computed, fully-vectorized group-stage simulator.

    Construct it once with the fixtures and a probability callable. Then call
    :meth:`simulate` repeatedly with different RNGs — the heavy work
    (probability lookups, integer indexing, sort keys) is done in numpy with
    no Python-level fixture loop.
    """

    def __init__(
        self,
        fixtures: pd.DataFrame,
        predict_fn: Callable[[str, str], np.ndarray],
        home_goal_rate: float = DEFAULT_HOME_GOAL_RATE,
        away_goal_rate: float = DEFAULT_AWAY_GOAL_RATE,
    ) -> None:
        """Initialize the engine.

        Args:
            fixtures: DataFrame with ``group``, ``team_a`` and ``team_b``.
            predict_fn: Callable returning ``[p_home, p_draw, p_away]``.
            home_goal_rate: Poisson rate for the first team's goals.
            away_goal_rate: Poisson rate for the second team's goals.
        """
        if "group" not in fixtures.columns:
            raise KeyError("fixtures must include a 'group' column")

        df = fixtures[["group", "team_a", "team_b"]].copy()
        df["team_a"] = df["team_a"].astype(str)
        df["team_b"] = df["team_b"].astype(str)
        df["group"] = df["group"].astype(str)
        df = df.dropna(subset=["team_a", "team_b", "group"]).reset_index(drop=True)

        teams = sorted(set(df["team_a"]).union(df["team_b"]))
        groups = sorted(df["group"].unique().tolist())
        self.team_to_idx: dict[str, int] = {t: i for i, t in enumerate(teams)}
        self.idx_to_team = np.array(teams, dtype=object)
        self.group_to_idx: dict[str, int] = {g: i for i, g in enumerate(groups)}
        self.idx_to_group = np.array(groups, dtype=object)
        self.n_teams = len(teams)
        self.n_groups = len(groups)
        self.n_fixtures = len(df)
        self.home_goal_rate = float(home_goal_rate)
        self.away_goal_rate = float(away_goal_rate)

        self.fixture_team_a = df["team_a"].map(self.team_to_idx).to_numpy(dtype=np.int64)
        self.fixture_team_b = df["team_b"].map(self.team_to_idx).to_numpy(dtype=np.int64)
        self.fixture_group = df["group"].map(self.group_to_idx).to_numpy(dtype=np.int64)

        self.team_group_idx = np.full(self.n_teams, -1, dtype=np.int64)
        self.team_group_idx[self.fixture_team_a] = self.fixture_group
        self.team_group_idx[self.fixture_team_b] = self.fixture_group

        probs = np.zeros((self.n_fixtures, 3), dtype=np.float64)
        for i in range(self.n_fixtures):
            a = teams[self.fixture_team_a[i]]
            b = teams[self.fixture_team_b[i]]
            row = np.asarray(predict_fn(a, b), dtype=np.float64)
            if row.shape != (3,) or not np.isfinite(row).all() or row.sum() <= 0:
                row = np.array([0.40, 0.25, 0.35])
            probs[i] = row / row.sum()
        self.fixture_probs = probs
        self.fixture_cum_probs = probs.cumsum(axis=1)

        per_group_indices: list[np.ndarray] = []
        for g_idx in range(self.n_groups):
            per_group_indices.append(np.where(self.team_group_idx == g_idx)[0])
        self._group_team_indices = per_group_indices

        logger.info(
            "VectorizedGroupStageEngine ready: %d fixtures, %d teams, %d groups",
            self.n_fixtures,
            self.n_teams,
            self.n_groups,
        )

    def simulate(self, rng: np.random.Generator) -> GroupStageResult:
        """Simulate the entire group stage in a single vectorized pass.

        Args:
            rng: Random number generator (used for outcomes, goals and
                tiebreak noise).

        Returns:
            :class:`GroupStageResult`.
        """
        n = self.n_fixtures

        u = rng.random(n)
        outcomes = (u[:, None] >= self.fixture_cum_probs[:, :2]).sum(axis=1).astype(np.int64)

        ga = rng.poisson(self.home_goal_rate, size=n).astype(np.int64)
        gb = rng.poisson(self.away_goal_rate, size=n).astype(np.int64)

        home_win = outcomes == 0
        away_win = outcomes == 2
        draw_mask = outcomes == 1

        ga = np.where(home_win & (gb >= ga), gb + 1, ga)
        gb = np.where(away_win & (ga >= gb), ga + 1, gb)
        gb = np.where(draw_mask, ga, gb)

        matches = np.zeros(self.n_teams, dtype=np.int64)
        wins = np.zeros(self.n_teams, dtype=np.int64)
        draws = np.zeros(self.n_teams, dtype=np.int64)
        losses = np.zeros(self.n_teams, dtype=np.int64)
        gf = np.zeros(self.n_teams, dtype=np.int64)
        ga_total = np.zeros(self.n_teams, dtype=np.int64)

        np.add.at(matches, self.fixture_team_a, 1)
        np.add.at(matches, self.fixture_team_b, 1)
        np.add.at(gf, self.fixture_team_a, ga)
        np.add.at(gf, self.fixture_team_b, gb)
        np.add.at(ga_total, self.fixture_team_a, gb)
        np.add.at(ga_total, self.fixture_team_b, ga)

        a_wins = ga > gb
        b_wins = ga < gb
        is_draw = ga == gb

        np.add.at(wins, self.fixture_team_a[a_wins], 1)
        np.add.at(losses, self.fixture_team_a[b_wins], 1)
        np.add.at(draws, self.fixture_team_a[is_draw], 1)
        np.add.at(wins, self.fixture_team_b[b_wins], 1)
        np.add.at(losses, self.fixture_team_b[a_wins], 1)
        np.add.at(draws, self.fixture_team_b[is_draw], 1)

        points = wins * POINTS_WIN + draws * POINTS_DRAW + losses * POINTS_LOSS
        goal_diff = gf - ga_total
        tiebreak_noise = rng.random(self.n_teams)

        standings_rows: list[dict] = []
        qualified_top_two: list[str] = []
        third_candidates_idx: list[int] = []

        for g_idx in range(self.n_groups):
            team_indices = self._group_team_indices[g_idx]
            if team_indices.size == 0:
                continue
            sort_pos = np.lexsort(
                (
                    tiebreak_noise[team_indices],
                    gf[team_indices],
                    goal_diff[team_indices],
                    points[team_indices],
                )
            )[::-1]
            ordered_idx = team_indices[sort_pos]
            for rank, team_idx in enumerate(ordered_idx, start=1):
                standings_rows.append(
                    {
                        "team": str(self.idx_to_team[team_idx]),
                        "group": str(self.idx_to_group[g_idx]),
                        "matches": int(matches[team_idx]),
                        "wins": int(wins[team_idx]),
                        "draws": int(draws[team_idx]),
                        "losses": int(losses[team_idx]),
                        "gf": int(gf[team_idx]),
                        "ga": int(ga_total[team_idx]),
                        "points": int(points[team_idx]),
                        "goal_diff": int(goal_diff[team_idx]),
                        "rank": rank,
                    }
                )
            for team_idx in ordered_idx[:2]:
                qualified_top_two.append(str(self.idx_to_team[team_idx]))
            if ordered_idx.size >= 3:
                third_candidates_idx.append(int(ordered_idx[2]))

        if third_candidates_idx:
            third_arr = np.asarray(third_candidates_idx, dtype=np.int64)
            third_sort = np.lexsort(
                (
                    tiebreak_noise[third_arr],
                    gf[third_arr],
                    goal_diff[third_arr],
                    points[third_arr],
                )
            )[::-1]
            third_ordered = third_arr[third_sort][:8]
            best_thirds = [str(self.idx_to_team[i]) for i in third_ordered]
        else:
            best_thirds = []

        log = [
            {
                "group": str(self.idx_to_group[self.fixture_group[i]]),
                "team_a": str(self.idx_to_team[self.fixture_team_a[i]]),
                "team_b": str(self.idx_to_team[self.fixture_team_b[i]]),
                "score_a": int(ga[i]),
                "score_b": int(gb[i]),
            }
            for i in range(n)
        ]

        return GroupStageResult(
            standings=pd.DataFrame(standings_rows),
            qualified_top_two=qualified_top_two,
            best_thirds=best_thirds,
            raw_match_log=log,
        )
