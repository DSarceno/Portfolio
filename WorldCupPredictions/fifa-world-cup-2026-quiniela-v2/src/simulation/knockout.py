"""Knockout-stage simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from src.simulation.bracket_generator import BracketPair
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class KnockoutLogEntry:
    """One simulated knockout match."""

    stage: str
    team_a: str
    team_b: str
    winner: str
    via_penalties: bool


@dataclass
class KnockoutResult:
    """Knockout-stage outcome."""

    champion: str
    runner_up: str
    third_place: str
    log: list[KnockoutLogEntry] = field(default_factory=list)
    rounds_reached: dict[str, str] = field(default_factory=dict)


def _decide_match(
    team_a: str,
    team_b: str,
    predict_fn: Callable[[str, str], np.ndarray],
    rng: np.random.Generator,
    stage: str,
) -> KnockoutLogEntry:
    """Simulate one knockout match (with penalty proxy if needed).

    Args:
        team_a: First team.
        team_b: Second team.
        predict_fn: Callable returning ``[p_home, p_draw, p_away]``.
        rng: Random generator.
        stage: Stage label (for logging).

    Returns:
        :class:`KnockoutLogEntry`.
    """
    probs = predict_fn(team_a, team_b)
    probs = probs / probs.sum()
    p_home, p_draw, p_away = probs.tolist()
    win_prob = p_home + 0.5 * p_draw
    via_penalties = rng.random() < p_draw
    winner = team_a if rng.random() < win_prob else team_b
    return KnockoutLogEntry(
        stage=stage,
        team_a=team_a,
        team_b=team_b,
        winner=winner,
        via_penalties=via_penalties,
    )


def simulate_knockout(
    bracket: list[BracketPair],
    predict_fn: Callable[[str, str], np.ndarray],
    rng: np.random.Generator | None = None,
) -> KnockoutResult:
    """Run the full knockout stage.

    Args:
        bracket: Round-of-32 pairings.
        predict_fn: Callable returning ``[p_home, p_draw, p_away]``.
        rng: Optional random generator.

    Returns:
        :class:`KnockoutResult`.
    """
    rng = rng or np.random.default_rng()
    log: list[KnockoutLogEntry] = []
    rounds_reached: dict[str, str] = {}

    def _record_round(teams: list[str], label: str) -> None:
        for t in teams:
            rounds_reached[t] = label

    teams = []
    for pair in bracket:
        teams.extend([pair.team_a, pair.team_b])
    _record_round(teams, "r32")

    stages = ["r32", "r16", "qf", "sf"]
    current = teams
    semifinal_losers: list[str] = []
    semifinal_winners: list[str] = []

    for stage in stages:
        if len(current) < 2:
            break
        if len(current) % 2 == 1:
            logger.warning(
                "Odd number of teams (%d) entering %s; advancing the last team without playing",
                len(current),
                stage,
            )
            bye = current.pop()
        else:
            bye = None
        next_round: list[str] = []
        round_pairs = [(current[i], current[i + 1]) for i in range(0, len(current), 2)]
        for a, b in round_pairs:
            entry = _decide_match(a, b, predict_fn, rng, stage)
            log.append(entry)
            next_round.append(entry.winner)
            if stage == "sf":
                semifinal_winners.append(entry.winner)
                loser = a if entry.winner == b else b
                semifinal_losers.append(loser)
        if bye is not None:
            next_round.append(bye)
        next_label = {"r32": "r16", "r16": "qf", "qf": "sf", "sf": "final"}[stage]
        _record_round(next_round, next_label)
        current = next_round

    runner_up = ""
    champion = current[0] if current else ""
    if len(current) >= 2:
        final_entry = _decide_match(current[0], current[1], predict_fn, rng, "final")
        log.append(final_entry)
        champion = final_entry.winner
        runner_up = current[1] if final_entry.winner == current[0] else current[0]
        _record_round([champion], "champion")
        _record_round([runner_up], "final")

    third_place = ""
    if len(semifinal_losers) >= 2:
        third_entry = _decide_match(
            semifinal_losers[0], semifinal_losers[1], predict_fn, rng, "third_place"
        )
        log.append(third_entry)
        third_place = third_entry.winner

    return KnockoutResult(
        champion=champion,
        runner_up=runner_up,
        third_place=third_place,
        log=log,
        rounds_reached=rounds_reached,
    )
