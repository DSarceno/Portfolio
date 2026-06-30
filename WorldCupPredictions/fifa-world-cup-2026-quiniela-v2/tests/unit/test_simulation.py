"""Tests for the simulation layer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.simulation.bracket_generator import (
    BracketPair,
    build_round_of_32_bracket,
    load_known_round_of_32,
)
from src.simulation.group_stage import simulate_group_stage
from src.simulation.knockout import simulate_knockout
from src.simulation.tournament_simulator import TournamentSimulator


def _uniform_predictor(_: str, __: str) -> np.ndarray:
    return np.array([0.4, 0.25, 0.35])


def _full_bracket_csv(path: Path, *, hole: bool = False) -> Path:
    """Write a complete (or, with hole=True, incomplete) 32-team bracket CSV."""
    teams = [f"T{i:02d}" for i in range(32)]
    lines = ["match_id,stage,feeds_winner_into,feeds_loser_into,date,team_a,team_b,neutral_venue"]
    for i in range(16):
        a, b = teams[2 * i], teams[2 * i + 1]
        if hole and i == 15:
            a = b = ""
        lines.append(f"R32_{i + 1:02d},ROUND_OF_32,R16_{i // 2 + 1:02d},,2026-06-29,{a},{b},False")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_group_stage_creates_standings(synthetic_fixtures: pd.DataFrame) -> None:
    result = simulate_group_stage(synthetic_fixtures, _uniform_predictor)
    assert not result.standings.empty
    assert set(result.qualified_top_two)
    assert isinstance(result.best_thirds, list)


def test_bracket_generation_size(synthetic_fixtures: pd.DataFrame) -> None:
    result = simulate_group_stage(synthetic_fixtures, _uniform_predictor)
    pairs = build_round_of_32_bracket(result.standings, result.best_thirds)
    assert len(pairs) >= 1
    for pair in pairs:
        assert pair.team_a != pair.team_b


def test_knockout_returns_a_champion(synthetic_fixtures: pd.DataFrame) -> None:
    result = simulate_group_stage(synthetic_fixtures, _uniform_predictor)
    bracket = build_round_of_32_bracket(result.standings, result.best_thirds)
    ko = simulate_knockout(bracket, _uniform_predictor)
    assert ko.champion


def test_tournament_simulator_aggregates(synthetic_fixtures: pd.DataFrame) -> None:
    sim = TournamentSimulator(synthetic_fixtures, _uniform_predictor)
    summary = sim.run(n_runs=20)
    assert summary.n_runs == 20
    assert "qualification_prob" in summary.qualification_probs.columns
    assert "championship_prob" in summary.championship_probs.columns


def test_load_known_round_of_32_full(tmp_path: Path) -> None:
    pairs = load_known_round_of_32(_full_bracket_csv(tmp_path / "ko.csv"))
    assert len(pairs) == 16
    teams = {t for p in pairs for t in (p.team_a, p.team_b)}
    assert len(teams) == 32
    # match_id order is preserved as slot order.
    assert pairs[0].slot == 1 and pairs[0].team_a == "T00" and pairs[0].team_b == "T01"


def test_load_known_round_of_32_incomplete_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not filled in"):
        load_known_round_of_32(_full_bracket_csv(tmp_path / "ko.csv", hole=True))


def test_load_known_round_of_32_duplicate_team_raises(tmp_path: Path) -> None:
    csv = _full_bracket_csv(tmp_path / "ko.csv")
    text = csv.read_text(encoding="utf-8").replace(",T31,", ",T00,")  # duplicate T00
    csv.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="distinct teams"):
        load_known_round_of_32(csv)


def test_run_from_known_bracket_only_bracket_teams_win(tmp_path: Path) -> None:
    bracket = load_known_round_of_32(_full_bracket_csv(tmp_path / "ko.csv"))
    sim = TournamentSimulator(
        pd.DataFrame({"team_a": [p.team_a for p in bracket], "team_b": [p.team_b for p in bracket]}),
        _uniform_predictor,
        vectorized_group_stage=False,
    )
    summary = sim.run_from_known_bracket(bracket, n_runs=200)

    bracket_teams = {t for p in bracket for t in (p.team_a, p.team_b)}
    # Champions can only come from the 32 teams that are actually in the bracket.
    assert set(summary.championship_probs["team"]).issubset(bracket_teams)
    # Probabilities form a distribution over a single champion per run.
    assert abs(summary.championship_probs["championship_prob"].sum() - 1.0) < 1e-9
    # All 32 teams are (trivially) qualified.
    assert len(summary.qualification_probs) == 32
    assert (summary.qualification_probs["qualification_prob"] == 1.0).all()


def test_known_winners_force_decided_ties(tmp_path: Path) -> None:
    bracket = load_known_round_of_32(_full_bracket_csv(tmp_path / "ko.csv"))
    fixtures = pd.DataFrame(
        {"team_a": [p.team_a for p in bracket], "team_b": [p.team_b for p in bracket]}
    )
    # Force the loser of the first R32 tie (T01) to "win" it deterministically;
    # the favourite (uniform) should then never be able to win that tie, so the
    # forced team must reach at least R16 in every run.
    pair = frozenset({bracket[0].team_a, bracket[0].team_b})
    forced = bracket[0].team_b
    sim = TournamentSimulator(fixtures, _uniform_predictor, vectorized_group_stage=False)
    summary = sim.run_from_known_bracket(bracket, n_runs=100, known_winners={pair: forced})

    # The eliminated side can never be champion.
    eliminated = bracket[0].team_a if forced == bracket[0].team_b else bracket[0].team_b
    assert eliminated not in set(summary.championship_probs["team"])


def test_run_from_known_bracket_is_deterministic(tmp_path: Path) -> None:
    bracket = load_known_round_of_32(_full_bracket_csv(tmp_path / "ko.csv"))
    fixtures = pd.DataFrame(
        {"team_a": [p.team_a for p in bracket], "team_b": [p.team_b for p in bracket]}
    )
    a = TournamentSimulator(fixtures, _uniform_predictor, seed=7, vectorized_group_stage=False)
    b = TournamentSimulator(fixtures, _uniform_predictor, seed=7, vectorized_group_stage=False)
    sa = a.run_from_known_bracket(bracket, n_runs=100)
    sb = b.run_from_known_bracket(bracket, n_runs=100)
    pd.testing.assert_frame_equal(sa.championship_probs, sb.championship_probs)
