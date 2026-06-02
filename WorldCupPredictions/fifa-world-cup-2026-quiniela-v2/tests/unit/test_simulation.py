"""Tests for the simulation layer."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.simulation.bracket_generator import build_round_of_32_bracket
from src.simulation.group_stage import simulate_group_stage
from src.simulation.knockout import simulate_knockout
from src.simulation.tournament_simulator import TournamentSimulator


def _uniform_predictor(_: str, __: str) -> np.ndarray:
    return np.array([0.4, 0.25, 0.35])


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
