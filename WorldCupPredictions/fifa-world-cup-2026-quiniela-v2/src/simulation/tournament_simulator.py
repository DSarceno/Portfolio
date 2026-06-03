"""Full tournament Monte-Carlo simulator."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.simulation.bracket_generator import build_round_of_32_bracket
from src.simulation.group_stage import (
    VectorizedGroupStageEngine,
    simulate_group_stage,
)
from src.simulation.knockout import simulate_knockout
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

_UNIFORM_FALLBACK = np.array([0.40, 0.25, 0.35])


@dataclass
class SimulationSummary:
    """Aggregate stats produced by :class:`TournamentSimulator`."""

    n_runs: int
    qualification_probs: pd.DataFrame
    round_reached_probs: pd.DataFrame
    championship_probs: pd.DataFrame
    bracket_paths: pd.DataFrame


class TournamentSimulator:
    """Repeatable tournament Monte-Carlo simulator.

    By default the simulator pre-computes probabilities for every ordered pair
    of teams in the fixture set (O(n_teams^2) calls to ``predict_fn``), turning
    the per-match prediction inside the hot loop into a O(1) dict lookup. This
    typically yields a 100-1000x speedup over calling ``predict_fn`` for every
    match in every simulation.
    """

    def __init__(
        self,
        fixtures: pd.DataFrame,
        predict_fn: Callable[[str, str], np.ndarray],
        seed: int = 42,
        cache_predictions: bool = True,
        vectorized_group_stage: bool = True,
    ) -> None:
        """Initialize the simulator.

        Args:
            fixtures: Group-stage fixtures DataFrame.
            predict_fn: Callable returning ``[p_home, p_draw, p_away]``.
            seed: Master seed for reproducibility.
            cache_predictions: When ``True`` (default), build a lookup table of
                probabilities for every ordered pair of teams in the fixture
                set before running simulations.
            vectorized_group_stage: When ``True`` (default), use the
                :class:`VectorizedGroupStageEngine` for the group stage. Set
                to ``False`` to fall back to the scalar reference simulator.
        """
        self.fixtures = fixtures
        self.seed = seed
        self._raw_predict_fn = predict_fn
        self.predict_fn = (
            self._build_cached_predict_fn() if cache_predictions else predict_fn
        )
        self.group_stage_engine: VectorizedGroupStageEngine | None = (
            VectorizedGroupStageEngine(fixtures=fixtures, predict_fn=self.predict_fn)
            if vectorized_group_stage
            else None
        )

    def _build_cached_predict_fn(self) -> Callable[[str, str], np.ndarray]:
        """Pre-compute probabilities for every ordered pair of teams.

        Returns:
            A fast ``Callable[[str, str], np.ndarray]`` backed by a dict.
        """
        teams_raw = pd.concat([self.fixtures["team_a"], self.fixtures["team_b"]]).dropna().tolist()
        teams = sorted({str(t) for t in teams_raw})
        n = len(teams)
        n_pairs = n * (n - 1)
        logger.info(
            "Caching predictions: %d teams -> %d ordered pairs",
            n,
            n_pairs,
        )
        cache: dict[tuple[str, str], np.ndarray] = {}
        for a in tqdm(teams, desc="Caching predictions", total=n):
            for b in teams:
                if a == b:
                    continue
                try:
                    probs = np.asarray(self._raw_predict_fn(a, b), dtype=float)
                except (KeyError, ValueError) as exc:
                    logger.debug("predict_fn failed for (%s, %s): %s", a, b, exc)
                    probs = _UNIFORM_FALLBACK.copy()
                if probs.shape != (3,) or not np.isfinite(probs).all() or probs.sum() <= 0:
                    probs = _UNIFORM_FALLBACK.copy()
                else:
                    probs = probs / probs.sum()
                cache[(a, b)] = probs

        def cached_fn(a: str, b: str) -> np.ndarray:
            key = (str(a), str(b))
            return cache.get(key, _UNIFORM_FALLBACK)

        return cached_fn

    def run(self, n_runs: int = 10_000) -> SimulationSummary:
        """Run *n_runs* simulations and aggregate the statistics.

        Args:
            n_runs: Number of full tournament repetitions.

        Returns:
            :class:`SimulationSummary`.
        """
        qualifications: dict[str, int] = defaultdict(int)
        round_reached: dict[tuple[str, str], int] = defaultdict(int)
        championships: dict[str, int] = defaultdict(int)
        bracket_path_log: list[dict] = []

        for run_idx in tqdm(range(n_runs), desc="Simulating tournament"):
            rng = np.random.default_rng(self.seed + run_idx)
            if self.group_stage_engine is not None:
                gs = self.group_stage_engine.simulate(rng)
            else:
                gs = simulate_group_stage(self.fixtures, self.predict_fn, rng=rng)
            for team in gs.qualified_top_two:
                qualifications[team] += 1
            for team in gs.best_thirds:
                qualifications[team] += 1

            bracket = build_round_of_32_bracket(gs.standings, gs.best_thirds)
            ko = simulate_knockout(bracket, self.predict_fn, rng=rng)

            for team, stage in ko.rounds_reached.items():
                round_reached[(team, stage)] += 1
            championships[ko.champion] += 1
            bracket_path_log.append(
                {
                    "run": run_idx,
                    "champion": ko.champion,
                    "runner_up": ko.runner_up,
                    "third_place": ko.third_place,
                }
            )

        n = max(n_runs, 1)
        qualification_df = (
            pd.DataFrame(
                [
                    {"team": team, "qualification_prob": count / n}
                    for team, count in qualifications.items()
                ]
            )
            .sort_values("qualification_prob", ascending=False)
            .reset_index(drop=True)
        )

        round_df = pd.DataFrame(
            [
                {"team": team, "stage": stage, "prob": count / n}
                for (team, stage), count in round_reached.items()
            ]
        ).sort_values(["team", "stage"]).reset_index(drop=True)

        champ_df = (
            pd.DataFrame(
                [{"team": team, "championship_prob": count / n} for team, count in championships.items()]
            )
            .sort_values("championship_prob", ascending=False)
            .reset_index(drop=True)
        )

        return SimulationSummary(
            n_runs=n,
            qualification_probs=qualification_df,
            round_reached_probs=round_df,
            championship_probs=champ_df,
            bracket_paths=pd.DataFrame(bracket_path_log),
        )
