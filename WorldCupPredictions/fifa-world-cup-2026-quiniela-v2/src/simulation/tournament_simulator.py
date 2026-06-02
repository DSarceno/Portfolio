"""Full tournament Monte-Carlo simulator."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.simulation.bracket_generator import build_round_of_32_bracket
from src.simulation.group_stage import simulate_group_stage
from src.simulation.knockout import simulate_knockout
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class SimulationSummary:
    """Aggregate stats produced by :class:`TournamentSimulator`."""

    n_runs: int
    qualification_probs: pd.DataFrame
    round_reached_probs: pd.DataFrame
    championship_probs: pd.DataFrame
    bracket_paths: pd.DataFrame


class TournamentSimulator:
    """Repeatable tournament Monte-Carlo simulator."""

    def __init__(
        self,
        fixtures: pd.DataFrame,
        predict_fn: Callable[[str, str], np.ndarray],
        seed: int = 42,
    ) -> None:
        """Initialize the simulator.

        Args:
            fixtures: Group-stage fixtures DataFrame.
            predict_fn: Callable returning ``[p_home, p_draw, p_away]``.
            seed: Master seed for reproducibility.
        """
        self.fixtures = fixtures
        self.predict_fn = predict_fn
        self.seed = seed

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
