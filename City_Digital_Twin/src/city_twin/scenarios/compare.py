"""Scenario comparison utilities.

Compares a baseline ground-truth rollout against an intervention rollout under an
*identical* noise seed, so differences are attributable to the intervention alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from city_twin.observation.synthetic import SyntheticObservations
from city_twin.scenarios.interventions import Intervention


@dataclass
class ScenarioComparison:
    baseline: np.ndarray  # (T+1, N) baseline ground truth
    scenario: np.ndarray  # (T+1, N) intervention ground truth

    @property
    def delta(self) -> np.ndarray:
        """Per-step, per-edge congestion difference (scenario - baseline)."""
        return self.scenario - self.baseline

    def mean_delta_over_time(self) -> np.ndarray:
        """Network-average congestion difference per step. Shape ``(T+1,)``."""
        return self.delta.mean(axis=1)


def compare_intervention(
    synth: SyntheticObservations,
    intervention: Intervention,
    n_steps: int,
    *,
    x0: np.ndarray | None = None,
) -> ScenarioComparison:
    """Run baseline and intervention rollouts from the same seed and compare them."""
    seed = synth.seed
    baseline_gen = SyntheticObservations(
        network=synth.network,
        dynamics=synth.dynamics,
        noise_speed_kmh=synth.noise_speed_kmh,
        observed_fraction=synth.observed_fraction,
        process_noise=synth.process_noise,
        seed=seed,
    )
    scenario_gen = SyntheticObservations(
        network=synth.network,
        dynamics=synth.dynamics,
        noise_speed_kmh=synth.noise_speed_kmh,
        observed_fraction=synth.observed_fraction,
        process_noise=synth.process_noise,
        seed=seed,
    )
    baseline = baseline_gen.generate_ground_truth(n_steps, x0=x0, intervention=None)
    scenario = scenario_gen.generate_ground_truth(n_steps, x0=x0, intervention=intervention)
    return ScenarioComparison(baseline=baseline, scenario=scenario)
