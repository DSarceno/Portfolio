"""Synthetic observation generator (Phase-1 data strategy).

Generates a known ground-truth congestion trajectory on the real network using the same
graph-diffusion dynamics, then emits noisy, partial **speed** observations from it. Because
the ground truth is known, estimation and forecast error can be measured directly — the
whole point of the synthetic-first strategy (see DATASETS.md).

The ground truth is the "real world": it includes process noise and is clipped to
``[0, 1]``, and any intervention is applied here, not to the estimator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from city_twin.dynamics.graph_diffusion import GraphDiffusionDynamics
from city_twin.ingestion.network import RoadNetwork
from city_twin.observation.schema import Observation, congestion_to_speed, speed_to_congestion
from city_twin.scenarios.interventions import Intervention


@dataclass
class SyntheticObservations:
    """Produce ground-truth trajectories and noisy partial observations.

    ``noise_speed_kmh`` is the measurement noise std in speed units. ``observed_fraction``
    is the expected fraction of edges measured at each step. ``process_noise`` is the
    ground-truth process-noise std in congestion units.
    """

    network: RoadNetwork
    dynamics: GraphDiffusionDynamics
    noise_speed_kmh: float = 4.0
    observed_fraction: float = 0.6
    process_noise: float = 0.01
    seed: int = 0

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def measurement_var(self) -> np.ndarray:
        """Per-edge measurement variance in congestion units: ``(sigma_v / v_free)^2``."""
        return (self.noise_speed_kmh / self.network.free_flow_speed_kmh) ** 2

    def generate_ground_truth(
        self,
        n_steps: int,
        *,
        x0: np.ndarray | None = None,
        intervention: Intervention | None = None,
    ) -> np.ndarray:
        """Roll the world forward ``n_steps`` steps. Returns ``(n_steps + 1, N)``."""
        n = self.network.n_edges
        if x0 is None:
            x = np.full(n, 0.1)
        else:
            x = np.clip(np.asarray(x0, dtype=float), 0.0, 1.0)

        traj = np.empty((n_steps + 1, n))
        traj[0] = x
        for t in range(n_steps):
            u = np.zeros(n)
            if intervention is not None:
                u = u + intervention.exogenous_input(t, n)
            noise = self._rng.normal(0.0, self.process_noise, size=n)
            x = self.dynamics.step(x, u=u + noise)
            if intervention is not None:
                x = intervention.enforce(x, t)
            x = np.clip(x, 0.0, 1.0)
            traj[t + 1] = x
        return traj

    def observe(self, ground_truth: np.ndarray) -> list[Observation]:
        """Emit noisy, partial observations for steps ``1 .. T`` of a ground-truth trajectory.

        Step 0 is the (known) initial condition and is not observed; observations begin at
        step 1, aligning with :meth:`KalmanFilter.filter`.
        """
        free_flow = self.network.free_flow_speed_kmh
        observations: list[Observation] = []
        for step in range(1, ground_truth.shape[0]):
            true_speed = congestion_to_speed(ground_truth[step], free_flow)
            noisy_speed = true_speed + self._rng.normal(
                0.0, self.noise_speed_kmh, size=self.network.n_edges
            )
            congestion = speed_to_congestion(noisy_speed, free_flow)
            mask = self._rng.random(self.network.n_edges) < self.observed_fraction
            observations.append(Observation(step=step, congestion=congestion, mask=mask))
        return observations
