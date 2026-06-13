"""Nonlinear simulation world model (Phase 4).

This is the *ground-truth world* used to study interventions — distinct from the
estimator's dynamics, so the state-estimation interfaces stay untouched (see
ARCHITECTURE.md / MODELING.md). On top of the Phase-1 decay + diffusion it adds two
physically-motivated nonlinearities:

* **Saturation:** congestion is self-reinforcing while it builds and saturates at
  capacity. The growth term ``g * x * (1 - x)`` is positive for ``0 < x < 1`` and vanishes
  at both the free-flow (``0``) and fully-saturated (``1``) limits, keeping the state in
  ``[0, 1]``.
* **Bottleneck back-pressure:** a saturated edge backs up its *upstream* neighbours. Using
  the directed downstream matrix ``D`` (``D[i, j] = 1`` if ``i`` feeds ``j``), the
  back-pressure on edge ``i`` is ``b * (D @ x) * (1 - x)`` — congestion of an edge's
  downstream segments pushes it up, scaled by its remaining capacity.

The model exposes :meth:`step` so it can drive the existing synthetic generator and scenario
comparison utilities (it satisfies ``StepModel``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from city_twin.ingestion.network import RoadNetwork


@dataclass
class WorldModel:
    """Nonlinear congestion-field model for the simulation world.

    ``decay`` and ``diffusion`` mirror the Phase-1 dynamics; ``growth`` controls
    self-reinforcing saturation; ``backpressure`` controls upstream bottleneck propagation.
    """

    network: RoadNetwork
    decay: float = 0.05
    diffusion: float = 0.02
    growth: float = 0.1
    backpressure: float = 0.1
    dt_minutes: float = 5.0
    _laplacian: np.ndarray = field(init=False, repr=False)
    _downstream: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if min(self.decay, self.diffusion, self.growth, self.backpressure) < 0:
            raise ValueError("model coefficients must be non-negative")
        self._laplacian = self.network.laplacian()
        self._downstream = self.network.downstream_matrix()

    @property
    def n_edges(self) -> int:
        return self.network.n_edges

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """Congestion rate of change before exogenous input (diagnostic / testing)."""
        relaxation = -self.decay * x
        diffusion = -self.diffusion * (self._laplacian @ x)
        saturation = self.growth * x * (1.0 - x)
        backpressure = self.backpressure * (self._downstream @ x) * (1.0 - x)
        return relaxation + diffusion + saturation + backpressure

    def step(self, x: np.ndarray, u: np.ndarray | None = None) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        nxt = x + self.derivative(x)
        if u is not None:
            nxt = nxt + np.asarray(u, dtype=float)
        return np.clip(nxt, 0.0, 1.0)

    def steps_for_minutes(self, minutes: float) -> int:
        return int(round(minutes / self.dt_minutes))
