"""Interventions for scenario evaluation.

An intervention is an exogenous input ``u_t`` that perturbs the *ground-truth* world
(see MODELING.md). Each intervention exposes two hooks applied during a ground-truth
rollout:

* :meth:`exogenous_input` — an additive term applied before clipping (pushes congestion).
* :meth:`enforce` — a hard constraint applied after the step (e.g. a closed edge pinned
  to full congestion).

Interventions never touch estimator internals; they only act on the world the estimator
observes. A scenario run is compared against a baseline run with identical noise seeds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class Intervention:
    """Base no-op intervention (the baseline scenario)."""

    def exogenous_input(self, step: int, n_edges: int) -> np.ndarray:
        return np.zeros(n_edges)

    def enforce(self, x: np.ndarray, step: int) -> np.ndarray:
        return x


@dataclass
class RoadClosure(Intervention):
    """Close one or more edges: pin their congestion to ``1.0`` while active.

    The pinned congestion propagates to neighbours through the diffusion term, raising
    downstream congestion — the expected effect of a closure.
    """

    edge_ids: list[int]
    start: int = 0
    end: int = 10**9

    def enforce(self, x: np.ndarray, step: int) -> np.ndarray:
        if self.start <= step < self.end:
            x = x.copy()
            x[self.edge_ids] = 1.0
        return x


@dataclass
class Accident(Intervention):
    """A localized capacity drop: inject extra congestion on edges for a duration.

    After the accident clears, the dynamics' decay relaxes congestion back toward baseline.
    """

    edge_ids: list[int]
    severity: float = 0.3
    start: int = 0
    duration: int = 5

    def exogenous_input(self, step: int, n_edges: int) -> np.ndarray:
        u = np.zeros(n_edges)
        if self.start <= step < self.start + self.duration:
            u[self.edge_ids] += self.severity
        return u


@dataclass
class Rainfall(Intervention):
    """A network-wide effect: add congestion to every edge while it rains."""

    intensity: float = 0.05
    start: int = 0
    duration: int = 10
    edge_ids: list[int] | None = field(default=None)  # None => all edges

    def exogenous_input(self, step: int, n_edges: int) -> np.ndarray:
        u = np.zeros(n_edges)
        if self.start <= step < self.start + self.duration:
            if self.edge_ids is None:
                u[:] += self.intensity
            else:
                u[self.edge_ids] += self.intensity
        return u
