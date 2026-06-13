"""Observation schema and speed <-> congestion conversions.

The hidden state is per-edge **congestion** (normalized speed deficit in ``[0, 1]``; see
MODELING.md). Observations are measured **speeds** on a subset of edges. This module
defines the conversion between the two and the in-memory observation container the
estimator consumes.

Working the estimator in *congestion units* keeps the observation map a pure selection
matrix (linear), which is what makes the Phase-1 Kalman update exact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def speed_to_congestion(
    speed_kmh: np.ndarray, free_flow_speed_kmh: np.ndarray
) -> np.ndarray:
    """Convert measured speed to congestion ``1 - speed/free_flow``, clipped to ``[0, 1]``."""
    ratio = np.asarray(speed_kmh, dtype=float) / np.asarray(free_flow_speed_kmh, dtype=float)
    return np.clip(1.0 - ratio, 0.0, 1.0)


def congestion_to_speed(
    congestion: np.ndarray, free_flow_speed_kmh: np.ndarray
) -> np.ndarray:
    """Convert congestion to expected speed ``free_flow * (1 - congestion)``."""
    c = np.clip(np.asarray(congestion, dtype=float), 0.0, 1.0)
    return np.asarray(free_flow_speed_kmh, dtype=float) * (1.0 - c)


@dataclass(frozen=True)
class Observation:
    """A single time step of noisy, partial congestion observations.

    ``congestion`` holds values aligned to the canonical edge index; only entries where
    ``mask`` is ``True`` are valid measurements. Unobserved edges are never imputed here —
    the estimator infers them.
    """

    step: int
    congestion: np.ndarray  # (N,) values; valid only where mask is True
    mask: np.ndarray  # (N,) bool

    def validate(self, n_edges: int) -> None:
        if self.congestion.shape != (n_edges,):
            raise ValueError(f"congestion must have shape ({n_edges},)")
        if self.mask.shape != (n_edges,):
            raise ValueError(f"mask must have shape ({n_edges},)")
        if self.mask.dtype != np.bool_:
            raise ValueError("mask must be a boolean array")

    @property
    def n_observed(self) -> int:
        return int(self.mask.sum())
