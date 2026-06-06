"""Reusable, Manim-independent helpers for animation scenes.

The functions here are pure NumPy and contain no Manim imports, so they can be
unit-tested without Manim installed and reused by both the framework's scene
base classes and by generated standalone scenes.
"""

from __future__ import annotations

import numpy as np

#: Whether Manim Community can be imported in the current environment.
try:  # pragma: no cover - environment dependent
    import manim  # noqa: F401

    MANIM_AVAILABLE = True
except Exception:  # noqa: BLE001 - any import failure means "not available"
    MANIM_AVAILABLE = False


def sample_trajectory(
    trajectory: np.ndarray,
    *,
    max_points: int = 1500,
) -> np.ndarray:
    """Downsample a trajectory to at most ``max_points`` rows.

    Animations rarely need every integration step; this keeps render time and
    memory bounded while preserving the shape of the curve.

    Parameters
    ----------
    trajectory:
        Array of shape ``(n_steps, n_dim)``.
    max_points:
        Maximum number of rows to keep.

    Returns
    -------
    numpy.ndarray
        The (possibly) downsampled trajectory.
    """
    traj = np.asarray(trajectory, dtype=float)
    if traj.ndim == 1:
        traj = traj[:, None]
    n = traj.shape[0]
    if n <= max_points:
        return traj
    idx = np.linspace(0, n - 1, max_points).astype(int)
    return traj[idx]


def fit_points_to_frame(
    points: np.ndarray,
    *,
    target_radius: float = 3.0,
) -> np.ndarray:
    """Center and isotropically scale points to fit a Manim frame.

    Parameters
    ----------
    points:
        Array of shape ``(n, d)`` with ``d`` in {1, 2, 3}.
    target_radius:
        Half-extent (in Manim units) the largest dimension should span.

    Returns
    -------
    numpy.ndarray
        Transformed points with the same shape, centered on the origin.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim == 1:
        pts = pts[:, None]
    center = pts.mean(axis=0)
    centered = pts - center
    max_abs = np.max(np.abs(centered))
    if max_abs == 0.0:
        return centered
    return centered * (target_radius / max_abs)


# --------------------------------------------------------------------------- #
# Manim-dependent base scene (only defined when Manim is importable).
# --------------------------------------------------------------------------- #
if MANIM_AVAILABLE:  # pragma: no cover - requires Manim + FFmpeg at runtime
    from manim import (
        BLUE,
        WHITE,
        YELLOW,
        Create,
        Dot,
        Scene,
        Text,
        VMobject,
    )

    class TrajectoryScene(Scene):
        """A reusable Manim scene that animates a precomputed 2-D trajectory.

        Subclasses set :attr:`points` (an ``(n, 2)`` array in Manim coordinates)
        and optionally :attr:`title_text` before construction.
        """

        points: np.ndarray = np.zeros((1, 2))
        title_text: str = ""

        def construct(self) -> None:  # noqa: D401 - Manim entry point
            """Build and play the trajectory animation."""
            pts = fit_points_to_frame(self.points)
            if self.title_text:
                title = Text(self.title_text, font_size=32).to_edge(direction=YELLOW * 0)
                self.add(title)

            path = VMobject(color=BLUE)
            path.set_points_smoothly([np.array([p[0], p[1], 0.0]) for p in pts])
            dot = Dot(color=WHITE).move_to([pts[0][0], pts[0][1], 0.0])
            self.add(dot)
            self.play(Create(path), run_time=6)
            self.wait(0.5)
