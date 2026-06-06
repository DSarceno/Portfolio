"""Manim scene components.

Manim is an *optional* dependency. This subpackage imports without it; the
reusable scene base classes are only defined when Manim is available
(:data:`MANIM_AVAILABLE`). The pure-NumPy helper :func:`sample_trajectory`
works regardless and is used both here and by generated scenes.
"""

from simgen.scenes.components import (
    MANIM_AVAILABLE,
    fit_points_to_frame,
    sample_trajectory,
)

__all__ = ["MANIM_AVAILABLE", "sample_trajectory", "fit_points_to_frame"]
