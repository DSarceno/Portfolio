"""Public access point for the reusable Manim base scene.

Importing this module never fails: if Manim is unavailable, :data:`TrajectoryScene`
is ``None`` and :data:`MANIM_AVAILABLE` is ``False``. This lets the rest of the
package (and the test suite) import scene utilities unconditionally.
"""

from __future__ import annotations

from simgen.scenes.components import MANIM_AVAILABLE

TrajectoryScene = None
if MANIM_AVAILABLE:  # pragma: no cover - requires Manim at runtime
    from simgen.scenes.components import TrajectoryScene  # type: ignore[assignment]

__all__ = ["MANIM_AVAILABLE", "TrajectoryScene"]
