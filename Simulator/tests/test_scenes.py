"""Tests for scene helpers and generated Manim scene validity.

These tests do not require Manim to be installed: the pure-NumPy helpers are
tested directly, and the generated ``scene.py`` is verified by byte-compiling
it (syntax check) rather than rendering it.
"""

from __future__ import annotations

import py_compile

import numpy as np
import pytest

from simgen import generate_project
from simgen.scenes.components import (
    MANIM_AVAILABLE,
    fit_points_to_frame,
    sample_trajectory,
)
from simgen.simulations.library import get_spec, list_specs


def test_manim_available_is_bool() -> None:
    assert isinstance(MANIM_AVAILABLE, bool)


def test_sample_trajectory_downsamples() -> None:
    traj = np.arange(10000, dtype=float).reshape(-1, 1)
    sampled = sample_trajectory(traj, max_points=500)
    assert sampled.shape == (500, 1)
    # Endpoints preserved.
    assert sampled[0, 0] == 0.0
    assert sampled[-1, 0] == 9999.0


def test_sample_trajectory_keeps_short_input() -> None:
    traj = np.zeros((10, 2))
    assert sample_trajectory(traj, max_points=500).shape == (10, 2)


def test_fit_points_to_frame_centers_and_scales() -> None:
    points = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    fitted = fit_points_to_frame(points, target_radius=3.0)
    # Centered: mean ~ 0.
    assert np.allclose(fitted.mean(axis=0), 0.0, atol=1e-9)
    # Largest absolute coordinate equals the target radius.
    assert np.max(np.abs(fitted)) == pytest.approx(3.0)


def test_fit_points_handles_degenerate_input() -> None:
    points = np.zeros((5, 2))
    fitted = fit_points_to_frame(points)
    assert np.allclose(fitted, 0.0)


@pytest.mark.parametrize("slug", list_specs())
def test_generated_scene_compiles(slug: str, tmp_path) -> None:
    """Every generated Manim scene file is syntactically valid Python."""
    project = generate_project(get_spec(slug), tmp_path)
    py_compile.compile(str(project.scene_path), doraise=True)
    source = project.scene_path.read_text(encoding="utf-8")
    assert f"{get_spec(slug).class_name}Scene" in source
