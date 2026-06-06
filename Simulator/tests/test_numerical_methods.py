"""Tests for numerical integrators and linear-algebra helpers."""

from __future__ import annotations

import numpy as np
import pytest

from simgen.numerical_methods import (
    AVAILABLE_METHODS,
    condition_number,
    integrate,
    is_symmetric,
    power_iteration,
    rk4_step,
    spectral_radius,
)


def _harmonic_rhs(omega: float):
    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        return np.array([y[1], -(omega**2) * y[0]])

    return rhs


@pytest.mark.parametrize("method", ["rk4", "rk45", "lsoda"])
def test_harmonic_oscillator_matches_analytic(method: str) -> None:
    """Integrators reproduce x(t) = cos(omega t) for the SHO."""
    omega = 2.0
    result = integrate(
        _harmonic_rhs(omega),
        np.array([1.0, 0.0]),
        (0.0, 5.0),
        num_points=2000,
        method=method,
    )
    assert result.success
    analytic = np.cos(omega * result.t)
    error = np.max(np.abs(result.y[:, 0] - analytic))
    assert error < 1e-3


def test_rk4_more_accurate_than_euler() -> None:
    """RK4 should be far more accurate than Euler at the same step count."""
    omega = 1.0
    y0 = np.array([1.0, 0.0])
    rk4 = integrate(_harmonic_rhs(omega), y0, (0.0, 10.0), num_points=400, method="rk4")
    euler = integrate(_harmonic_rhs(omega), y0, (0.0, 10.0), num_points=400, method="euler")
    analytic = np.cos(omega * rk4.t)
    err_rk4 = np.max(np.abs(rk4.y[:, 0] - analytic))
    err_euler = np.max(np.abs(euler.y[:, 0] - analytic))
    assert err_rk4 < err_euler


def test_rk4_step_signature() -> None:
    """A single RK4 step advances a scalar exponential correctly."""

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        return y

    y1 = rk4_step(rhs, 0.0, np.array([1.0]), 0.1)
    assert y1[0] == pytest.approx(np.exp(0.1), abs=1e-5)


def test_integrate_rejects_unknown_method() -> None:
    with pytest.raises(ValueError):
        integrate(_harmonic_rhs(1.0), np.array([1.0, 0.0]), (0.0, 1.0), method="nope")


def test_integrate_rejects_bad_span() -> None:
    with pytest.raises(ValueError):
        integrate(_harmonic_rhs(1.0), np.array([1.0, 0.0]), (1.0, 0.0))


def test_available_methods_nonempty() -> None:
    assert "rk4" in AVAILABLE_METHODS
    assert all(isinstance(v, str) for v in AVAILABLE_METHODS.values())


def test_spectral_radius_diagonal() -> None:
    matrix = np.diag([1.0, -3.0, 2.0])
    assert spectral_radius(matrix) == pytest.approx(3.0)


def test_power_iteration_dominant_eigenvalue() -> None:
    matrix = np.array([[2.0, 0.0], [0.0, 5.0]])
    value, vector = power_iteration(matrix)
    assert value == pytest.approx(5.0, abs=1e-6)
    assert np.abs(vector[1]) == pytest.approx(1.0, abs=1e-6)


def test_is_symmetric() -> None:
    assert is_symmetric(np.array([[1.0, 2.0], [2.0, 1.0]]))
    assert not is_symmetric(np.array([[1.0, 2.0], [0.0, 1.0]]))


def test_condition_number_identity() -> None:
    assert condition_number(np.eye(3)) == pytest.approx(1.0)
