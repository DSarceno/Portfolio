"""Single-step ODE update rules.

Each function advances a state vector ``y`` by one time step ``dt`` for a
first-order system ``dy/dt = f(t, y)``. These are the building blocks used by
the fixed-step integrators in :mod:`simgen.numerical_methods.integrators`.

All functions are pure (no side effects) and return a new array.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

RHS = Callable[[float, np.ndarray], np.ndarray]


def euler_step(f: RHS, t: float, y: np.ndarray, dt: float) -> np.ndarray:
    """Advance one step with the explicit (forward) Euler method.

    First-order accurate, ``O(dt)`` local truncation error. Cheap but only
    conditionally stable; included for pedagogical comparison.

    Parameters
    ----------
    f:
        Right-hand side ``f(t, y)`` returning ``dy/dt``.
    t:
        Current time.
    y:
        Current state vector.
    dt:
        Time step.

    Returns
    -------
    numpy.ndarray
        The state at ``t + dt``.
    """
    return y + dt * np.asarray(f(t, y), dtype=float)


def rk4_step(f: RHS, t: float, y: np.ndarray, dt: float) -> np.ndarray:
    """Advance one step with the classic 4th-order Runge–Kutta method.

    Fourth-order accurate, ``O(dt^4)`` local error. The workhorse explicit
    method for smooth, non-stiff systems.
    """
    y = np.asarray(y, dtype=float)
    k1 = np.asarray(f(t, y), dtype=float)
    k2 = np.asarray(f(t + 0.5 * dt, y + 0.5 * dt * k1), dtype=float)
    k3 = np.asarray(f(t + 0.5 * dt, y + 0.5 * dt * k2), dtype=float)
    k4 = np.asarray(f(t + dt, y + dt * k3), dtype=float)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def velocity_verlet_step(
    acceleration: Callable[[float, np.ndarray, np.ndarray], np.ndarray],
    t: float,
    positions: np.ndarray,
    velocities: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance one step with the velocity Verlet symplectic integrator.

    Suitable for second-order conservative systems ``d^2 x/dt^2 = a(t, x, v)``.
    Symplectic integrators conserve a shadow Hamiltonian and therefore exhibit
    excellent long-term energy behaviour for mechanical systems.

    Parameters
    ----------
    acceleration:
        Function ``a(t, x, v)`` returning the acceleration.
    t:
        Current time.
    positions:
        Current position vector.
    velocities:
        Current velocity vector.
    dt:
        Time step.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        New ``(positions, velocities)`` at ``t + dt``.
    """
    x = np.asarray(positions, dtype=float)
    v = np.asarray(velocities, dtype=float)
    a = np.asarray(acceleration(t, x, v), dtype=float)
    x_new = x + v * dt + 0.5 * a * dt * dt
    a_new = np.asarray(acceleration(t + dt, x_new, v + a * dt), dtype=float)
    v_new = v + 0.5 * (a + a_new) * dt
    return x_new, v_new
