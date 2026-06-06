"""High-level time integration for first-order ODE systems.

The :func:`integrate` dispatcher provides a uniform interface over:

* fixed-step explicit schemes implemented in this package
  (``"euler"``, ``"rk4"``); and
* SciPy's adaptive ``solve_ivp`` methods (``"rk45"``, ``"dopri5"``,
  ``"radau"``, ``"bdf"``, ``"lsoda"``) for non-stiff and stiff problems.

All methods return a uniform :class:`IntegrationResult`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from simgen.numerical_methods.ode_solvers import euler_step, rk4_step
from simgen.utilities.logging_config import get_logger

logger = get_logger(__name__)

RHS = Callable[[float, np.ndarray], np.ndarray]

#: Mapping of method name -> human readable description.
AVAILABLE_METHODS: dict[str, str] = {
    "euler": "Fixed-step explicit (forward) Euler, 1st order.",
    "rk4": "Fixed-step classic Runge-Kutta, 4th order.",
    "rk45": "SciPy adaptive Dormand-Prince RK45 (non-stiff).",
    "dopri5": "SciPy adaptive Dormand-Prince 5(4) (non-stiff).",
    "radau": "SciPy implicit Radau IIA (stiff).",
    "bdf": "SciPy backward differentiation formula (stiff).",
    "lsoda": "SciPy LSODA with automatic stiffness switching.",
}

_SCIPY_METHODS = {
    "rk45": "RK45",
    "dopri5": "DOP853",
    "radau": "Radau",
    "bdf": "BDF",
    "lsoda": "LSODA",
}


@dataclass
class IntegrationResult:
    """Container for the output of a time integration.

    Attributes
    ----------
    t:
        1-D array of time points, shape ``(n_steps,)``.
    y:
        State history with shape ``(n_steps, n_dim)`` (row per time point).
    method:
        Name of the integration method used.
    success:
        Whether the integrator reported success.
    message:
        Human-readable status message.
    """

    t: np.ndarray
    y: np.ndarray
    method: str
    success: bool = True
    message: str = "ok"

    @property
    def n_steps(self) -> int:
        """Number of time points."""
        return int(self.t.shape[0])

    @property
    def n_dim(self) -> int:
        """Dimensionality of the state vector."""
        return int(self.y.shape[1]) if self.y.ndim == 2 else 1


def _integrate_fixed(
    f: RHS,
    y0: np.ndarray,
    t: np.ndarray,
    stepper: Callable[[RHS, float, np.ndarray, float], np.ndarray],
    method: str,
) -> IntegrationResult:
    """Integrate on a fixed time grid using a single-step ``stepper``."""
    y0 = np.asarray(y0, dtype=float)
    history = np.empty((t.shape[0], y0.shape[0]), dtype=float)
    history[0] = y0
    for i in range(t.shape[0] - 1):
        dt = float(t[i + 1] - t[i])
        history[i + 1] = stepper(f, float(t[i]), history[i], dt)
        if not np.all(np.isfinite(history[i + 1])):
            logger.warning("Non-finite state at step %d with method %s", i + 1, method)
            return IntegrationResult(
                t=t[: i + 2],
                y=history[: i + 2],
                method=method,
                success=False,
                message=f"Non-finite state encountered at t={t[i + 1]:.6g}",
            )
    return IntegrationResult(t=t, y=history, method=method)


def integrate(
    f: RHS,
    y0: np.ndarray,
    t_span: tuple[float, float],
    *,
    num_points: int = 2000,
    method: str = "rk4",
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> IntegrationResult:
    """Integrate ``dy/dt = f(t, y)`` from ``t_span[0]`` to ``t_span[1]``.

    Parameters
    ----------
    f:
        Right-hand side ``f(t, y) -> dy/dt``.
    y0:
        Initial state vector.
    t_span:
        ``(t0, t1)`` integration interval.
    num_points:
        Number of output time points (uniformly spaced).
    method:
        One of the keys in :data:`AVAILABLE_METHODS`.
    rtol, atol:
        Relative/absolute tolerances for adaptive SciPy methods.

    Returns
    -------
    IntegrationResult
        Time grid and state history.

    Raises
    ------
    ValueError
        If ``method`` is unknown or inputs are inconsistent.
    """
    method = method.lower()
    if method not in AVAILABLE_METHODS:
        raise ValueError(
            f"Unknown method {method!r}. Available: {sorted(AVAILABLE_METHODS)}"
        )
    t0, t1 = float(t_span[0]), float(t_span[1])
    if t1 <= t0:
        raise ValueError(f"t_span must be increasing, got {t_span!r}")
    if num_points < 2:
        raise ValueError("num_points must be >= 2")

    y0 = np.asarray(y0, dtype=float).ravel()
    t = np.linspace(t0, t1, num_points)

    if method == "euler":
        return _integrate_fixed(f, y0, t, euler_step, method)
    if method == "rk4":
        return _integrate_fixed(f, y0, t, rk4_step, method)

    # Adaptive SciPy methods.
    from scipy.integrate import solve_ivp  # local import keeps base import light

    scipy_name = _SCIPY_METHODS[method]
    solution = solve_ivp(
        fun=f,
        t_span=(t0, t1),
        y0=y0,
        method=scipy_name,
        t_eval=t,
        rtol=rtol,
        atol=atol,
        dense_output=False,
    )
    return IntegrationResult(
        t=solution.t,
        y=solution.y.T,
        method=method,
        success=bool(solution.success),
        message=str(solution.message),
    )
