"""Numerical methods: ODE integrators, time-stepping schemes and linear algebra.

The public surface is the :func:`integrate` dispatcher plus the individual
fixed-step integrators and the SciPy-backed adaptive bridge.
"""

from simgen.numerical_methods.integrators import (
    AVAILABLE_METHODS,
    IntegrationResult,
    integrate,
)
from simgen.numerical_methods.linalg import (
    condition_number,
    is_symmetric,
    power_iteration,
    spectral_radius,
)
from simgen.numerical_methods.ode_solvers import (
    euler_step,
    rk4_step,
    velocity_verlet_step,
)

__all__ = [
    "integrate",
    "IntegrationResult",
    "AVAILABLE_METHODS",
    "euler_step",
    "rk4_step",
    "velocity_verlet_step",
    "condition_number",
    "spectral_radius",
    "power_iteration",
    "is_symmetric",
]
