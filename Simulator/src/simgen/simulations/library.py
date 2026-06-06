"""Built-in library of validated reference phenomenon specifications.

These serve three purposes:

1. They let the whole pipeline (generation, running, plotting, animation,
   reporting) work fully offline with no API key.
2. They are the fallback when LLM generation is unavailable.
3. They are the fixtures for the deterministic test suite.

Each spec is hand-derived and unit-checked; see ``docs/SCIENTIFIC_BACKGROUND.md``
for the underlying physics and mathematics.
"""

from __future__ import annotations

from simgen.simulations.spec import (
    AnimationSpec,
    Parameter,
    PhenomenonSpec,
    PlotSpec,
    StateVariable,
)


def _lorenz() -> PhenomenonSpec:
    """The Lorenz system: a canonical 3-D chaotic attractor."""
    return PhenomenonSpec(
        name="Lorenz System",
        slug="lorenz-system",
        domain="physics",
        summary=(
            "A simplified model of atmospheric convection (Lorenz, 1963) whose "
            "trajectories settle onto a strange attractor and exhibit sensitive "
            "dependence on initial conditions — deterministic chaos."
        ),
        state_variables=[
            StateVariable("x", "Convection rate", "Proportional to convection intensity."),
            StateVariable("y", "Horizontal temperature variation", "Temperature difference."),
            StateVariable("z", "Vertical temperature variation", "Distortion of vertical profile."),
        ],
        parameters=[
            Parameter("sigma", 10.0, "Prandtl number."),
            Parameter("rho", 28.0, "Rayleigh number (relative to critical)."),
            Parameter("beta", 8.0 / 3.0, "Geometric aspect-ratio factor."),
        ],
        derivatives=[
            "sigma * (y - x)",
            "x * (rho - z) - y",
            "x * y - beta * z",
        ],
        initial_conditions=[1.0, 1.0, 1.0],
        t_start=0.0,
        t_end=40.0,
        num_points=8000,
        method="rk4",
        equations_latex=[
            r"\dot{x} = \sigma (y - x)",
            r"\dot{y} = x(\rho - z) - y",
            r"\dot{z} = xy - \beta z",
        ],
        plots=[
            PlotSpec("phase3d", "Lorenz attractor", x="x", y="y", z="z"),
            PlotSpec("time_series", "x(t)", x="t", y="x"),
        ],
        animation=AnimationSpec(
            kind="phase3d",
            title="The Lorenz Attractor",
            description="A trajectory winding around the two lobes of the strange attractor.",
            x="x",
            y="y",
            z="z",
        ),
        theory=(
            "The Lorenz equations arise from a severe Galerkin truncation of the "
            "Rayleigh--Benard convection problem. For rho > 24.74 the origin and the "
            "two symmetric fixed points are all unstable, and trajectories are "
            "attracted to a fractal set of dimension ~2.06."
        ),
        references=["strogatz2015", "lorenz1963"],
    )


def _double_pendulum() -> PhenomenonSpec:
    """The planar double pendulum: a low-dimensional mechanical chaotic system."""
    domega1 = (
        "(m2 * l1 * omega1**2 * sin(theta2 - theta1) * cos(theta2 - theta1)"
        " + m2 * g * sin(theta2) * cos(theta2 - theta1)"
        " + m2 * l2 * omega2**2 * sin(theta2 - theta1)"
        " - (m1 + m2) * g * sin(theta1))"
        " / ((m1 + m2) * l1 - m2 * l1 * cos(theta2 - theta1)**2)"
    )
    domega2 = (
        "(-m2 * l2 * omega2**2 * sin(theta2 - theta1) * cos(theta2 - theta1)"
        " + (m1 + m2) * (g * sin(theta1) * cos(theta2 - theta1)"
        " - l1 * omega1**2 * sin(theta2 - theta1)"
        " - g * sin(theta2)))"
        " / ((l2 / l1) * ((m1 + m2) * l1 - m2 * l1 * cos(theta2 - theta1)**2))"
    )
    return PhenomenonSpec(
        name="Double Pendulum",
        slug="double-pendulum",
        domain="physics",
        summary=(
            "Two coupled rigid pendulums. Despite being deterministic and having "
            "only two degrees of freedom, the motion is chaotic for generic energies."
        ),
        state_variables=[
            StateVariable("theta1", "Upper angle", "Angle of the first arm from vertical.", "rad"),
            StateVariable("omega1", "Upper angular velocity", "Time derivative of theta1.", "rad/s"),
            StateVariable("theta2", "Lower angle", "Angle of the second arm from vertical.", "rad"),
            StateVariable("omega2", "Lower angular velocity", "Time derivative of theta2.", "rad/s"),
        ],
        parameters=[
            Parameter("m1", 1.0, "Mass of the upper bob.", "kg"),
            Parameter("m2", 1.0, "Mass of the lower bob.", "kg"),
            Parameter("l1", 1.0, "Length of the upper arm.", "m"),
            Parameter("l2", 1.0, "Length of the lower arm.", "m"),
            Parameter("g", 9.81, "Gravitational acceleration.", "m/s^2"),
        ],
        derivatives=["omega1", domega1, "omega2", domega2],
        initial_conditions=[1.5707963267948966, 0.0, 1.5707963267948966, 0.0],
        t_start=0.0,
        t_end=20.0,
        num_points=6000,
        method="rk4",
        equations_latex=[
            r"\dot{\theta}_1 = \omega_1,\quad \dot{\theta}_2 = \omega_2",
            r"\dot{\omega}_1 = f_1(\theta_1,\theta_2,\omega_1,\omega_2)",
            r"\dot{\omega}_2 = f_2(\theta_1,\theta_2,\omega_1,\omega_2)",
        ],
        plots=[
            PlotSpec("phase2d", "Phase portrait (theta1, omega1)", x="theta1", y="omega1"),
            PlotSpec("time_series", "Angles vs time", x="t", y="theta1"),
        ],
        animation=AnimationSpec(
            kind="phase2d",
            title="Double Pendulum Phase Space",
            description="Angular position versus angular velocity of the upper arm.",
            x="theta1",
            y="omega1",
        ),
        theory=(
            "Derived from the Euler--Lagrange equations of the planar double "
            "pendulum. The configuration space is a 2-torus and the dynamics are "
            "Hamiltonian; energy is conserved by the exact flow."
        ),
        energy_expression=(
            "0.5 * m1 * (l1 * omega1)**2"
            " + 0.5 * m2 * ((l1 * omega1)**2 + (l2 * omega2)**2"
            " + 2 * l1 * l2 * omega1 * omega2 * cos(theta1 - theta2))"
            " - (m1 + m2) * g * l1 * cos(theta1) - m2 * g * l2 * cos(theta2)"
        ),
        references=["goldstein2002", "strogatz2015"],
    )


def _harmonic_oscillator() -> PhenomenonSpec:
    """The simple harmonic oscillator: the prototype of linear oscillation."""
    return PhenomenonSpec(
        name="Simple Harmonic Oscillator",
        slug="simple-harmonic-oscillator",
        domain="physics",
        summary=(
            "A mass on an ideal spring. The restoring force is proportional to "
            "displacement, giving sinusoidal motion at a single natural frequency."
        ),
        state_variables=[
            StateVariable("x", "Displacement", "Displacement from equilibrium.", "m"),
            StateVariable("v", "Velocity", "Rate of change of displacement.", "m/s"),
        ],
        parameters=[Parameter("omega", 2.0, "Natural angular frequency.", "rad/s")],
        derivatives=["v", "-omega**2 * x"],
        initial_conditions=[1.0, 0.0],
        t_start=0.0,
        t_end=20.0,
        num_points=4000,
        method="rk4",
        equations_latex=[r"\dot{x} = v", r"\dot{v} = -\omega^2 x"],
        plots=[
            PlotSpec("time_series", "Displacement vs time", x="t", y="x"),
            PlotSpec("phase2d", "Phase portrait", x="x", y="v"),
        ],
        animation=AnimationSpec(
            kind="phase2d",
            title="Harmonic Oscillator Phase Space",
            description="The closed elliptical orbit of a conservative oscillator.",
            x="x",
            y="v",
        ),
        theory=(
            "The equation x'' + omega^2 x = 0 has solution x(t) = A cos(omega t + phi). "
            "Energy E = (1/2)v^2 + (1/2)omega^2 x^2 is conserved, so phase-space "
            "orbits are ellipses."
        ),
        energy_expression="0.5 * v**2 + 0.5 * omega**2 * x**2",
        references=["goldstein2002", "strogatz2015"],
    )


def _duffing() -> PhenomenonSpec:
    """The driven Duffing oscillator: a forced nonlinear oscillator."""
    return PhenomenonSpec(
        name="Duffing Oscillator",
        slug="duffing-oscillator",
        domain="physics",
        summary=(
            "A damped, periodically driven oscillator with a cubic nonlinearity. "
            "For suitable forcing it exhibits a chaotic strange attractor."
        ),
        state_variables=[
            StateVariable("x", "Displacement", "Displacement of the oscillator.", "m"),
            StateVariable("v", "Velocity", "Rate of change of displacement.", "m/s"),
        ],
        parameters=[
            Parameter("delta", 0.3, "Linear damping coefficient."),
            Parameter("alpha", -1.0, "Linear stiffness coefficient."),
            Parameter("beta", 1.0, "Cubic stiffness coefficient."),
            Parameter("gamma", 0.37, "Forcing amplitude."),
            Parameter("omega", 1.2, "Forcing angular frequency."),
        ],
        derivatives=["v", "gamma * cos(omega * t) - delta * v - alpha * x - beta * x**3"],
        initial_conditions=[0.1, 0.0],
        t_start=0.0,
        t_end=120.0,
        num_points=8000,
        method="rk45",
        equations_latex=[
            r"\dot{x} = v",
            r"\dot{v} = \gamma\cos(\omega t) - \delta v - \alpha x - \beta x^3",
        ],
        plots=[
            PlotSpec("phase2d", "Duffing phase portrait", x="x", y="v"),
            PlotSpec("time_series", "Displacement vs time", x="t", y="x"),
        ],
        animation=AnimationSpec(
            kind="phase2d",
            title="Duffing Oscillator",
            description="A forced nonlinear oscillator tracing a chaotic attractor.",
            x="x",
            y="v",
        ),
        theory=(
            "The Duffing equation x'' + delta x' + alpha x + beta x^3 = gamma cos(omega t) "
            "models a hardening/softening spring. The double-well case (alpha<0, beta>0) "
            "produces homoclinic tangling and chaos."
        ),
        references=["guckenheimer1983", "strogatz2015"],
    )


def _van_der_pol() -> PhenomenonSpec:
    """The Van der Pol oscillator: a self-sustaining nonlinear limit cycle."""
    return PhenomenonSpec(
        name="Van der Pol Oscillator",
        slug="van-der-pol-oscillator",
        domain="mathematics",
        summary=(
            "A nonlinear oscillator with nonlinear damping that injects energy at "
            "small amplitude and removes it at large amplitude, producing a unique "
            "stable limit cycle."
        ),
        state_variables=[
            StateVariable("x", "Position", "Oscillator coordinate."),
            StateVariable("v", "Velocity", "Rate of change of position."),
        ],
        parameters=[Parameter("mu", 2.0, "Nonlinearity / damping strength.")],
        derivatives=["v", "mu * (1 - x**2) * v - x"],
        initial_conditions=[2.0, 0.0],
        t_start=0.0,
        t_end=40.0,
        num_points=4000,
        method="rk45",
        equations_latex=[r"\dot{x} = v", r"\dot{v} = \mu(1 - x^2)v - x"],
        plots=[
            PlotSpec("phase2d", "Van der Pol limit cycle", x="x", y="v"),
            PlotSpec("time_series", "Position vs time", x="t", y="x"),
        ],
        animation=AnimationSpec(
            kind="phase2d",
            title="Van der Pol Limit Cycle",
            description="Trajectories from inside and outside spiral onto the limit cycle.",
            x="x",
            y="v",
        ),
        theory=(
            "x'' - mu(1 - x^2)x' + x = 0. By the Poincare--Bendixson theorem the "
            "system has a unique attracting limit cycle for mu>0; for large mu the "
            "motion becomes a relaxation oscillation."
        ),
        references=["strogatz2015", "guckenheimer1983"],
    )


#: Registry of reference specs keyed by slug. Builders are called on demand so
#: that each caller receives an independent, freshly validated instance.
_BUILDERS = {
    "lorenz-system": _lorenz,
    "double-pendulum": _double_pendulum,
    "simple-harmonic-oscillator": _harmonic_oscillator,
    "duffing-oscillator": _duffing,
    "van-der-pol-oscillator": _van_der_pol,
}


def list_specs() -> list[str]:
    """Return the sorted slugs of all available reference specs."""
    return sorted(_BUILDERS)


def get_spec(slug: str) -> PhenomenonSpec:
    """Return a fresh reference spec by slug.

    Parameters
    ----------
    slug:
        One of the slugs returned by :func:`list_specs`.

    Returns
    -------
    PhenomenonSpec

    Raises
    ------
    KeyError
        If ``slug`` is not a known reference spec.
    """
    try:
        return _BUILDERS[slug]()
    except KeyError as exc:
        raise KeyError(
            f"Unknown spec {slug!r}. Available: {list_specs()}"
        ) from exc


def all_specs() -> list[PhenomenonSpec]:
    """Return a freshly built instance of every reference spec."""
    return [builder() for builder in _BUILDERS.values()]
