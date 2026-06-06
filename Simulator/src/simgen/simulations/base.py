"""Simulation base classes and the spec-to-simulation compiler.

The framework represents every phenomenon as a first-order ODE system. A
:class:`PhenomenonSpec` is compiled into an :class:`ODESimulation` whose
right-hand side is built from the spec's derivative expressions, evaluated in a
restricted namespace that exposes only NumPy and a curated set of mathematical
functions (no builtins), so arbitrary code execution is prevented.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from simgen.numerical_methods.integrators import IntegrationResult, integrate
from simgen.simulations.spec import PhenomenonSpec
from simgen.utilities.logging_config import get_logger

logger = get_logger(__name__)

RHS = Callable[[float, np.ndarray], np.ndarray]


def _safe_namespace() -> dict[str, object]:
    """Build the restricted evaluation namespace for equation expressions.

    Exposes NumPy as ``np`` plus common math functions and constants, but no
    Python builtins, preventing access to ``__import__``, ``open`` etc.
    """
    allowed = {
        "np": np,
        "pi": math.pi,
        "e": math.e,
        "tau": math.tau,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "asin": np.arcsin,
        "acos": np.arccos,
        "atan": np.arctan,
        "atan2": np.arctan2,
        "sinh": np.sinh,
        "cosh": np.cosh,
        "tanh": np.tanh,
        "exp": np.exp,
        "log": np.log,
        "log10": np.log10,
        "sqrt": np.sqrt,
        "abs": np.abs,
        "sign": np.sign,
        "floor": np.floor,
        "ceil": np.ceil,
        "power": np.power,
    }
    return allowed


@dataclass
class SimulationResult:
    """The result of running a simulation.

    Attributes
    ----------
    t:
        Time grid, shape ``(n_steps,)``.
    states:
        State history, shape ``(n_steps, n_dim)``.
    symbols:
        Ordered state-variable symbols matching the columns of ``states``.
    method:
        Integration method used.
    success:
        Whether integration succeeded.
    message:
        Status message from the integrator.
    """

    t: np.ndarray
    states: np.ndarray
    symbols: list[str]
    method: str
    success: bool = True
    message: str = "ok"

    @property
    def n_steps(self) -> int:
        """Number of time points in the trajectory."""
        return int(self.t.shape[0])

    @property
    def dt(self) -> float:
        """Sampling interval (assumes a uniform grid)."""
        if self.t.shape[0] < 2:
            return 0.0
        return float(self.t[1] - self.t[0])

    def column(self, symbol: str) -> np.ndarray:
        """Return the trajectory of a single state variable by symbol.

        Parameters
        ----------
        symbol:
            State-variable symbol, or the literal ``"t"`` for time.

        Raises
        ------
        KeyError
            If ``symbol`` is not a known state variable or ``"t"``.
        """
        if symbol == "t":
            return self.t
        try:
            idx = self.symbols.index(symbol)
        except ValueError as exc:
            raise KeyError(
                f"Unknown variable {symbol!r}; known: {self.symbols + ['t']}"
            ) from exc
        return self.states[:, idx]

    def as_dict(self) -> dict[str, np.ndarray]:
        """Return a mapping of every symbol (and ``t``) to its trajectory."""
        data = {"t": self.t}
        for i, sym in enumerate(self.symbols):
            data[sym] = self.states[:, i]
        return data


class Simulation(ABC):
    """Abstract base class for all simulations.

    Subclasses implement :meth:`rhs` (the right-hand side of the ODE system)
    and may override :meth:`energy` for conservation diagnostics.
    """

    def __init__(self, spec: PhenomenonSpec) -> None:
        self.spec = spec

    @abstractmethod
    def rhs(self, t: float, state: np.ndarray) -> np.ndarray:
        """Return ``d(state)/dt`` at time ``t``."""

    def energy(self, state: np.ndarray) -> float | None:
        """Return the system energy for ``state`` (``None`` if undefined)."""
        return None

    def run(self) -> SimulationResult:
        """Integrate the system over the spec's time span.

        Returns
        -------
        SimulationResult
            The trajectory and metadata.
        """
        result: IntegrationResult = integrate(
            self.rhs,
            np.asarray(self.spec.initial_conditions, dtype=float),
            self.spec.t_span,
            num_points=self.spec.num_points,
            method=self.spec.method,
        )
        if not result.success:
            logger.warning("Integration reported failure: %s", result.message)
        return SimulationResult(
            t=result.t,
            states=result.y,
            symbols=self.spec.state_symbols,
            method=result.method,
            success=result.success,
            message=result.message,
        )


class ODESimulation(Simulation):
    """A simulation whose RHS is compiled from a :class:`PhenomenonSpec`.

    Derivative expressions are compiled once with :func:`compile` and evaluated
    in the restricted namespace on each step for speed and safety.
    """

    def __init__(self, spec: PhenomenonSpec) -> None:
        super().__init__(spec)
        self._symbols = spec.state_symbols
        self._params = spec.parameter_map
        self._base_ns = _safe_namespace()
        self._compiled = [
            compile(expr, f"<derivative:{sym}>", "eval")
            for sym, expr in zip(self._symbols, spec.derivatives, strict=True)
        ]
        self._energy_code = (
            compile(spec.energy_expression, "<energy>", "eval")
            if spec.energy_expression
            else None
        )

    def _eval_namespace(self, t: float, state: np.ndarray) -> dict[str, object]:
        ns = dict(self._base_ns)
        ns.update(self._params)
        for sym, value in zip(self._symbols, state, strict=False):
            ns[sym] = value
        ns["t"] = t
        return ns

    def rhs(self, t: float, state: np.ndarray) -> np.ndarray:
        ns = self._eval_namespace(t, np.asarray(state, dtype=float))
        try:
            return np.array(
                [float(eval(code, {"__builtins__": {}}, ns)) for code in self._compiled],
                dtype=float,
            )
        except Exception as exc:  # noqa: BLE001 - surface a clear error to caller
            raise RuntimeError(
                f"Error evaluating derivatives for {self.spec.slug!r}: {exc}"
            ) from exc

    def energy(self, state: np.ndarray) -> float | None:
        if self._energy_code is None:
            return None
        ns = self._eval_namespace(0.0, np.asarray(state, dtype=float))
        return float(eval(self._energy_code, {"__builtins__": {}}, ns))


def build_simulation(spec: PhenomenonSpec) -> ODESimulation:
    """Compile a :class:`PhenomenonSpec` into a runnable :class:`ODESimulation`.

    Parameters
    ----------
    spec:
        A validated phenomenon specification.

    Returns
    -------
    ODESimulation
        Ready to ``.run()``.
    """
    return ODESimulation(spec)
