"""The :class:`PhenomenonSpec` data model.

A ``PhenomenonSpec`` is the single source of truth that both the LLM engine and
the offline library produce, and that the code generator and runner consume. It
is a plain-dataclass model (no third-party dependency) with strict validation
and round-trippable JSON serialisation.

Governing equations are stored as Python expression strings for a first-order
system ``d(state_i)/dt = derivatives[i]``. Each expression may reference:

* the state-variable symbols (e.g. ``x``, ``y``, ``z``);
* the parameter symbols (e.g. ``sigma``, ``rho``, ``beta``);
* the independent variable ``t``;
* ``np`` (NumPy) and common math functions (``sin``, ``cos``, ``exp`` ...).

Second-order mechanical systems are expressed by introducing velocity state
variables and writing the corresponding first-order pair, which keeps the data
model uniform.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from simgen.utilities.io import slugify

#: Integration methods accepted in a spec (mirrors numerical_methods).
VALID_METHODS = {"euler", "rk4", "rk45", "dopri5", "radau", "bdf", "lsoda"}

#: Plot/animation kinds the framework knows how to render.
VALID_PLOT_KINDS = {"time_series", "phase2d", "phase3d"}


class SpecValidationError(ValueError):
    """Raised when a :class:`PhenomenonSpec` fails validation."""


@dataclass
class StateVariable:
    """A single state-space coordinate.

    Attributes
    ----------
    symbol:
        Python-identifier symbol used in equations (e.g. ``"x"``).
    name:
        Human-readable name (e.g. ``"x-position"``).
    description:
        Short description of the physical/mathematical meaning.
    unit:
        Unit string (may be ``"dimensionless"``).
    """

    symbol: str
    name: str = ""
    description: str = ""
    unit: str = "dimensionless"

    def __post_init__(self) -> None:
        if not self.symbol.isidentifier():
            raise SpecValidationError(
                f"State variable symbol {self.symbol!r} is not a valid identifier"
            )
        self.name = self.name or self.symbol


@dataclass
class Parameter:
    """A constant parameter of the system.

    Attributes
    ----------
    symbol:
        Python-identifier symbol used in equations.
    value:
        Numerical value.
    description:
        Short description.
    unit:
        Unit string.
    """

    symbol: str
    value: float
    description: str = ""
    unit: str = "dimensionless"

    def __post_init__(self) -> None:
        if not self.symbol.isidentifier():
            raise SpecValidationError(
                f"Parameter symbol {self.symbol!r} is not a valid identifier"
            )
        self.value = float(self.value)


@dataclass
class PlotSpec:
    """A single figure to produce from the simulation output.

    Attributes
    ----------
    kind:
        One of :data:`VALID_PLOT_KINDS`.
    title:
        Figure title.
    x, y, z:
        Variable symbols (or the literal ``"t"`` for time) mapped to axes.
        ``y``/``z`` may be ``None`` depending on ``kind``.
    """

    kind: str
    title: str = ""
    x: str = "t"
    y: str | None = None
    z: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in VALID_PLOT_KINDS:
            raise SpecValidationError(
                f"Plot kind {self.kind!r} not in {sorted(VALID_PLOT_KINDS)}"
            )


@dataclass
class AnimationSpec:
    """Description of the Manim animation to generate.

    Attributes
    ----------
    kind:
        One of :data:`VALID_PLOT_KINDS` (``"phase3d"`` produces a 3-D scene).
    title:
        On-screen title.
    description:
        Narration / caption text shown in the animation.
    x, y, z:
        Variable symbols mapped to axes.
    trail:
        Whether to draw a fading trail behind the moving point.
    """

    kind: str = "phase2d"
    title: str = ""
    description: str = ""
    x: str = "t"
    y: str | None = None
    z: str | None = None
    trail: bool = True

    def __post_init__(self) -> None:
        if self.kind not in VALID_PLOT_KINDS:
            raise SpecValidationError(
                f"Animation kind {self.kind!r} not in {sorted(VALID_PLOT_KINDS)}"
            )


@dataclass
class PhenomenonSpec:
    """Complete, self-contained description of a phenomenon to simulate.

    See the module docstring for the meaning of the equation expressions.
    """

    name: str
    summary: str
    domain: str = "physics"
    slug: str = ""
    state_variables: list[StateVariable] = field(default_factory=list)
    parameters: list[Parameter] = field(default_factory=list)
    derivatives: list[str] = field(default_factory=list)
    initial_conditions: list[float] = field(default_factory=list)
    t_start: float = 0.0
    t_end: float = 10.0
    num_points: int = 2000
    method: str = "rk4"
    equations_latex: list[str] = field(default_factory=list)
    plots: list[PlotSpec] = field(default_factory=list)
    animation: AnimationSpec = field(default_factory=AnimationSpec)
    theory: str = ""
    references: list[str] = field(default_factory=list)
    energy_expression: str | None = None

    def __post_init__(self) -> None:
        if not self.slug:
            self.slug = slugify(self.name)
        self.validate()

    # ------------------------------------------------------------------ #
    # Derived properties
    # ------------------------------------------------------------------ #
    @property
    def state_symbols(self) -> list[str]:
        """Ordered list of state-variable symbols."""
        return [sv.symbol for sv in self.state_variables]

    @property
    def parameter_map(self) -> dict[str, float]:
        """Mapping of parameter symbol -> value."""
        return {p.symbol: p.value for p in self.parameters}

    @property
    def dimension(self) -> int:
        """Number of state variables."""
        return len(self.state_variables)

    @property
    def class_name(self) -> str:
        """A PascalCase class name derived from the slug (for generated code)."""
        return "".join(part.capitalize() for part in self.slug.split("-")) or "Phenomenon"

    @property
    def t_span(self) -> tuple[float, float]:
        """The integration interval ``(t_start, t_end)``."""
        return (self.t_start, self.t_end)

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    def validate(self) -> None:
        """Validate internal consistency.

        Raises
        ------
        SpecValidationError
            If the spec is structurally inconsistent.
        """
        if self.dimension == 0:
            raise SpecValidationError("Spec must declare at least one state variable")
        if len(self.derivatives) != self.dimension:
            raise SpecValidationError(
                f"Number of derivatives ({len(self.derivatives)}) must equal number "
                f"of state variables ({self.dimension})"
            )
        if len(self.initial_conditions) != self.dimension:
            raise SpecValidationError(
                f"Number of initial conditions ({len(self.initial_conditions)}) must "
                f"equal number of state variables ({self.dimension})"
            )
        symbols = self.state_symbols
        if len(set(symbols)) != len(symbols):
            raise SpecValidationError(f"Duplicate state-variable symbols in {symbols}")
        param_symbols = [p.symbol for p in self.parameters]
        if len(set(param_symbols)) != len(param_symbols):
            raise SpecValidationError(f"Duplicate parameter symbols in {param_symbols}")
        overlap = set(symbols) & set(param_symbols)
        if overlap:
            raise SpecValidationError(
                f"Symbols used as both state and parameter: {sorted(overlap)}"
            )
        if self.method not in VALID_METHODS:
            raise SpecValidationError(
                f"Method {self.method!r} not in {sorted(VALID_METHODS)}"
            )
        if self.t_end <= self.t_start:
            raise SpecValidationError("t_end must be greater than t_start")
        if self.num_points < 2:
            raise SpecValidationError("num_points must be >= 2")

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PhenomenonSpec:
        """Build a :class:`PhenomenonSpec` from a (possibly nested) dict.

        Nested dataclasses are reconstructed from their dict form, so this is
        the inverse of :meth:`to_dict` and also accepts raw LLM/JSON output.

        Raises
        ------
        SpecValidationError
            If required keys are missing or values are inconsistent.
        """
        data = dict(data)  # shallow copy; do not mutate caller's object
        try:
            state_variables = [
                StateVariable(**sv) if isinstance(sv, dict) else sv
                for sv in data.pop("state_variables", [])
            ]
            parameters = [
                Parameter(**p) if isinstance(p, dict) else p
                for p in data.pop("parameters", [])
            ]
            plots = [
                PlotSpec(**pl) if isinstance(pl, dict) else pl
                for pl in data.pop("plots", [])
            ]
            animation_raw = data.pop("animation", None)
            animation = (
                AnimationSpec(**animation_raw)
                if isinstance(animation_raw, dict)
                else (animation_raw or AnimationSpec())
            )
        except TypeError as exc:
            raise SpecValidationError(f"Malformed nested spec field: {exc}") from exc

        if "name" not in data or "summary" not in data:
            raise SpecValidationError("Spec requires at least 'name' and 'summary'")

        return cls(
            state_variables=state_variables,
            parameters=parameters,
            plots=plots,
            animation=animation,
            **data,
        )
