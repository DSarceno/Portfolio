"""A minimal, dependency-free units system for dimensional bookkeeping.

This is intentionally lightweight: it tracks the seven SI base dimensions and
supports multiplication, division, powers and addition (with a dimensional
consistency check). It is sufficient for validating that derived quantities in
simulations and reports are dimensionally sound, without pulling in a heavy
units library.

Dimensions are represented as a 7-tuple of exponents in the order
``(length, mass, time, current, temperature, amount, luminous_intensity)``.
"""

from __future__ import annotations

from dataclasses import dataclass

_DIMENSION_NAMES = ("m", "kg", "s", "A", "K", "mol", "cd")
_NDIM = len(_DIMENSION_NAMES)
Dimension = tuple[int, ...]

DIMENSIONLESS: Dimension = (0, 0, 0, 0, 0, 0, 0)

#: Convenience base dimensions.
LENGTH: Dimension = (1, 0, 0, 0, 0, 0, 0)
MASS: Dimension = (0, 1, 0, 0, 0, 0, 0)
TIME: Dimension = (0, 0, 1, 0, 0, 0, 0)
CURRENT: Dimension = (0, 0, 0, 1, 0, 0, 0)
TEMPERATURE: Dimension = (0, 0, 0, 0, 1, 0, 0)


class UnitError(ValueError):
    """Raised when an operation violates dimensional consistency."""


@dataclass(frozen=True)
class Quantity:
    """A scalar value tagged with SI dimensions.

    Attributes
    ----------
    value:
        Numerical magnitude (in SI base units).
    dimension:
        7-tuple of base-dimension exponents.
    """

    value: float
    dimension: Dimension = DIMENSIONLESS

    def __post_init__(self) -> None:
        if len(self.dimension) != _NDIM:
            raise UnitError(
                f"Dimension must have {_NDIM} exponents, got {len(self.dimension)}"
            )

    @property
    def is_dimensionless(self) -> bool:
        """Whether the quantity carries no dimensions."""
        return self.dimension == DIMENSIONLESS

    def __add__(self, other: Quantity) -> Quantity:
        self._check_same_dimension(other, "add")
        return Quantity(self.value + other.value, self.dimension)

    def __sub__(self, other: Quantity) -> Quantity:
        self._check_same_dimension(other, "subtract")
        return Quantity(self.value - other.value, self.dimension)

    def __mul__(self, other: Quantity | float | int) -> Quantity:
        if isinstance(other, (int, float)):
            return Quantity(self.value * other, self.dimension)
        new_dim = tuple(a + b for a, b in zip(self.dimension, other.dimension, strict=True))
        return Quantity(self.value * other.value, new_dim)

    __rmul__ = __mul__

    def __truediv__(self, other: Quantity | float | int) -> Quantity:
        if isinstance(other, (int, float)):
            return Quantity(self.value / other, self.dimension)
        new_dim = tuple(a - b for a, b in zip(self.dimension, other.dimension, strict=True))
        return Quantity(self.value / other.value, new_dim)

    def __pow__(self, exponent: int) -> Quantity:
        new_dim = tuple(d * exponent for d in self.dimension)
        return Quantity(self.value**exponent, new_dim)

    def _check_same_dimension(self, other: Quantity, op: str) -> None:
        if self.dimension != other.dimension:
            raise UnitError(
                f"Cannot {op} quantities with dimensions "
                f"{self.unit_string()} and {other.unit_string()}"
            )

    def unit_string(self) -> str:
        """Return a compact human-readable unit string (e.g. ``"m s^-2"``)."""
        parts = []
        for name, exp in zip(_DIMENSION_NAMES, self.dimension, strict=True):
            if exp == 0:
                continue
            parts.append(name if exp == 1 else f"{name}^{exp}")
        return " ".join(parts) if parts else "dimensionless"

    def __repr__(self) -> str:
        return f"Quantity({self.value!r}, '{self.unit_string()}')"


def dimensionless(value: float) -> Quantity:
    """Construct a dimensionless :class:`Quantity`."""
    return Quantity(value, DIMENSIONLESS)


#: Simple linear unit conversion factors to SI base units.
_CONVERSIONS: dict[str, float] = {
    "m": 1.0,
    "km": 1.0e3,
    "cm": 1.0e-2,
    "mm": 1.0e-3,
    "s": 1.0,
    "ms": 1.0e-3,
    "minute": 60.0,
    "hour": 3600.0,
    "kg": 1.0,
    "g": 1.0e-3,
    "degree": 0.017453292519943295,  # to radians
    "radian": 1.0,
}


def convert(value: float, from_unit: str, to_unit: str) -> float:
    """Convert ``value`` between two compatible linear units.

    Parameters
    ----------
    value:
        Magnitude in ``from_unit``.
    from_unit, to_unit:
        Unit names present in the internal conversion table.

    Returns
    -------
    float
        The converted magnitude.

    Raises
    ------
    UnitError
        If a unit is unknown.
    """
    if from_unit not in _CONVERSIONS:
        raise UnitError(f"Unknown source unit {from_unit!r}")
    if to_unit not in _CONVERSIONS:
        raise UnitError(f"Unknown target unit {to_unit!r}")
    return value * _CONVERSIONS[from_unit] / _CONVERSIONS[to_unit]
