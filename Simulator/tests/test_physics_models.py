"""Tests for physical constants and the units system."""

from __future__ import annotations

import pytest

from simgen.physics import (
    CONSTANTS,
    Quantity,
    UnitError,
    convert,
    dimensionless,
    get_constant,
)
from simgen.physics.units import LENGTH, MASS, TIME


def test_speed_of_light_value() -> None:
    c = get_constant("speed_of_light")
    assert c.value == pytest.approx(2.99792458e8)
    assert c.unit == "m/s"


def test_all_constants_have_metadata() -> None:
    for const in CONSTANTS.values():
        assert const.symbol
        assert const.unit
        assert const.description
        assert isinstance(const.value, float)


def test_unknown_constant_raises() -> None:
    with pytest.raises(KeyError):
        get_constant("unobtanium_constant")


def test_quantity_multiplication_combines_dimensions() -> None:
    force = Quantity(2.0, MASS) * Quantity(3.0, LENGTH) / (Quantity(1.0, TIME) ** 2)
    assert force.value == pytest.approx(6.0)
    assert force.unit_string() == "m kg s^-2"


def test_quantity_addition_requires_matching_dimensions() -> None:
    with pytest.raises(UnitError):
        Quantity(1.0, LENGTH) + Quantity(1.0, TIME)


def test_quantity_addition_same_dimension() -> None:
    total = Quantity(2.0, LENGTH) + Quantity(3.0, LENGTH)
    assert total.value == pytest.approx(5.0)
    assert total.dimension == LENGTH


def test_dimensionless_helper() -> None:
    q = dimensionless(4.0)
    assert q.is_dimensionless
    assert q.unit_string() == "dimensionless"


def test_convert_length_and_time() -> None:
    assert convert(1.0, "km", "m") == pytest.approx(1000.0)
    assert convert(90.0, "degree", "radian") == pytest.approx(1.5707963, abs=1e-6)


def test_convert_unknown_unit_raises() -> None:
    with pytest.raises(UnitError):
        convert(1.0, "parsec", "m")
