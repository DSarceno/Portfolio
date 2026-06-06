"""Curated physical constants in SI units (CODATA 2018 values).

Each constant carries its value, unit string, symbol and a short description so
it can be displayed in reports and validated in tests.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PhysicalConstant:
    """A named physical constant with metadata.

    Attributes
    ----------
    symbol:
        Conventional symbol (e.g. ``"c"``).
    value:
        Numerical value in SI units.
    unit:
        SI unit string.
    description:
        Human-readable description.
    """

    symbol: str
    value: float
    unit: str
    description: str


#: Registry of physical constants keyed by lowercase name.
CONSTANTS: dict[str, PhysicalConstant] = {
    "speed_of_light": PhysicalConstant(
        "c", 2.99792458e8, "m/s", "Speed of light in vacuum (exact)."
    ),
    "gravitational_constant": PhysicalConstant(
        "G", 6.67430e-11, "m^3 kg^-1 s^-2", "Newtonian constant of gravitation."
    ),
    "planck_constant": PhysicalConstant(
        "h", 6.62607015e-34, "J s", "Planck constant (exact)."
    ),
    "reduced_planck_constant": PhysicalConstant(
        "hbar", 1.054571817e-34, "J s", "Reduced Planck constant h/(2*pi)."
    ),
    "boltzmann_constant": PhysicalConstant(
        "k_B", 1.380649e-23, "J/K", "Boltzmann constant (exact)."
    ),
    "elementary_charge": PhysicalConstant(
        "e", 1.602176634e-19, "C", "Elementary charge (exact)."
    ),
    "electron_mass": PhysicalConstant(
        "m_e", 9.1093837015e-31, "kg", "Electron rest mass."
    ),
    "proton_mass": PhysicalConstant(
        "m_p", 1.67262192369e-27, "kg", "Proton rest mass."
    ),
    "avogadro_constant": PhysicalConstant(
        "N_A", 6.02214076e23, "mol^-1", "Avogadro constant (exact)."
    ),
    "gas_constant": PhysicalConstant(
        "R", 8.314462618, "J mol^-1 K^-1", "Molar gas constant."
    ),
    "vacuum_permittivity": PhysicalConstant(
        "epsilon_0", 8.8541878128e-12, "F/m", "Vacuum electric permittivity."
    ),
    "vacuum_permeability": PhysicalConstant(
        "mu_0", 1.25663706212e-6, "N/A^2", "Vacuum magnetic permeability."
    ),
    "standard_gravity": PhysicalConstant(
        "g_0", 9.80665, "m/s^2", "Standard acceleration due to gravity (exact)."
    ),
    "stefan_boltzmann_constant": PhysicalConstant(
        "sigma", 5.670374419e-8, "W m^-2 K^-4", "Stefan-Boltzmann constant."
    ),
}


def get_constant(name: str) -> PhysicalConstant:
    """Return the constant registered under ``name``.

    Parameters
    ----------
    name:
        Lowercase registry key (e.g. ``"speed_of_light"``).

    Returns
    -------
    PhysicalConstant

    Raises
    ------
    KeyError
        If ``name`` is not registered.
    """
    try:
        return CONSTANTS[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown constant {name!r}. Available: {sorted(CONSTANTS)}"
        ) from exc
