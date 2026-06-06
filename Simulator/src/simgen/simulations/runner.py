"""Convenience entry points for running a phenomenon specification."""

from __future__ import annotations

import numpy as np

from simgen.simulations.base import SimulationResult, build_simulation
from simgen.simulations.spec import PhenomenonSpec
from simgen.utilities.logging_config import get_logger

logger = get_logger(__name__)


def run_spec(spec: PhenomenonSpec) -> SimulationResult:
    """Compile and run a :class:`PhenomenonSpec`, returning its trajectory.

    Parameters
    ----------
    spec:
        A validated phenomenon specification.

    Returns
    -------
    SimulationResult
    """
    logger.info("Running simulation %r with method %s", spec.slug, spec.method)
    simulation = build_simulation(spec)
    return simulation.run()


def energy_series(spec: PhenomenonSpec, result: SimulationResult) -> np.ndarray | None:
    """Compute the energy time series for ``result`` if the spec defines energy.

    Parameters
    ----------
    spec:
        The specification (must contain an ``energy_expression`` to be useful).
    result:
        A previously computed :class:`SimulationResult`.

    Returns
    -------
    numpy.ndarray | None
        Energy at each time point, or ``None`` if no energy expression exists.
    """
    if spec.energy_expression is None:
        return None
    simulation = build_simulation(spec)
    energies = np.array(
        [simulation.energy(state) for state in result.states], dtype=float
    )
    return energies
