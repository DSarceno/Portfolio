"""Simulation core: the spec model, base classes, runner and reference library."""

from simgen.simulations.base import (
    ODESimulation,
    Simulation,
    SimulationResult,
    build_simulation,
)
from simgen.simulations.runner import run_spec
from simgen.simulations.spec import (
    AnimationSpec,
    Parameter,
    PhenomenonSpec,
    PlotSpec,
    StateVariable,
)

__all__ = [
    "PhenomenonSpec",
    "StateVariable",
    "Parameter",
    "PlotSpec",
    "AnimationSpec",
    "Simulation",
    "ODESimulation",
    "SimulationResult",
    "build_simulation",
    "run_spec",
]
