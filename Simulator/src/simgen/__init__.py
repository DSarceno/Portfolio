"""simgen — prompt-driven generator of math/physics simulations.

Public API
----------
- :func:`generate_project` — render a :class:`PhenomenonSpec` to a project dir.
- :func:`run_spec` — compile and integrate a spec, returning a result.
- :func:`generate_from_prompt` — prompt -> spec -> project (LLM-backed).
- :class:`PhenomenonSpec` and friends — the specification data model.
- :mod:`simgen.simulations.library` — offline reference phenomena.

Heavy/optional integrations (Matplotlib plotting, Manim scenes, the Anthropic
client) are imported lazily by their submodules so that ``import simgen`` stays
light and never fails on a missing optional dependency.
"""

from __future__ import annotations

from simgen.codegen.renderer import GeneratedProject, generate_project
from simgen.llm.generator import SimulationGenerator, generate_from_prompt
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

__version__ = "0.1.0"

__all__ = [
    "__version__",
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
    "GeneratedProject",
    "generate_project",
    "SimulationGenerator",
    "generate_from_prompt",
]
