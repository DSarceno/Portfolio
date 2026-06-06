"""Render a :class:`PhenomenonSpec` into a self-contained project directory.

The generated project depends only on NumPy, SciPy, Matplotlib (and Manim for
the animation). It does **not** import ``simgen``, so generated projects are
fully portable. Rendering is deterministic: identical specs yield identical
files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from simgen.reporting.latex import escape_latex
from simgen.simulations.spec import PhenomenonSpec
from simgen.utilities.io import ensure_directory, write_json, write_text
from simgen.utilities.logging_config import get_logger

logger = get_logger(__name__)

_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"

#: Mapping from spec method name to SciPy ``solve_ivp`` method (None = fixed step).
_SCIPY_METHOD = {
    "euler": None,
    "rk4": None,
    "rk45": "RK45",
    "dopri5": "DOP853",
    "radau": "Radau",
    "bdf": "BDF",
    "lsoda": "LSODA",
}


@dataclass
class GeneratedProject:
    """Paths of the files produced for one phenomenon.

    Attributes
    ----------
    root:
        Project directory.
    simulation_path, plot_path, scene_path, report_path, readme_path, spec_path:
        Absolute paths of the generated artifacts.
    """

    root: Path
    simulation_path: Path
    plot_path: Path
    scene_path: Path
    report_path: Path
    readme_path: Path
    spec_path: Path

    def as_dict(self) -> dict[str, str]:
        """Return a string mapping of artifact name -> path."""
        return {
            "root": str(self.root),
            "simulation": str(self.simulation_path),
            "plot": str(self.plot_path),
            "scene": str(self.scene_path),
            "report": str(self.report_path),
            "readme": str(self.readme_path),
            "spec": str(self.spec_path),
        }


def _environment() -> Environment:
    """Construct the Jinja2 environment used for rendering."""
    return Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )


def _build_context(spec: PhenomenonSpec) -> dict[str, object]:
    """Assemble the template rendering context from a spec."""
    assignments = [f"    {sym} = state[{i}]" for i, sym in enumerate(spec.state_symbols)]
    param_lines = [
        f"{p.symbol} = {p.value!r}  # {p.description}" if p.description else f"{p.symbol} = {p.value!r}"
        for p in spec.parameters
    ]
    plots_repr = repr(
        [
            {"kind": p.kind, "title": p.title, "x": p.x, "y": p.y, "z": p.z}
            for p in spec.plots
        ]
    )
    animation_repr = repr(
        {
            "kind": spec.animation.kind,
            "title": spec.animation.title,
            "description": spec.animation.description,
            "x": spec.animation.x,
            "y": spec.animation.y,
            "z": spec.animation.z,
            "trail": spec.animation.trail,
        }
    )
    param_table = [
        {
            "symbol": p.symbol,
            "value": f"{p.value:.6g}",
            "unit": escape_latex(p.unit),
            "description": escape_latex(p.description),
        }
        for p in spec.parameters
    ]
    state_table = [
        {
            "symbol": sv.symbol,
            "name": escape_latex(sv.name),
            "unit": escape_latex(sv.unit),
            "description": escape_latex(sv.description),
        }
        for sv in spec.state_variables
    ]
    return {
        "spec": spec,
        "class_name": spec.class_name,
        "name_tex": escape_latex(spec.name),
        "summary_tex": escape_latex(spec.summary),
        "theory_tex": escape_latex(spec.theory),
        "domain_tex": escape_latex(spec.domain),
        "method_tex": escape_latex(spec.method),
        "param_table": param_table,
        "state_table": state_table,
        "state_symbols": spec.state_symbols,
        "assignments": assignments,
        "param_lines": param_lines,
        "derivatives": spec.derivatives,
        "scipy_method": _SCIPY_METHOD[spec.method],
        "is_fixed_step": _SCIPY_METHOD[spec.method] is None,
        "plots": spec.plots,
        "plots_repr": plots_repr,
        "animation": spec.animation,
        "animation_repr": animation_repr,
    }


def generate_project(
    spec: PhenomenonSpec,
    output_dir: str | Path = "generated",
) -> GeneratedProject:
    """Generate a complete project directory for ``spec``.

    Parameters
    ----------
    spec:
        A validated phenomenon specification.
    output_dir:
        Parent directory; the project is written to ``output_dir/<slug>``.

    Returns
    -------
    GeneratedProject
        Paths of the generated files.
    """
    spec.validate()
    env = _environment()
    context = _build_context(spec)

    root = ensure_directory(Path(output_dir) / spec.slug)
    logger.info("Generating project for %r at %s", spec.slug, root)

    simulation_path = write_text(
        root / "simulation.py", env.get_template("simulation.py.j2").render(**context)
    )
    plot_path = write_text(
        root / "plot.py", env.get_template("plot.py.j2").render(**context)
    )
    scene_path = write_text(
        root / "scene.py", env.get_template("scene.py.j2").render(**context)
    )
    report_path = write_text(
        root / "report.tex", env.get_template("report.tex.j2").render(**context)
    )
    readme_path = write_text(
        root / "README.md", env.get_template("readme.md.j2").render(**context)
    )
    spec_path = write_json(root / "spec.json", spec.to_dict())

    return GeneratedProject(
        root=root,
        simulation_path=simulation_path,
        plot_path=plot_path,
        scene_path=scene_path,
        report_path=report_path,
        readme_path=readme_path,
        spec_path=spec_path,
    )
