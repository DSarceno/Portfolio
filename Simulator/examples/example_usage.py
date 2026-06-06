"""End-to-end example of using the ``simgen`` Python API.

This script:

1. Lists the offline reference phenomena.
2. Generates a complete project for two of them.
3. Runs each simulation and prints a short numerical summary.
4. Renders the configured plots to PNG.
5. Demonstrates the prompt -> project flow (offline fallback if no API key).

Run with:

    python examples/example_usage.py

It requires only the core dependencies (numpy, scipy, matplotlib, jinja2) and
works fully offline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from simgen import generate_project, run_spec
from simgen.llm.generator import SimulationGenerator
from simgen.mathematics import (
    dominant_frequency,
    largest_lyapunov_exponent,
    relative_energy_drift,
)
from simgen.scenes.components import sample_trajectory
from simgen.simulations.library import get_spec, list_specs
from simgen.simulations.runner import energy_series

OUTPUT_DIR = Path("generated")


def _echo(message: str = "") -> None:
    """Write a line to stdout (this is a user-facing example script)."""
    sys.stdout.write(message + "\n")


def demo_library_overview() -> None:
    """Print the available offline reference phenomena."""
    _echo("Available reference phenomena:")
    for slug in list_specs():
        _echo(f"  - {slug}: {get_spec(slug).name}")
    _echo()


def demo_generate_and_run(slug: str) -> None:
    """Generate, run, analyse and plot a single reference phenomenon."""
    spec = get_spec(slug)
    _echo(f"=== {spec.name} ({slug}) ===")

    project = generate_project(spec, OUTPUT_DIR)
    _echo(f"  generated -> {project.root}")

    result = run_spec(spec)
    _echo(
        f"  integrated {result.n_steps} steps; "
        f"final state = {np.round(result.states[-1], 4).tolist()}"
    )

    # Energy conservation diagnostic (if the spec defines an energy).
    energy = energy_series(spec, result)
    if energy is not None and abs(energy[0]) > 1e-9:
        _echo(f"  relative energy drift = {relative_energy_drift(energy):.2e}")

    # Dominant frequency of the first state variable.
    first = result.symbols[0]
    _echo(
        f"  dominant frequency of {first}(t) = "
        f"{dominant_frequency(result.column(first), result.dt):.4f}"
    )

    # Render the plots.
    from simgen.visualization.plotting import render_all_plots

    figures = render_all_plots(spec, result, project.root)
    for fig in figures:
        _echo(f"  figure -> {fig}")
    _echo()


def demo_lyapunov() -> None:
    """Contrast a chaotic and a regular system via the Lyapunov exponent."""
    _echo("=== Lyapunov exponent: chaos vs. regularity ===")
    lorenz = run_spec(get_spec("lorenz-system"))
    sampled = sample_trajectory(lorenz.states, max_points=3000)
    eff_dt = (lorenz.t[-1] - lorenz.t[0]) / (sampled.shape[0] - 1)
    lle = largest_lyapunov_exponent(sampled, eff_dt, fit_fraction=0.5)
    _echo(f"  Lorenz largest Lyapunov exponent ~ {lle:.3f} (literature ~ 0.906)")

    sho = run_spec(get_spec("simple-harmonic-oscillator"))
    lle_reg = largest_lyapunov_exponent(sho.states, sho.dt, fit_fraction=0.5)
    _echo(f"  Harmonic oscillator largest Lyapunov exponent ~ {lle_reg:.3f} (~ 0)")
    _echo()


def demo_prompt_flow() -> None:
    """Demonstrate the prompt -> spec -> project flow (offline-safe)."""
    _echo("=== Prompt-driven generation (offline fallback if no API key) ===")
    generator = SimulationGenerator()
    spec, project = generator.generate_from_prompt(
        "A chaotic Lorenz-like convection system", OUTPUT_DIR
    )
    _echo(f"  prompt resolved to: {spec.name} ({spec.slug})")
    _echo(f"  project at: {project.root}")
    _echo()


def main() -> int:
    """Run the full demonstration."""
    demo_library_overview()
    demo_generate_and_run("lorenz-system")
    demo_generate_and_run("simple-harmonic-oscillator")
    demo_lyapunov()
    demo_prompt_flow()
    _echo("Done. Explore the 'generated/' directory for the output.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
