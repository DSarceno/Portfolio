# Simulator — Prompt-Driven Scientific Simulation Generator (`simgen`)

> Describe a mathematical or physical phenomenon in a single sentence and get back
> a complete, runnable mini-project: numerical simulation code, publication-quality
> plots, a Manim animation, and a LaTeX report.

`simgen` is an **LLM-backed generator**. You write a short prompt such as

```text
"The Lorenz system, a 3D chaotic attractor, with sigma=10, rho=28, beta=8/3."
```

and the engine produces a self-contained project:

```text
generated/lorenz-system/
├── simulation.py     # governing equations + numerical integration
├── plot.py           # matplotlib figures (phase space + time series)
├── scene.py          # Manim Community scene
├── report.tex        # compilable LaTeX report
├── spec.json         # the structured specification used to build everything
└── README.md         # how to run this generated project
```

Everything is built on a reusable framework of numerical solvers, plotting
utilities, Manim components and LaTeX helpers, so the generated code is real,
readable Python — not opaque blobs.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Repository Structure](#repository-structure)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Rendering](#rendering)
7. [Report Compilation](#report-compilation)
8. [Examples](#examples)
9. [Troubleshooting](#troubleshooting)
10. [References](#references)

---

## Project Overview

Traditional educational simulation projects target a *single* phenomenon. This
project instead targets the **act of building such a project**. Given a free-text
prompt, `simgen`:

1. Sends the prompt to the Claude API and asks for a **structured specification**
   (`PhenomenonSpec`) describing state variables, parameters, governing
   equations, initial conditions, integration settings, plots, an animation
   description and report text.
2. Validates that specification against a strict schema.
3. **Renders** the specification into concrete artifacts using Jinja2 templates
   and the framework's solver / plotting / animation / reporting modules.
4. Optionally **runs** the generated simulation, **plots** the results, **renders**
   the Manim animation and **compiles** the LaTeX report.

When no API key is available, `simgen` falls back to a built-in **library of
reference specifications** (Lorenz system, double pendulum, simple harmonic
oscillator, damped driven oscillator, Van der Pol oscillator) so the whole
pipeline — including the test suite — runs fully offline and deterministically.

See [`docs/SCIENTIFIC_BACKGROUND.md`](docs/SCIENTIFIC_BACKGROUND.md) for the
science and [`docs/TECHNICAL_DOCUMENTATION.md`](docs/TECHNICAL_DOCUMENTATION.md)
for the architecture.

## Features

- **Prompt → simulation**: natural-language description to runnable code.
- **Four artifact types per phenomenon**: simulation code, plots, Manim scene, LaTeX report.
- **Reusable numerical core**: explicit/implicit ODE integrators (Euler, RK4, RK45, Velocity Verlet) plus a SciPy bridge.
- **Offline fallback library**: five validated reference systems, no API key required.
- **Deterministic codegen**: the same `PhenomenonSpec` always yields the same files.
- **Dual interface**: a `simgen` command-line tool *and* an importable Python API.
- **Fully tested**: pytest suite covering numerical methods, physics, mathematics, simulations and generated scenes.
- **Production hygiene**: type hints, docstrings, logging, error handling, black/ruff configuration.

## Repository Structure

```text
Simulator/
├── README.md                       # this file
├── LICENSE                         # MIT
├── run_all.bat                     # Windows one-command full-pipeline launcher
├── requirements.txt                # core dependencies
├── environment.yml                 # conda environment
├── pyproject.toml                  # packaging + tooling config
├── .env.example                    # configuration template
├── docs/                           # nine guides + diagrams
├── src/simgen/                     # the package
│   ├── cli.py                      # command-line entry point
│   ├── config/                     # settings
│   ├── utilities/                  # logging, IO helpers
│   ├── numerical_methods/          # integrators, ODE solvers, linear algebra
│   ├── physics/                    # constants, units
│   ├── mathematics/                # analysis helpers (Lyapunov, FFT, fits)
│   ├── simulations/                # spec model, base classes, runner, library
│   ├── visualization/              # matplotlib plotting
│   ├── scenes/                     # Manim base scene + components
│   ├── reporting/                  # LaTeX report generation
│   ├── llm/                        # Claude client, prompts, schema, generator
│   └── codegen/                    # Jinja2 templates + renderer
├── assets/                         # images, audio, data, fonts
├── reports/                        # top-level LaTeX report + bibliography
├── tests/                          # pytest suite
├── examples/                       # example_usage.py
├── rendered_videos/                # Manim output target
└── REPOSITORY_COMPLETENESS_REPORT.md
```

A directory-by-directory explanation lives in
[`docs/PROJECT_STRUCTURE.md`](docs/PROJECT_STRUCTURE.md).

## Installation

Quick version (full instructions in
[`docs/INSTALLATION_GUIDE.md`](docs/INSTALLATION_GUIDE.md)):

```bash
# 1. Create and activate a virtual environment (named `simu`)
python -m venv simu
# Windows PowerShell:
simu\Scripts\Activate.ps1
# Linux/macOS:
source simu/bin/activate

# 2. Install the package (editable) with dev tools
pip install -e ".[dev]"

# 3. (Optional) LLM + animation support
pip install -e ".[all]"
```

The core install (numpy, scipy, matplotlib, jinja2) is enough to run
simulations, plots, the offline library and the test suite. The `llm` extra
adds `anthropic`; the `animation` extra adds `manim` (which itself needs a
system FFmpeg and LaTeX installation — see the installation guide).

## Usage

### Command line

```bash
# Generate a project from a prompt (uses the Claude API if a key is set,
# otherwise matches the offline library).
simgen generate "The Lorenz attractor with sigma=10, rho=28, beta=8/3" --run --plot

# List the built-in offline reference phenomena
simgen list

# Generate directly from a library phenomenon (no API needed)
simgen generate --from-library double-pendulum --run --plot

# Show resolved configuration
simgen info
```

### One-command full run (Windows)

The [`run_all.bat`](run_all.bat) launcher in the project root runs the whole
pipeline in one shot: it activates the `simu` virtual environment, ensures
`simgen` is installed, then **generates → runs → plots** the project and — when
the system tools are available — **renders the Manim animation** and **compiles
the LaTeX report**. If FFmpeg or LaTeX are missing, those steps are skipped
gracefully (the rest still runs).

```bat
run_all.bat                                       REM default offline demo (lorenz-system)
run_all.bat -l double-pendulum                    REM any built-in library phenomenon
run_all.bat "The Lorenz attractor, sigma=10"      REM free-text prompt (needs API key; quote it)
```

It expects the virtual environment at `simu\`. Free-text prompts must be quoted;
`-l <slug>` and the no-argument default are fully offline.

### Python API

```python
from simgen import generate_project, run_spec
from simgen.simulations.library import get_spec

# Offline: build a project from a known reference spec
spec = get_spec("lorenz-system")
project = generate_project(spec, output_dir="generated")
print(project.simulation_path)

# Run the simulation in-process and inspect the result
result = run_spec(spec)
print(result.states.shape, result.t[-1])
```

LLM-backed generation:

```python
from simgen.llm.generator import SimulationGenerator

gen = SimulationGenerator()                     # reads ANTHROPIC_API_KEY
spec = gen.spec_from_prompt("A damped driven pendulum showing period doubling")
project = gen.generate(spec)
```

## Rendering

Render a generated Manim scene (see [`docs/RENDERING_GUIDE.md`](docs/RENDERING_GUIDE.md)):

```bash
# Low quality preview
manim -pql generated/lorenz-system/scene.py LorenzSystemScene
# High quality
manim -qh generated/lorenz-system/scene.py LorenzSystemScene
```

## Report Compilation

```bash
cd generated/lorenz-system
pdflatex report.tex
pdflatex report.tex     # second pass resolves references
```

The top-level umbrella report in [`reports/report.tex`](reports/report.tex)
documents the whole framework and is compiled with:

```bash
cd reports
pdflatex report.tex
bibtex report
pdflatex report.tex
pdflatex report.tex
```

## Examples

A complete, runnable walkthrough is in
[`examples/example_usage.py`](examples/example_usage.py):

```bash
python examples/example_usage.py
```

It generates projects for several library phenomena, runs the simulations,
produces plots and prints a short numerical summary.

## Troubleshooting

Common issues and fixes are collected in
[`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md). Highlights:

- **`anthropic` not installed / no API key** → generation falls back to the
  offline library; install `simgen[llm]` and set `ANTHROPIC_API_KEY` for
  arbitrary prompts.
- **Manim cannot find LaTeX/FFmpeg** → see the installation guide.
- **Stiff system blows up** → use the `RK45` or `LSODA` (SciPy) method in the spec.

## References

A full bibliography (books, papers, online sources) is in
[`reports/bibliography.bib`](reports/bibliography.bib) and discussed in
[`docs/SCIENTIFIC_BACKGROUND.md`](docs/SCIENTIFIC_BACKGROUND.md). Key starting points:

- Strogatz, *Nonlinear Dynamics and Chaos* (2015).
- Press et al., *Numerical Recipes* (2007).
- Hairer, Nørsett & Wanner, *Solving Ordinary Differential Equations I* (1993).
- The Manim Community documentation: <https://docs.manim.community/>.

---

Licensed under the [MIT License](LICENSE).
