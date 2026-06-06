# Technical Documentation

This document explains the architecture, module responsibilities, data flow,
numerical methods and rendering workflow of `simgen`.

## Architecture overview

`simgen` is organised into three layers:

1. **Interpretation layer** (`simgen.llm`): turns a natural-language prompt into
   a validated `PhenomenonSpec` using the Claude API, with an offline library
   fallback.
2. **Core scientific layer** (`simgen.numerical_methods`, `simgen.physics`,
   `simgen.mathematics`, `simgen.simulations`): integrators, constants/units,
   analysis tools, and the spec/simulation model.
3. **Generation & presentation layer** (`simgen.codegen`,
   `simgen.visualization`, `simgen.scenes`, `simgen.reporting`): renders the spec
   into standalone code, figures, animations and reports.

```
prompt ──▶ llm.generator ──▶ PhenomenonSpec ──▶ codegen.renderer ──▶ project files
                 │                  │
                 │ (fallback)       ├──▶ simulations.runner ──▶ SimulationResult
                 ▼                  │
        simulations.library        ├──▶ visualization.plotting ──▶ PNGs
                                    └──▶ reporting.latex ──▶ report.tex
```

See [diagrams/architecture.md](diagrams/architecture.md) and
[diagrams/data_flow.md](diagrams/data_flow.md).

## Module responsibilities

| Module | Responsibility |
|--------|----------------|
| `config.settings` | Resolve configuration from env / `.env` / defaults. |
| `utilities.logging_config` | Namespaced logging to stderr. |
| `utilities.io` | Slugify, JSON/text/array IO, directory helpers. |
| `numerical_methods.ode_solvers` | Single-step rules (Euler, RK4, velocity Verlet). |
| `numerical_methods.integrators` | `integrate()` dispatcher + SciPy bridge. |
| `numerical_methods.linalg` | Spectral radius, power iteration, conditioning. |
| `physics.constants` | CODATA physical constants with metadata. |
| `physics.units` | Lightweight SI dimensional algebra. |
| `mathematics.analysis` | Spectra, autocorrelation, energy drift, Lyapunov. |
| `simulations.spec` | `PhenomenonSpec` data model + validation + JSON. |
| `simulations.base` | `Simulation`/`ODESimulation`, spec compiler, results. |
| `simulations.runner` | `run_spec()`, `energy_series()`. |
| `simulations.library` | Five validated reference specs. |
| `visualization.plotting` | Matplotlib rendering of plots. |
| `scenes.components` | Manim-independent helpers + reusable scene. |
| `reporting.latex` | LaTeX escaping + result summaries. |
| `codegen.renderer` | Jinja2 rendering of standalone artifacts. |
| `llm.schema` | JSON schema + parsing/validation of model output. |
| `llm.prompts` | System/user prompt construction. |
| `llm.client` | Anthropic API wrapper (tool use). |
| `llm.generator` | Orchestration + fallback. |
| `cli` | Argparse CLI. |

## Data flow

1. The user provides a prompt (CLI/API).
2. `llm.generator.SimulationGenerator.spec_from_prompt` calls
   `llm.client.ClaudeClient.generate_spec_dict`, which uses Anthropic tool-use to
   return a dict matching `llm.schema.SPEC_JSON_SCHEMA`.
3. `llm.schema.spec_from_json` parses and validates it into a `PhenomenonSpec`.
   On any failure (no key, no dependency, validation error) the generator falls
   back to `simulations.library`.
4. `codegen.renderer.generate_project` renders the spec into a project directory.
5. Optionally, `simulations.runner.run_spec` integrates the system and
   `visualization.plotting.render_all_plots` produces figures.

The `PhenomenonSpec` is the single source of truth shared by all consumers, so
code, figures, animation and report are always mutually consistent.

## Numerical methods

The state is always a first-order ODE system `dy/dt = f(t, y)`. Derivative
expressions from the spec are compiled once (`compile(..., "eval")`) and
evaluated in a restricted namespace exposing only NumPy and curated math
functions — no Python builtins — which prevents arbitrary code execution.

Available integration methods (`numerical_methods.AVAILABLE_METHODS`):

- `euler`, `rk4` — fixed-step, implemented in-package.
- `rk45`, `dopri5`, `radau`, `bdf`, `lsoda` — SciPy `solve_ivp` adaptive methods
  (the last three handle stiff systems).

Velocity-Verlet is available for second-order conservative systems where
long-time energy conservation matters.

## Rendering workflow

Generated artifacts are **standalone**: `simulation.py` depends only on
NumPy/SciPy; `plot.py` imports the local `simulation` module; `scene.py` imports
the local `simulation` module and Manim. This means a generated project can be
zipped and run elsewhere without installing `simgen`. See
[diagrams/rendering_pipeline.md](diagrams/rendering_pipeline.md) and
[RENDERING_GUIDE.md](RENDERING_GUIDE.md).

## Error handling & logging

All modules log via `utilities.logging_config.get_logger`. Logging goes to
`stderr`; the CLI's user-facing output goes to `stdout`. The LLM client
normalises SDK/network errors into `LLMUnavailableError`, which the generator
catches to trigger the offline fallback.

## Testing strategy

The suite (`tests/`) runs fully offline and deterministically:
analytic comparisons (SHO, exponential decay), energy conservation, a Lyapunov
sanity check (chaotic > regular), unit/constant checks, spec round-tripping, and
byte-compilation of every generated Python artifact. See
[../tests](../tests) and `pytest -q`.
