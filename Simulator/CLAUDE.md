# CLAUDE.md — `simgen` project memory

Compact, high-signal context. Read this first; inspect other files only for a specific task.

## PROJECT_IDENTITY
- **Name:** `simgen` (repo dir `Simulator`). Package `src/simgen`, console script `simgen`.
- **Purpose:** prompt → complete runnable mini-project (simulation code + plots + Manim animation + LaTeX report).
- **Phenomenon:** open-ended (any math/physics ODE-expressible system); NOT a single fixed topic.
- **Edu objectives:** teach modelling, numerical integration, simulation, visualization, analysis.
- **Outputs per phenomenon:** `simulation.py`, `plot.py`, `scene.py`, `report.tex`, `README.md`, `spec.json`.
- **Audience:** university/grad courses, sci-comm/YouTube, self-study, research demos.
- **Interfaces:** CLI + Python API. **Engine:** LLM-backed (Claude API) with offline library fallback.

## PROJECT_MISSION
- Automate the *construction* of educational simulation projects from a one-line prompt.
- Solve: turning natural-language phenomenon descriptions into consistent, reproducible code/figures/video/report.
- Success = one validated `PhenomenonSpec` drives all four artifacts; generated code runs standalone; reports compile; tests pass.

## ARCHITECTURE_OVERVIEW
`PhenomenonSpec` is the single source of truth shared by every consumer.
```
src/simgen/
├─ config/        Settings from env/.env/defaults (get_settings, cached)
├─ utilities/     logging_config (stderr, namespaced), io (slugify, json/text/array)
├─ numerical_methods/  ode_solvers (euler/rk4/verlet), integrators (integrate dispatcher+SciPy), linalg
├─ physics/       constants (CODATA), units (SI dimensional algebra)
├─ mathematics/   analysis (power_spectrum, autocorr, energy drift, Lyapunov/Rosenstein)
├─ simulations/   spec (PhenomenonSpec+validate+JSON), base (compiler→ODESimulation, SimulationResult),
│                 runner (run_spec, energy_series), library (5 reference specs)
├─ visualization/ plotting (matplotlib Agg; time_series/phase2d/phase3d)
├─ scenes/        components (MANIM_AVAILABLE guard, sample_trajectory, fit_points_to_frame), base_scene
├─ reporting/     latex (escape_latex, summarize_result)
├─ llm/           schema (SPEC_JSON_SCHEMA, spec_from_json), prompts, client (Anthropic tool-use), generator
├─ codegen/       renderer (generate_project → files), templates/*.j2
└─ cli.py         argparse: generate / list / info
```
- **Dep direction:** llm/codegen → simulations → numerical_methods/physics/mathematics → utilities. No reverse imports.
- **Optional deps imported lazily/guarded:** `anthropic` (llm.client), `manim` (scenes), `scipy` (integrators, only for adaptive methods). `import simgen` never requires them.

## REPOSITORY_MAP
| Path | Purpose | Modify when | Risk |
|---|---|---|---|
| `src/simgen/simulations/spec.py` | Spec model + validation | add spec fields/methods | breaks codegen+llm+tests |
| `src/simgen/simulations/base.py` | spec→sim compiler, restricted eval | change exec/integration semantics | security/correctness |
| `src/simgen/simulations/library.py` | 5 reference specs (offline + test fixtures) | add phenomenon | parametrized tests pick up |
| `src/simgen/codegen/renderer.py` + `templates/*.j2` | generates standalone artifacts | change generated output | regen+compile breakage |
| `src/simgen/numerical_methods/integrators.py` | `integrate()` + method registry | add solver | sync spec.VALID_METHODS + renderer._SCIPY_METHOD |
| `src/simgen/llm/*` | prompt→spec via Claude | change schema/prompt | spec parse failures |
| `src/simgen/cli.py` | CLI | add command/flag | UX |
| `run_all.bat` | Windows full-pipeline launcher (venv `simu`) | change run UX | low |
| `reports/report.tex`,`bibliography.bib` | umbrella report | docs/refs | latex build |
| `tests/` | 51 tests, offline | any code change | CI gate |
| `pyproject.toml` | packaging, extras, tool cfg | deps/entrypoints | install |
| `.env.example` | config template ONLY (no real secrets) | new setting | secret leak (tracked file) |

## EXECUTION_FLOW
```
prompt / --from-library
   ↓ llm.generator.SimulationGenerator.spec_from_prompt
   ↓   (Claude tool-use → schema.spec_from_json) | fallback: library keyword match
PhenomenonSpec (validated)
   ↓ codegen.renderer.generate_project  → simulation.py/plot.py/scene.py/report.tex/README.md/spec.json
   ↓ (optional) simulations.runner.run_spec → SimulationResult
   ↓ (optional) visualization.render_all_plots → figure_*.png
   ↓ (optional) manim scene.py → mp4 ; pdflatex report.tex → pdf
Final outputs in generated/<slug>/
```

## MANIM_SYSTEM
- **Generated scene:** one class `<PascalCaseSlug>Scene` per project (`spec.class_name + "Scene"`). 3D→`ThreeDScene` (phase3d), else `Scene`.
- **Standalone:** `scene.py` inserts its dir on `sys.path`, `from simulation import simulate, STATE_SYMBOLS`; depends on manim only (not simgen).
- **Pipeline in scene:** simulate → `_column` → `_downsample`(≤600 pts) → `_fit`(center+scale to frame) → `VMobject.set_points_smoothly` + moving `Dot`; 3D adds ambient camera rotation.
- **Reusable (framework, optional manim):** `scenes/components.py` `TrajectoryScene` (only defined if `MANIM_AVAILABLE`); pure helpers `sample_trajectory`, `fit_points_to_frame` always available.
- **Render:** `manim -pql|-qh generated/<slug>/scene.py <Class>Scene`. Tests validate scenes by byte-compile (no render needed).

## SCIENTIFIC_MODEL
- **Formalism:** all systems as first-order ODE `dy/dt=f(t,y)`. 2nd-order→velocity pair; PDE→method-of-lines.
- **Spec derivatives:** Python expr strings, one per state var, in order; may use state syms, params, `t`, `np`, curated fns (sin/cos/exp/sqrt/log/tanh/atan2/...); `**` for powers.
- **Reference library:** lorenz (chaos, 3D), double-pendulum (chaotic, energy expr), simple-harmonic-oscillator (analytic+energy benchmark), duffing-oscillator (forced chaos, rk45), van-der-pol-oscillator (limit cycle, rk45).
- **Validation:** analytic match (SHO `cos ωt`, decay `e^-t`), energy drift (SHO ≈5.6e-11), Lyapunov (Lorenz≈0.89 vs lit 0.906) > regular.
- **Note:** relative energy-drift metric ill-conditioned when E0≈0 (double pendulum horizontal start) — use SHO for conservation tests.

## NUMERICAL_METHODS
- **Methods (`AVAILABLE_METHODS`):** `euler`,`rk4` (fixed, in-package); `rk45`,`dopri5`,`radau`,`bdf`,`lsoda` (SciPy `solve_ivp`). `radau/bdf/lsoda`=stiff. velocity_verlet for conservative 2nd-order.
- **Selection:** rk4 default (smooth nonstiff); rk45/lsoda adaptive; radau/bdf stiff.
- **Eval safety:** derivative exprs `compile(...,'eval')` once, run in restricted namespace (numpy+math only, `__builtins__={}`) — no arbitrary code exec.
- **Stability/conv:** explicit RK4 O(h^4), not symplectic (drifts long-term); non-finite state → integrator returns success=False, truncated trajectory.
- **linalg:** spectral_radius, power_iteration, condition_number, is_symmetric.

## DATA_FLOW
```
prompt → llm.generator/client/schema → PhenomenonSpec → codegen.renderer → files
                                              ↓ simulations.runner.run_spec → SimulationResult (t,states,symbols)
                                              ↓ visualization.plotting → PNG ; reporting.latex feeds report.tex
```
Files: `llm/generator.py`,`llm/client.py`,`llm/schema.py` → `simulations/spec.py` → `codegen/renderer.py`+`templates/` → `simulations/runner.py`+`base.py` → `visualization/plotting.py` / `reporting/latex.py`.

## CODEBASE_RULES
**Coding:** type hints everywhere; numpydoc-style docstrings; `get_logger(__name__)` (stderr); explicit error handling; NO `print` in package code (CLI uses `_echo`→stdout; generated scripts may print in `__main__`). black line-length 100, ruff (E,F,I,W,B,UP).
**Architecture:** respect dep direction (no reverse imports); optional deps stay lazy/guarded; generated artifacts must stay standalone (no `simgen` import); `PhenomenonSpec` is the contract.
**Testing:** all tests offline+deterministic; new phenomenon/method/plot needs a test; generated Python must byte-compile.
**Docs:** update `docs/` + `REPOSITORY_COMPLETENESS_REPORT.md` when adding capabilities; keep `EXTENSION_GUIDE.md` in sync.

## DEVELOPMENT_WORKFLOWS
- **New animation kind:** add to `VALID_PLOT_KINDS` (spec.py) → render in `visualization/plotting.py` + `templates/plot.py.j2` → extend `templates/scene.py.j2` → test.
- **New simulation/phenomenon:** add builder in `library.py`, register in `_BUILDERS` (parametrized tests auto-include).
- **New numerical method:** `ode_solvers.py` (if fixed-step) → `integrators.py` (AVAILABLE_METHODS+branch) → `spec.VALID_METHODS` → `renderer._SCIPY_METHOD` (+template if fixed) → test.
- **New physics/math model:** add module fn under `physics/` or `mathematics/`, export in subpkg `__init__`, add test.
- **New docs:** add `docs/X.md`, link from README/PROJECT_STRUCTURE.
- **New tests:** drop `tests/test_*.py`; keep offline.

## BUILD_COMMANDS
```bash
python -m venv simu              # canonical venv name (gitignored); activate before install
pip install -e ".[dev]"          # core+tests ; extras: llm, animation, all
pytest -q                        # 51 tests, offline
ruff check src tests ; black src tests
simgen list ; simgen info
simgen generate "Lorenz attractor" --run --plot        # LLM or fallback
simgen generate --from-library double-pendulum --run --plot
python examples/example_usage.py
manim -qh generated/<slug>/scene.py <Class>Scene
cd generated/<slug> && python plot.py && pdflatex report.tex && pdflatex report.tex
cd reports && pdflatex report.tex && bibtex report && pdflatex report.tex && pdflatex report.tex
run_all.bat [-l <slug> | "<prompt>"]   # Windows: one-shot generate→run→plot(+manim/latex if present)
```

## CRITICAL_FILES
```
src/simgen/simulations/spec.py     contract; validation
src/simgen/simulations/base.py     compiler + restricted eval
src/simgen/codegen/renderer.py     + templates/*.j2  (generated output)
src/simgen/numerical_methods/integrators.py  method registry
src/simgen/llm/schema.py, prompts.py         model↔spec
pyproject.toml                     deps/entrypoint/extras
reports/report.tex, bibliography.bib
tests/                             behavior lock
```

## DEPENDENCY_SUMMARY
| Package | Why | Required? |
|---|---|---|
| numpy | arrays, integration | core |
| scipy | adaptive/stiff `solve_ivp` | core (lazy import) |
| matplotlib | plots (Agg) | core |
| jinja2 | codegen templates | core |
| anthropic | LLM spec generation | extra `llm` |
| manim | animations | extra `animation` |
| pytest(+cov) | tests | extra `dev` |

## TESTING_STRATEGY
- **Files:** test_numerical_methods, test_math_models, test_physics_models, test_simulations, test_scenes (51 tests).
- **Unit:** integrators, linalg, units/constants, spec validation/roundtrip.
- **Numerical validation:** analytic match, RK4>Euler, energy conservation.
- **Scientific:** Lyapunov chaotic>regular.
- **Rendering validation:** generated simulation/plot/scene byte-compile; generated sim importable & correct.
- **New features:** must add tests; keep deterministic & offline (no API/manim/latex).

## KNOWN_LIMITATIONS
- LLM may emit plausible-but-imperfect specs → rely on validation + offline library.
- Explicit RK4 not symplectic → energy drift over long horizons (use stiff/adaptive or verlet).
- Chaos: only statistical/geometric features robust beyond ~1/λmax horizon.
- Lyapunov estimator (Rosenstein) is O(n²) neighbor search; downsample long trajectories; needs transient removal.
- No native FEM/spectral PDE modules yet (method-of-lines pattern only).
- Manim/LaTeX/FFmpeg are system-dependent; not exercised in tests.
- `relative_energy_drift` ill-conditioned when E0≈0.

## FUTURE_EXTENSION_POINTS
- Native PDE templates (finite-difference, spectral); auto stiffness detection/method selection.
- Parameter sweeps / bifurcation diagrams; two-trajectory divergence scenes; vector-field/slope-field animations.
- More library phenomena; richer report sections; caching of LLM specs.

## SESSION_BOOTSTRAP
- **Mission:** prompt→consistent simulation project; spec is the contract.
- **Architectural assumptions:** src-layout package; one-way deps; optional deps lazy; generated code standalone (no simgen import).
- **Scientific assumptions:** everything is first-order ODE; derivative exprs in restricted numpy namespace.
- **Dev priorities:** keep tests green/offline; keep generated artifacts compiling; maintain spec↔schema↔renderer consistency.
- **Preferred patterns:** dataclasses + validation; Jinja templates for output; `get_logger`; numpydoc docstrings; `_echo` for CLI stdout.
- **Common pitfalls:** in `.tex.j2` avoid literal `{%`/`%}` (Jinja delim clash); adding a method needs 4 synced edits (see workflow); param/state symbols must be valid identifiers & disjoint; don't import manim/anthropic at module top level.
- **Before changes:** read spec.py + base.py + renderer.py; run `pytest -q`.
- **After changes:** `pytest -q`, `ruff check`, `black`, regenerate+byte-compile a sample, update docs/completeness report.

## SESSION_START_CHECKLIST
```
[ ] Read PROJECT_IDENTITY + ARCHITECTURE_OVERVIEW + EXECUTION_FLOW
[ ] If touching output: read MANIM_SYSTEM / codegen templates
[ ] Identify affected modules (respect dep direction)
[ ] Run pytest -q (baseline green)
[ ] Implement; add/adjust tests
[ ] ruff + black; regenerate & byte-compile a sample project
[ ] Update docs + REPOSITORY_COMPLETENESS_REPORT.md
[ ] Re-run pytest -q
```
