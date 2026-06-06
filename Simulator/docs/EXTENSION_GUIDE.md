# Extension Guide

How to extend `simgen`: add reference phenomena, numerical methods, plot/scene
kinds, and customise code generation.

## Adding a reference phenomenon (offline library)

1. Open [`src/simgen/simulations/library.py`](../src/simgen/simulations/library.py).
2. Write a builder function returning a `PhenomenonSpec`. Express the dynamics as
   a first-order system; convert any second-order equation to a velocity pair.

   ```python
   def _my_system() -> PhenomenonSpec:
       return PhenomenonSpec(
           name="My System",
           slug="my-system",
           summary="One-paragraph description.",
           state_variables=[StateVariable("x"), StateVariable("v")],
           parameters=[Parameter("k", 1.0, "stiffness")],
           derivatives=["v", "-k * x"],          # one expression per state var
           initial_conditions=[1.0, 0.0],
           t_start=0.0, t_end=20.0, num_points=4000, method="rk4",
           equations_latex=[r"\dot{x}=v", r"\dot{v}=-k x"],
           plots=[PlotSpec("phase2d", "Phase", x="x", y="v")],
           animation=AnimationSpec(kind="phase2d", x="x", y="v"),
           energy_expression="0.5*v**2 + 0.5*k*x**2",   # optional
       )
   ```

3. Register it in the `_BUILDERS` dict: `"my-system": _my_system`.
4. It is now available via `simgen list`, `simgen generate --from-library
   my-system`, and the test suite (the parametrised tests pick it up
   automatically).

### Rules for derivative expressions

- One expression per state variable, in the same order.
- Use only state symbols, parameter symbols, `t`, `np`, and the curated
  functions (`sin`, `cos`, `exp`, `sqrt`, `log`, `tanh`, `atan2`, ...).
- Use `**` for powers (not `^`).
- State and parameter symbols must be valid Python identifiers and disjoint.

## Adding a numerical method

1. For a single-step rule, add it to
   [`numerical_methods/ode_solvers.py`](../src/simgen/numerical_methods/ode_solvers.py).
2. Wire it into the dispatcher in
   [`numerical_methods/integrators.py`](../src/simgen/numerical_methods/integrators.py):
   add an entry to `AVAILABLE_METHODS` and a branch in `integrate()`.
3. Add the method name to `VALID_METHODS` in
   [`simulations/spec.py`](../src/simgen/simulations/spec.py) and to the
   `_SCIPY_METHOD` map in
   [`codegen/renderer.py`](../src/simgen/codegen/renderer.py) (use `None` for a
   fixed-step method and extend the generated `simulation.py.j2` template if
   needed).
4. Add a test in `tests/test_numerical_methods.py`.

## Adding a plot or animation kind

1. Add the kind to `VALID_PLOT_KINDS` in `simulations/spec.py`.
2. Implement rendering in
   [`visualization/plotting.py`](../src/simgen/visualization/plotting.py) and in
   the standalone template
   [`codegen/templates/plot.py.j2`](../src/simgen/codegen/templates/plot.py.j2).
3. Extend the Manim template
   [`codegen/templates/scene.py.j2`](../src/simgen/codegen/templates/scene.py.j2)
   if the new kind needs a new animation.

## Customising code generation

The generated files come from Jinja2 templates in
[`src/simgen/codegen/templates/`](../src/simgen/codegen/templates/). Edit these to
change the structure/style of generated `simulation.py`, `plot.py`, `scene.py`,
`report.tex` or `README.md`. The rendering context is assembled in
`codegen.renderer._build_context`; add fields there to expose new data to the
templates.

> **Note on LaTeX templates:** because Jinja's block delimiter is `{% %}`, avoid
> the literal sequences `{%` and `%}` in `.tex.j2` files (they collide). Use
> separate braces/lines instead.

## Extending the LLM schema

To let the model emit new spec fields:

1. Add the field to `PhenomenonSpec` (and its validation) in
   `simulations/spec.py`.
2. Add it to `SPEC_JSON_SCHEMA` in
   [`llm/schema.py`](../src/simgen/llm/schema.py).
3. Mention it in the system prompt in
   [`llm/prompts.py`](../src/simgen/llm/prompts.py).
4. Use it in the templates / renderer context.

## Running checks

```bash
pytest -q
black src tests
ruff check src tests
```
