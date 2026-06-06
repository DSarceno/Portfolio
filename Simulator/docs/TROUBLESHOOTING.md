# Troubleshooting

Solutions to common installation, generation, rendering, LaTeX and numerical
issues.

## Installation issues

**`pip install -e .` fails building NumPy/SciPy.**
Upgrade pip and use prebuilt wheels: `python -m pip install --upgrade pip`. On
Linux, ensure a C/Fortran toolchain (`build-essential`, `gfortran`) is present,
or prefer the conda environment (`environment.yml`).

**`simgen: command not found`.**
The console script is installed by `pip install -e .`. Ensure your virtual
environment is activated, or run via `python -m simgen.cli`.

**Wrong Python version.**
`simgen` needs Python ≥ 3.10. Check with `python --version`.

## Generation issues

**Generation always uses the offline library / ignores my prompt.**
This happens when no API key is configured or `anthropic` is not installed. Check
`simgen info` (`api_key_present` must be `True`) and install the extra:
`pip install -e ".[llm]"`. Use `--no-fallback` to force an error instead of the
silent fallback.

**`LLMUnavailableError: Claude API call failed`.**
Network/credentials/SDK error. Verify the key, your network, and that the
`anthropic` package is current. The model name comes from `SIMGEN_MODEL`.

**`SpecValidationError` from model output.**
The model returned an inconsistent spec (e.g. derivative count ≠ state-variable
count, or overlapping symbols). Re-run; if it persists, refine the prompt or file
the spec to the library manually (see EXTENSION_GUIDE.md).

**`RuntimeError: Error evaluating derivatives`.**
A derivative expression referenced an unknown symbol or used unsupported syntax
(e.g. `^` instead of `**`). Inspect `spec.json` and fix the expression.

## Rendering issues

**`manim: command not found`.**
Install the animation extra: `pip install -e ".[animation]"`.

**Manim error: `latex` / `dvisvgm` not found.**
Manim needs a LaTeX install for text. Install MiKTeX/MacTeX/TeX Live (see
INSTALLATION_GUIDE.md). On MiKTeX, allow on-the-fly package installation.

**Manim error: FFmpeg not found.**
Install FFmpeg and ensure it is on PATH (`ffmpeg -version`). The conda
environment includes it.

**Scene name not found.**
The class is `<PascalCaseSlug>Scene`. See the table in RENDERING_GUIDE.md or run
`grep "class .*Scene" generated/<slug>/scene.py`.

## LaTeX issues

**`pdflatex` missing packages.**
On MiKTeX, accept the install prompt (or pre-install with the MiKTeX console). On
TeX Live, install `texlive-latex-extra` and `texlive-science`.

**References/citations show as `[?]`.**
Run the full sequence: `pdflatex` → `bibtex` → `pdflatex` → `pdflatex`. The
per-project reports do not use bibtex (they list references inline) and only need
two `pdflatex` passes.

**Figure missing in the report.**
Per-project reports use `\IfFileExists`, so they compile even without figures.
Run `python plot.py` in the project directory first to include them.

## Numerical issues

**Solution blows up / `Non-finite state encountered`.**
The system is likely stiff or the step too large. In the spec, switch `method`
to `rk45`, `lsoda`, `radau`, or `bdf`, or increase `num_points`.

**Energy drifts in a "conservative" system.**
Explicit RK4 is not symplectic; over long times it drifts. Use a shorter horizon,
more points, or a symplectic integrator (velocity Verlet) for mechanical
systems. Note the relative-drift metric is ill-conditioned when `E0 ≈ 0`.

**Lyapunov estimate looks wrong.**
The estimator needs a long trajectory on the attractor. Discard transients,
increase `t_end`, and fit over the linear divergence region
(`fit_fraction` < 1.0). Downsample very long trajectories before estimating.

## Still stuck?

Run the test suite to localise the problem (`pytest -q -x`), enable verbose
logging (`simgen -v ...` or `SIMGEN_LOG_LEVEL=DEBUG`), and consult
[TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md).
