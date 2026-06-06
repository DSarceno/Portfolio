# User Guide

This guide shows how to use `simgen` day to day: generating projects, running
simulations, rendering videos and reports, and understanding the outputs.

## Quick Start

```bash
# 1. Install (see INSTALLATION_GUIDE.md)
pip install -e ".[dev]"

# 2. Generate a project from the offline library, run it, and plot
simgen generate --from-library double-pendulum --run --plot

# 3. Inspect the output
ls generated/double-pendulum
```

You will find `simulation.py`, `plot.py`, `scene.py`, `report.tex`,
`README.md`, `spec.json`, and (after `--plot`) `figure_*.png`.

### One command (Windows)

To run the entire pipeline at once, use the root launcher. It activates the
`simu` virtual environment, installs `simgen` if needed, then generates, runs
and plots the project — and renders the Manim animation / compiles the LaTeX
report when FFmpeg and LaTeX are available (otherwise those steps are skipped):

```bat
run_all.bat                            REM default offline demo (lorenz-system)
run_all.bat -l double-pendulum         REM any built-in library phenomenon
run_all.bat "A driven, damped pendulum showing period doubling"   REM prompt (quote it)
```

## Generating from a prompt (LLM)

With an `ANTHROPIC_API_KEY` configured (see the installation guide):

```bash
simgen generate "A driven, damped pendulum that shows period doubling" --run --plot
```

`simgen` asks Claude for a structured specification, validates it, and renders
the project. If the API is unavailable, it falls back to the closest offline
reference phenomenon (use `--no-fallback` to make it fail instead).

## Running Simulations

Three equivalent ways:

```bash
# (a) As part of generation
simgen generate --from-library lorenz-system --run

# (b) From the generated standalone script
python generated/lorenz-system/simulation.py     # writes data.npz

# (c) From the Python API
python -c "from simgen import run_spec; from simgen.simulations.library import get_spec; \
print(run_spec(get_spec('lorenz-system')).states.shape)"
```

## Rendering Videos

```bash
manim -pql generated/lorenz-system/scene.py LorenzSystemScene   # preview
manim -qh  generated/lorenz-system/scene.py LorenzSystemScene   # high quality
```

The scene class name is `<PascalCaseSlug>Scene`. See
[RENDERING_GUIDE.md](RENDERING_GUIDE.md) for quality presets and batch rendering.

## Generating Reports

```bash
cd generated/lorenz-system
python plot.py            # produce figures referenced by the report
pdflatex report.tex
pdflatex report.tex       # second pass resolves \IfFileExists / references
```

## Understanding Outputs

| Output | What it is |
|--------|------------|
| `data.npz` | NumPy archive with `t`, `states` (shape `n × dim`), and `symbols`. |
| `figure_*.png` | Plots described by the spec (time series / phase portraits). |
| `*.mp4` (under `media/`) | Manim render output. |
| `report.pdf` | Compiled LaTeX report. |
| `spec.json` | The exact specification used to generate everything. |

### Loading results in Python

```python
import numpy as np
data = np.load("generated/lorenz-system/data.npz", allow_pickle=True)
t, states, symbols = data["t"], data["states"], data["symbols"]
print(symbols, states.shape)
```

### Analysing results

```python
from simgen import run_spec
from simgen.simulations.library import get_spec
from simgen.mathematics import dominant_frequency, largest_lyapunov_exponent

result = run_spec(get_spec("simple-harmonic-oscillator"))
print("dominant frequency:", dominant_frequency(result.column("x"), result.dt))
```

## Listing available phenomena

```bash
simgen list
```

## Configuration

`simgen info` prints the resolved settings (model, token budget, output
directory, whether an API key is present). Settings come from environment
variables or a `.env` file; see [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md).
