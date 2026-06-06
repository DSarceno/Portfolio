# Installation Guide

This guide covers installing `simgen` and its optional toolchains (LaTeX,
FFmpeg, Manim, the Anthropic client) on Windows, macOS and Linux.

## 1. Python setup

`simgen` requires **Python 3.10 or newer**.

```bash
python --version          # confirm >= 3.10
```

Create and activate a virtual environment:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# Windows cmd
.venv\Scripts\activate.bat
# macOS / Linux
source .venv/bin/activate
```

Install the package:

```bash
# Core (numpy, scipy, matplotlib, jinja2) — enough for simulations,
# plots, the offline library and the test suite.
pip install -e .

# With development tools (pytest, coverage)
pip install -e ".[dev]"

# With the LLM client (anthropic)
pip install -e ".[llm]"

# With Manim animation support
pip install -e ".[animation]"

# Everything
pip install -e ".[all,dev]"
```

Alternatively, with conda:

```bash
conda env create -f environment.yml
conda activate simgen
pip install -e .
```

## 2. Configuration (LLM)

LLM-backed generation needs an Anthropic API key. Copy the template and edit it:

```bash
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY=sk-ant-...
```

Or export the variable in your shell:

```bash
# macOS / Linux
export ANTHROPIC_API_KEY=sk-ant-...
# Windows PowerShell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
```

Verify with `simgen info` (the `api_key_present` line should read `True`).
Without a key, `simgen` still works using the offline reference library.

## 3. FFmpeg setup (for Manim)

Manim encodes frames into video with FFmpeg.

- **Windows:** `winget install Gyan.FFmpeg` or `choco install ffmpeg`, or use the
  conda package (`environment.yml` already includes `ffmpeg`).
- **macOS:** `brew install ffmpeg`.
- **Linux (Debian/Ubuntu):** `sudo apt install ffmpeg`.

Confirm with `ffmpeg -version`.

## 4. LaTeX setup

LaTeX is needed to (a) compile reports and (b) render mathematical text inside
Manim.

- **Windows:** [MiKTeX](https://miktex.org/) (install missing packages on the
  fly) or TeX Live.
- **macOS:** [MacTeX](https://tug.org/mactex/) (or `brew install --cask mactex`).
- **Linux:** `sudo apt install texlive-full` (or a smaller subset:
  `texlive-latex-extra texlive-fonts-recommended texlive-science`).

Confirm with `pdflatex --version` and `bibtex --version`.

## 5. Manim setup

After FFmpeg and LaTeX are available:

```bash
pip install -e ".[animation]"
manim --version
```

See the [Manim installation docs](https://docs.manim.community/en/stable/installation.html)
for platform-specific notes (e.g. Cairo/Pango build dependencies on Linux).

## 6. Verify the installation

```bash
# Run the test suite (offline, no API key or Manim required)
pytest -q

# Generate a project from the offline library and run it
simgen generate --from-library lorenz-system --run --plot

# Run the example walkthrough
python examples/example_usage.py
```

If all of the above succeed, your installation is complete. For problems, see
[TROUBLESHOOTING.md](TROUBLESHOOTING.md).
