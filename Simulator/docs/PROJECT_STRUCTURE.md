# Project Structure

A directory-by-directory explanation of the repository.

```text
Simulator/
├── README.md                       Project overview and quick start.
├── LICENSE                         MIT license.
├── requirements.txt                Core pip dependencies.
├── environment.yml                 Conda environment (incl. ffmpeg, manim).
├── pyproject.toml                  Packaging, entry point, tool config.
├── .env.example                    Configuration template (copy to .env).
├── .gitignore                      Ignored build/output artifacts.
│
├── docs/                           Documentation (this directory).
│   ├── USER_GUIDE.md
│   ├── INSTALLATION_GUIDE.md
│   ├── TECHNICAL_DOCUMENTATION.md
│   ├── PROJECT_STRUCTURE.md
│   ├── SCIENTIFIC_BACKGROUND.md
│   ├── VIDEO_PRODUCTION_GUIDE.md
│   ├── RENDERING_GUIDE.md
│   ├── EXTENSION_GUIDE.md
│   ├── TROUBLESHOOTING.md
│   └── diagrams/
│       ├── architecture.md
│       ├── data_flow.md
│       └── rendering_pipeline.md
│
├── src/simgen/                     The installable package.
│   ├── __init__.py                 Public API surface.
│   ├── cli.py                      `simgen` command-line interface.
│   ├── config/                     Settings resolution.
│   ├── utilities/                  Logging + IO helpers.
│   ├── numerical_methods/          Integrators + linear algebra.
│   ├── physics/                    Constants + units.
│   ├── mathematics/                Time-series / dynamics analysis.
│   ├── simulations/                Spec model, base classes, runner, library.
│   ├── visualization/              Matplotlib plotting.
│   ├── scenes/                     Manim components (optional dependency).
│   ├── reporting/                  LaTeX helpers.
│   ├── llm/                        Claude client, prompts, schema, generator.
│   └── codegen/                    Jinja2 templates + renderer.
│       └── templates/              simulation/plot/scene/report/readme .j2
│
├── assets/                         Static assets used by docs/animations.
│   ├── images/                     Diagrams, logos.
│   ├── audio/                      Optional narration tracks.
│   ├── data/                       Reference datasets.
│   └── fonts/                      Custom fonts for animations.
│
├── reports/                        Top-level framework report.
│   ├── report.tex                  Umbrella LaTeX report.
│   └── bibliography.bib            Books, articles, online references.
│
├── tests/                          pytest suite.
│   ├── test_numerical_methods.py
│   ├── test_math_models.py
│   ├── test_physics_models.py
│   ├── test_simulations.py
│   └── test_scenes.py
│
├── examples/
│   └── example_usage.py            End-to-end walkthrough script.
│
├── rendered_videos/                Target directory for final videos.
│   └── .gitkeep
│
├── generated/                      (git-ignored) generated projects land here.
│
└── REPOSITORY_COMPLETENESS_REPORT.md   Final audit.
```

## `src/` layout rationale

The package uses the **src layout** (`src/simgen`). This prevents accidental
imports of the in-tree package before installation and is the recommended modern
Python packaging convention. Install with `pip install -e .` to develop against
it.

## Generated project layout

Each generated phenomenon lives in `generated/<slug>/`:

```text
generated/<slug>/
├── simulation.py    Standalone equations + integrator (numpy/scipy only).
├── plot.py          Standalone Matplotlib figures.
├── scene.py         Standalone Manim scene.
├── report.tex       Standalone LaTeX report.
├── README.md        How to run this generated project.
└── spec.json        The specification used to generate everything.
```

Generated projects do **not** import `simgen`; they are portable.
