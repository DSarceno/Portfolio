# Rendering Pipeline Diagram

From a `PhenomenonSpec` to the four artifact types and their final outputs.

```mermaid
flowchart LR
    SPEC["PhenomenonSpec"] --> R["codegen.renderer\n(Jinja2 templates)"]

    R --> SIM["simulation.py\n(numpy/scipy)"]
    R --> PL["plot.py\n(matplotlib)"]
    R --> SC["scene.py\n(manim)"]
    R --> RT["report.tex\n(LaTeX)"]

    SIM -->|python simulation.py| NPZ["data.npz\n(t, states, symbols)"]
    PL -->|python plot.py| PNG["figure_*.png"]
    SC -->|manim -qh| MP4["video .mp4\n(rendered_videos/)"]
    RT -->|pdflatex x2| PDF["report.pdf"]

    PNG -.referenced by.-> RT
```

## Artifact dependencies

- `plot.py` and `scene.py` both `import simulation` (same directory) to obtain
  the trajectory via `simulate()`.
- `report.tex` references `figure_*.png` through `\IfFileExists`, so it compiles
  whether or not the figures have been generated yet.
- Each artifact is **standalone**: generated projects depend only on
  NumPy/SciPy/Matplotlib (and Manim for the animation), never on `simgen`.

## Tooling per stage

| Artifact | Tool | Command |
|----------|------|---------|
| `simulation.py` | Python + NumPy/SciPy | `python simulation.py` |
| `plot.py` | Matplotlib (Agg) | `python plot.py` |
| `scene.py` | Manim + FFmpeg + LaTeX | `manim -qh scene.py <Class>Scene` |
| `report.tex` | pdfLaTeX | `pdflatex report.tex` (×2) |
