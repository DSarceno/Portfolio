# Video Production Guide

This guide describes how to turn generated scenes into finished educational
videos: the recommended scene sequence, the rendering workflow, export settings,
and a simple editing workflow.

## Recommended scene sequence

For an educational video about a phenomenon, present material in this order
(mirroring the structure of the generated report):

1. **Intuition** — a title card and a plain-language description (the spec's
   `summary`). Show the phenomenon in motion (the generated animation).
2. **Mathematics** — display the governing equations (`equations_latex`).
3. **Simulation** — show the phase portrait / time series figures from
   `plot.py`.
4. **Advanced concepts** — for chaotic systems, discuss sensitivity to initial
   conditions and the Lyapunov exponent; for limit cycles, the attracting set.
5. **Applications** — relate the model to real systems (convection, mechanical
   oscillators, electronics).

The generated `scene.py` covers step 1. Steps 2–3 use the static figures; steps
4–5 are narration over those visuals.

## Production workflow

1. **Generate** the project:
   ```bash
   simgen generate --from-library lorenz-system --run --plot
   ```
2. **Render figures** (already done by `--plot`) and the **animation**:
   ```bash
   manim -qh --media_dir rendered_videos generated/lorenz-system/scene.py LorenzSystemScene
   ```
3. **Compile the report** for on-screen equations/screenshots:
   ```bash
   cd generated/lorenz-system && pdflatex report.tex && pdflatex report.tex
   ```
4. **Assemble** the clips, figures and narration in a video editor.

## Export settings

| Target | Resolution | FPS | Codec | Notes |
|--------|-----------|-----|-------|-------|
| YouTube 1080p | 1920×1080 | 60 | H.264 | `manim -qh` |
| YouTube 4K | 3840×2160 | 60 | H.264 | `manim -qk` |
| Lecture slides | 1280×720 | 30 | H.264 | `manim -qm` |
| GIF preview | 854×480 | 15 | — | `manim -ql --format gif` |

Render audio separately and mix it in your editor; keep narration scripts in
`assets/audio/` if you wish to version them.

## Editing workflow

1. Import the Manim MP4(s) from `rendered_videos/`.
2. Add title cards and lower-thirds (the phenomenon name and key parameters from
   `spec.json`).
3. Cut between the animation and the static phase-portrait/time-series figures.
4. Overlay equations (screenshot from `report.pdf`, or re-create with your
   editor's LaTeX support).
5. Add narration and background music; duck the music under narration.
6. Export with the settings above.

## Tips

- Keep each concept on screen long enough to read (≥ 3 s for an equation).
- Use the same colour for a variable across the animation, plots and slides.
- For chaotic systems, show two nearby initial conditions diverging to make
  sensitive dependence vivid (generate two specs with slightly different
  `initial_conditions`).
