# Rendering Guide

How to render the generated Manim animations at different quality levels.

## Prerequisites

- Manim Community (`pip install -e ".[animation]"`)
- FFmpeg on the PATH
- A LaTeX installation (Manim uses it for math text)

See [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md).

## Scene names

Each generated `scene.py` contains one scene class named
`<PascalCaseSlug>Scene`. Examples:

| Slug | Scene class |
|------|-------------|
| `lorenz-system` | `LorenzSystemScene` |
| `double-pendulum` | `DoublePendulumScene` |
| `simple-harmonic-oscillator` | `SimpleHarmonicOscillatorScene` |
| `duffing-oscillator` | `DuffingOscillatorScene` |
| `van-der-pol-oscillator` | `VanDerPolOscillatorScene` |

## Quality presets

Manim quality flags map to resolution/frame-rate:

| Flag | Quality | Resolution | FPS |
|------|---------|-----------|-----|
| `-ql` | low | 854×480 | 15 |
| `-qm` | medium | 1280×720 | 30 |
| `-qh` | high | 1920×1080 | 60 |
| `-qp` | 2k production | 2560×1440 | 60 |
| `-qk` | 4k | 3840×2160 | 60 |

Add `-p` to preview (auto-play) when done.

### Low-quality render (fast iteration)

```bash
manim -pql generated/lorenz-system/scene.py LorenzSystemScene
```

### High-quality render

```bash
manim -qh generated/lorenz-system/scene.py LorenzSystemScene
```

### Production render (1440p, no preview)

```bash
manim -qp generated/lorenz-system/scene.py LorenzSystemScene
```

### Output location

By default Manim writes to `media/videos/<scene-file>/<resolution>/`. To collect
final cuts in this repo's `rendered_videos/`, pass `--media_dir`:

```bash
manim -qh --media_dir rendered_videos generated/lorenz-system/scene.py LorenzSystemScene
```

## Batch rendering

Render every generated scene (PowerShell):

```powershell
Get-ChildItem generated -Directory | ForEach-Object {
    $slug = $_.Name
    $cls = ($slug -split '-' | ForEach-Object { $_.Substring(0,1).ToUpper() + $_.Substring(1) }) -join ''
    manim -qh --media_dir rendered_videos "generated/$slug/scene.py" "${cls}Scene"
}
```

Bash:

```bash
for dir in generated/*/; do
  slug=$(basename "$dir")
  cls=$(echo "$slug" | awk -F- '{for(i=1;i<=NF;i++){$i=toupper(substr($i,1,1)) substr($i,2)}}1' OFS='')
  manim -qh --media_dir rendered_videos "$dir/scene.py" "${cls}Scene"
done
```

## Common options

- `-s` — render only the final frame (a PNG still).
- `--format gif` — export an animated GIF instead of MP4.
- `-a` — render all scenes in the file (here, just one).
- `--fps 30` — override the frame rate.

## Troubleshooting

If Manim cannot find LaTeX or FFmpeg, or a render fails, see
[TROUBLESHOOTING.md](TROUBLESHOOTING.md).
