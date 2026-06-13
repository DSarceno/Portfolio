# Guatemala City Mobility Digital Twin

A **Digital Twin** of urban mobility for Guatemala City: a state-space modeling and
simulation system that estimates *hidden* transportation states from noisy observations,
forecasts how those states evolve, and evaluates hypothetical interventions.

This is **not** a dashboard, a BI platform, or a generic traffic-prediction notebook.
It is a scientific machine-learning and simulation engine built around one architecture:

```
Observation → State Estimation → Dynamics → Simulation
```

## What it is meant to answer

- What will traffic conditions look like in 30, 60, and 120 minutes?
- What happens if a major avenue is partially closed?
- What is the expected impact of a road accident?
- Which zones are approaching congestion saturation?
- How does rainfall affect network dynamics?
- Which mobility interventions produce the largest improvements?

## Documentation

Read these first — they define the scientific and architectural foundation:

| Document | Purpose |
|----------|---------|
| [PROJECT_VISION.md](PROJECT_VISION.md) | Why this exists, what it is and is not, success criteria |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Layer separation and data flow (non-negotiable) |
| [MODELING.md](MODELING.md) | Hidden states, observations, estimators, dynamics, validation |
| [DATASETS.md](DATASETS.md) | Data sources, observation schema, synthetic-first strategy |

## Status

**Phases 1, 3, and 4 complete.** The scientific core works end-to-end:

- **Phase 1** — a road network with a canonical edge index, a synthetic observation
  generator, a Kalman Filter estimating hidden congestion from noisy/partial observations,
  graph-diffusion dynamics with 30/60/120-min forecasting, a simulation loop, and scenario
  interventions (closure, accident, rainfall).
- **Phase 3** — a hybrid physics-informed **Neural ODE** dynamics model (torchdiffeq)
  behind the same interface; the Kalman Filter becomes an **Extended Kalman Filter**
  (autograd Jacobian) for nonlinear dynamics, and the Neural ODE beats the linear baseline
  on forecast RMSE for nonlinear traffic.
- **Phase 4** — a nonlinear simulation **world model** with congestion saturation and
  *directional* upstream **bottleneck back-pressure**, plus **zone-level stress**
  aggregation — kept separate from the estimator so state-estimation interfaces are intact.

Validated by **41 tests**, `ruff` and `mypy` clean. API and visualization are later phases.
See [PROJECT_VISION.md](PROJECT_VISION.md#phases) for the roadmap.

### Run it

```bash
python -m venv .venv && .venv/Scripts/python -m pip install -e ".[dev]"  # Windows
# optional Phase-3 ML stack (Neural ODE); tests skip cleanly without it:
.venv/Scripts/python -m pip install -e ".[ml]"
pytest          # 41 tests
ruff check .
mypy src
```

## Layout

```
src/city_twin/
  ingestion/     road network + raw data acquisition (OSMnx)
  observation/   observation schema + noise model (synthetic now, real adapters later)
  estimation/    Kalman / Extended Kalman state estimators
  dynamics/      state-transition / forecasting models
  simulation/    simulation loop
  scenarios/     intervention configurations (closure, accident, rainfall)
  api/           FastAPI service (later phase)
  viz/           Streamlit prototype (later phase)
tests/           deterministic validation tests
```

## Stack

Python 3.12 · NumPy/SciPy · PyTorch (+ Lightning) · torchdiffeq/TorchDyn ·
OSMnx/NetworkX/GeoPandas · Pandas/Polars · FastAPI · PostgreSQL/TimescaleDB ·
Prefect · Streamlit · Docker. Most of these arrive in later phases; Phase 1 needs only
the scientific-computing and geospatial core.
