# CLAUDE.md — Guatemala City Mobility Digital Twin

Read [PROJECT_VISION.md](PROJECT_VISION.md), [ARCHITECTURE.md](ARCHITECTURE.md),
[MODELING.md](MODELING.md), and [DATASETS.md](DATASETS.md) before modeling work.

## Non-negotiable rules

- **Preserve the architecture:** `Observation → State Estimation → Dynamics → Simulation`.
  Keep layer separation (see ARCHITECTURE.md). Never mix simulation into API routes,
  visualization into modeling, or let ingestion define mathematical assumptions.
- **This is a Digital Twin / state-space system**, not a dashboard, BI platform, or generic
  forecasting notebook. Reject features that don't serve state estimation, dynamics,
  simulation, or scientific validation.
- **Validate every modeling change scientifically** (RMSE vs. synthetic ground truth at
  30/60/120 min; simulation sanity for closure/accident/rainfall). No validation, no tests,
  or architecture drift = not done.
- **Keep work incremental and minimal.** Smallest scientifically valid solution first; no
  premature infrastructure or dashboards.
- **Ask before changing sensitive areas:** simulation engine, state-estimation layer, model
  definitions, core mathematical assumptions, dataset interpretation, forecasting
  assumptions.

## Current phase

**Phases 1, 3, and 4 complete.** Phase 1: network, observation schema + synthetic
generator, Kalman estimator, graph-diffusion dynamics, simulation loop, scenario
interventions. Phase 3: hybrid physics-informed Neural ODE dynamics (torchdiffeq) behind
the same `DynamicsModel` interface; Kalman Filter acts as an EKF (autograd Jacobian) for
nonlinear dynamics and beats the linear baseline on forecast RMSE. Phase 4: nonlinear
`WorldModel` (saturation + directional upstream bottleneck back-pressure) and zone-level
stress aggregation, kept separate from the estimator. All validated (41 tests, ruff + mypy
clean). ML deps are optional (`pip install -e ".[ml]"`); tests skip cleanly without torch.
Next candidate: Phase 2 (storage) or Phase 5 (API/viz) — see PROJECT_VISION.md → Phases.

## Validation commands

`pytest` · `ruff check .` · `mypy .` — do not claim validation passed unless actually run.
