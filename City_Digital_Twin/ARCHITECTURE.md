# Architecture

The architecture is **non-negotiable**. Every feature must fit this separation of layers.

## Layered pipeline

```
Data Ingestion
  → Observation Layer
    → State Estimation Layer
      → Dynamics / Forecasting Layer
        → Simulation Layer
          → Scenario Evaluation Layer
            → Visualization / API Layer
```

Data flows downward. Each layer depends only on the contracts of the layer(s) above it,
never on their internals.

## Layers and responsibilities

| Layer | Package | Responsibility | Must NOT |
|-------|---------|----------------|----------|
| Data Ingestion | `ingestion` | Acquire the road network (OSMnx) and raw data; produce the graph and raw records | Define mathematical/modeling assumptions |
| Observation | `observation` | Define the observation schema; model measurement noise and partial/missing observations; emit `y_t` | Contain estimation or dynamics logic |
| State Estimation | `estimation` | Estimate hidden state `x_t` from observations `y_t` (Kalman / Extended Kalman) | Contain simulation or visualization logic |
| Dynamics / Forecasting | `dynamics` | Define the state-transition model `x_{t+Δ} = f(x_t, …)`; roll state forward for forecasts | Read raw data directly or render output |
| Simulation | `simulation` | Run the closed estimation→dynamics loop forward over time | Embed scenario/intervention policy or API code |
| Scenario Evaluation | `scenarios` | Define interventions (closure, accident, rainfall) as modifications to dynamics/network; compare against baseline | Implement the simulation loop itself |
| Visualization / API | `viz`, `api` | Expose and visualize results | Contain modeling, estimation, or simulation logic |

## Hard separation rules

- **Do not** mix simulation logic into API routes.
- **Do not** mix visualization logic into modeling code.
- **Do not** let data ingestion define mathematical assumptions.
- **Do not** casually alter the Digital Twin philosophy or core architecture.

## Sensitive areas (ask before changing)

The simulation engine, the state-estimation layer, model definitions, core mathematical
assumptions, dataset interpretation, and forecasting assumptions are scientifically
load-bearing. Changes affecting their foundation require explicit discussion first.

## Interfaces between layers (Phase 1 contracts)

These are the seams that keep layers swappable. Concrete types are defined in
[MODELING.md](MODELING.md); here we fix *what crosses each boundary*.

- **Ingestion → Observation:** a directed road-network graph (NetworkX) with a stable,
  ordered edge index. The edge ordering defines the index of every state/observation vector.
- **Observation → Estimation:** at each time step, an observation vector `y_t` plus a mask
  of which edges were observed (partial observability is first-class, not an error case).
- **Estimation → Dynamics:** the current state estimate `x_t` and its covariance `P_t`.
- **Dynamics → Simulation:** a transition operator that advances `(x_t, P_t)` by one step.
- **Scenarios → Dynamics/Ingestion:** an intervention object that perturbs the network or
  the transition operator (e.g. remove an edge, drop a segment's capacity, apply a global
  rainfall factor). Scenarios never mutate estimator internals.
- **Simulation/Scenarios → API/Viz:** read-only result objects (state trajectories,
  forecasts, diagnostics). The presentation layers never call back into modeling.

## Why this separation

Hidden-state estimation is only meaningful if the observation model, the estimator, and
the dynamics are independently specified and independently testable. Collapsing any two of
them hides the assumptions we most need to validate. The layer boundaries are what let us
swap a synthetic observation source for a real feed, or a graph-diffusion transition for a
Neural ODE, without rewriting the rest of the system.
