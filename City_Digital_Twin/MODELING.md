# Modeling

This document defines the scientific core: the hidden state, the observation model, the
state estimator, the dynamics, forecasting, simulation, and how every modeling change is
validated. It is the most sensitive document in the project — changes here affect the
scientific foundation and must be discussed before implementation.

## State-space formulation

The Digital Twin is a discrete-time state-space model:

```
x_{t+Δ} = f(x_t, u_t) + w_t      (dynamics / process model,    process noise w_t)
y_t     = h(x_t)      + v_t      (observation model,           measurement noise v_t)
```

- `x_t` — the **hidden state** we estimate (latent congestion across the network).
- `y_t` — **noisy, partial observations** (measured speeds/travel times on some edges).
- `u_t` — exogenous inputs / interventions (closures, accidents, rainfall).
- `w_t`, `v_t` — process and measurement noise.

## Hidden state (Phase 1)

For a strategic subgraph with `N` directed edges:

- `x_t ∈ ℝ^N` is the per-edge **congestion level**, defined as a normalized **speed
  deficit**:

  ```
  congestion = 1 − (current_speed / free_flow_speed),   clipped to [0, 1]
  ```

  `0` = free flow, `1` = fully saturated (standstill).

- This is *hidden*: we never observe it directly. We observe noisy speeds on a subset of
  edges and must infer congestion everywhere, including unobserved edges.
- Optional augmentation (deferred): a rate-of-change term per edge for smoother dynamics.

**Why per-edge congestion as normalized speed deficit:** it is bounded, interpretable,
comparable across roads of different free-flow speeds, and directly relatable to the
observed quantity (speed) through a simple, near-linear observation map — which keeps the
Phase 1 estimator tractable.

## Observation model (Phase 1)

`h` maps hidden congestion to expected measured speed on **observed** edges only:

```
expected_speed_e = free_flow_speed_e · (1 − x_e)
```

- Partial observability is modeled with an **observation matrix / mask** `H_t` selecting the
  edges reported at step `t`. Unobserved edges contribute nothing at that step.
- Measurement noise `v_t` is additive Gaussian with covariance `R` (per-edge variance).
- This map is linear in `x_t` for fixed free-flow speeds → a linear Kalman update is valid
  in Phase 1. If we later observe travel time instead of speed, `h` becomes nonlinear and
  we move to the Extended Kalman Filter.

## State estimator

- **Phase 1 baseline: Kalman Filter (KF).** Linear dynamics + linear observation map make
  the KF the correct, minimal, *optimal* estimator. It naturally handles noise, partial
  observations (time-varying `H_t`), and produces a covariance `P_t` quantifying
  uncertainty on unobserved edges.
- **Extended Kalman Filter (EKF):** adopted when dynamics or the observation map become
  nonlinear (e.g. travel-time observations, capacity-dependent propagation).
- The estimator consumes `(y_t, mask)` and produces `(x_t, P_t)`. It contains no dynamics
  definition of its own — it *uses* the dynamics operator from the `dynamics` layer.

## Dynamics / forecasting (Phase 1)

**Phase 1 transition: graph diffusion + decay.** Congestion spreads to topological
neighbors and relaxes toward free flow:

```
x_{t+Δ} = x_t + Δ · ( −α · x_t  −  β · L · x_t ) + w_t
```

- `−α · x_t` — **decay** toward free flow (congestion dissipates over time).
- `−β · L · x_t` — **diffusion** via the (directed) graph Laplacian `L`: congestion
  propagates between connected segments.
- `α, β ≥ 0` are interpretable parameters; `w_t` is Gaussian process noise `Q`.

This is the smallest dynamics that captures the two non-negotiable behaviors — **dissipation**
and **network propagation** — while remaining linear (KF-compatible) and validatable.

- **Phase 3 (implemented):** `f` can be a **hybrid physics-informed Neural ODE**
  (`torchdiffeq`) — explicit decay + graph-diffusion physics plus a learned, network-aware
  residual MLP. It conforms to the same `DynamicsModel` interface
  (`city_twin.dynamics.neural_ode`), so the estimator and simulator are unchanged. Because
  the transition is now nonlinear, covariance is propagated by **linearizing the one-step
  map with autograd** — i.e. the Kalman Filter operates as an **Extended Kalman Filter**.
  The Neural ODE is validated to beat the linear baseline on forecast RMSE for nonlinear
  (logistic-congestion) ground truth. The linear model remains the validated baseline; the
  Neural ODE must match or beat it before adoption.

**Forecasting:** apply `f` repeatedly with no new observations to roll the state forward to
the **30 / 60 / 120-minute** horizons, propagating covariance for uncertainty bands.

## Simulation & scenarios

The simulation loop alternates dynamics prediction and (when available) estimator updates.
Interventions are **inputs `u_t`** that modify the network or the transition operator:

- **Road closure** — remove an edge (or set its capacity/throughput to ~0); changes `L`.
- **Accident** — localized capacity drop on one or more edges for a duration.
- **Rainfall** — global (or zonal) speed-reduction factor scaling free-flow speeds /
  increasing congestion growth.

Scenarios are compared against a **baseline** run with identical noise seeds.

## Mathematical assumptions

**Safe to assume:** observations contain noise; key states are hidden; dynamics evolve
continuously over time; congestion propagates through the network; network structure
matters; weather/incidents/closures may affect dynamics.

**Do NOT assume:** observed metrics fully describe the system; roads behave independently;
historical patterns always repeat; dashboard metrics represent mobility dynamics; a single
model suffices for all city conditions.

## Validation requirements

Every modeling or simulation change ships with validation. At minimum:

- **Estimation:** state RMSE against synthetic ground truth, under varying **noise** and
  **observation sparsity**; correct behavior with **missing** observations; recovery of
  unobserved edges via propagation.
- **Forecasting:** RMSE at **30, 60, 120-minute** horizons; **forecast stability** (no
  blow-up); uncertainty growth is monotonic without observations.
- **Simulation sanity:** closure increases congestion downstream of the closed edge;
  accident raises local congestion then relaxes; rainfall raises network-wide congestion;
  bottleneck propagation and congestion saturation behave physically.
- **Robustness:** sensitivity to sparse/bad data; edge traffic scenarios; congestion
  saturation at the `[0,1]` bounds.

## Testing requirements

Deterministic fixtures (fixed seeds, small hand-built graphs). Meaningful tests for: data
validation, observation schema, state estimation, dynamics, simulation loop, scenario
configuration, forecast output shape, missing-data behavior, and edge traffic conditions.
Avoid tests that only check code runs without validating behavior.

## Known limitations (Phase 1)

- Synthetic observations are a modeling convenience, not Guatemala City reality; absolute
  numbers are not yet meaningful — *relative* behavior and estimator/forecast accuracy are.
- Linear graph-diffusion dynamics cannot capture nonlinear flow phenomena (shockwaves,
  capacity-dependent breakdown); that is the explicit motivation for Phase 3.
- Strategic-subgraph resolution ignores local-street effects until densification.
