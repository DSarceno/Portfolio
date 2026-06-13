# Datasets

This document records what data the Digital Twin consumes, how it is interpreted, and the
assumptions attached to each source. Data ingestion **must not** define mathematical
modeling assumptions — those live in [MODELING.md](MODELING.md). This file describes
*sources and schema*, not estimator behavior.

## Strategy: synthetic-first

Phase 1 runs on **synthetic observations generated over the real road network**. We do this
deliberately:

- The real OSM network gives realistic topology, connectivity, and segment geometry.
- A synthetic observation generator gives us **ground-truth hidden state**, so we can
  *measure* estimation and forecast error (RMSE at 30/60/120 min). No real feed provides
  ground truth for a *latent* congestion state.
- Real feeds attach later through a stable adapter interface (see *Real data adapters*),
  leaving the estimation and dynamics layers untouched.

## Road network (real, used from Phase 1)

- **Source:** OpenStreetMap via **OSMnx**.
- **Region:** Guatemala City metropolitan area (system scope).
- **Phase 1 filter — strategic subgraph:** keep `motorway`, `trunk`, `primary`, and
  `secondary` road classes (and their `_link` ramps). Residential/service streets are
  excluded initially to keep the state vector tractable; later phases densify.
- **Representation:** a **directed** graph (NetworkX). Each directed edge is a road segment
  and carries: length, free-flow speed (from OSM `maxspeed`/class defaults), number of
  lanes if available, and a stable integer index. **The ordered edge index defines the
  index of every state and observation vector in the system.**

## Observation schema (Phase 1)

An observation at time `t` describes measured traffic on a *subset* of edges:

| Field | Type | Meaning |
|-------|------|---------|
| `timestamp` | datetime (UTC) | observation time |
| `edge_id` | int | index into the canonical edge ordering |
| `speed` | float (km/h) | measured mean speed on the segment |
| `travel_time` | float (s) | measured traversal time (derivable from speed + length) |
| `source` | enum | `synthetic` now; `google_routes`, `waze`, etc. later |

Observations are **noisy** and **partial**: only some edges are reported at each step, and
reported values include measurement noise. Missing edges are represented by an observation
mask, never by imputed zeros. The estimator is responsible for inferring unobserved edges.

## Synthetic observation generator (Phase 1)

Generates a known ground-truth congestion trajectory on the strategic subgraph, then emits
noisy, partially-sampled observations from it. It must support, for validation:

- baseline (diurnal) congestion patterns,
- adjustable observation noise level,
- adjustable observation sparsity (fraction of edges reported),
- injectable events (a localized congestion spike) for scenario sanity checks.

The exact noise and dynamics models are specified in [MODELING.md](MODELING.md).

## Real data adapters (later phases)

Each real source implements the same **Observation Layer** contract (emit `(timestamp,
edge_id, speed, travel_time, source)` records keyed to the canonical edge index). Likely
order of integration and their caveats:

| Source | Signal | Caveats |
|--------|--------|---------|
| **Google Routes/Maps API** | travel time / typical-traffic speed per segment | paid, rate-limited, indirect observation of latent state |
| **Waze / TomTom / HERE** | speeds, jams, incidents | require commercial/partnership agreements |
| **Municipal / open data** (Transmetro, Transurbano GTFS, EMETRA) | transit schedules, limited real-time | mostly transit, sparse real-time coverage |

## Exogenous data (Phase 2+)

- **Weather / rainfall** — candidate global or zonal modifier of dynamics.
- **Incidents / closures** — candidate localized modifiers; also drive scenario definitions.

These are *modifiers* of dynamics, introduced only when both the data and a validated
modeling use for them exist.

## Interpretation assumptions

- Observed metrics (speed/travel time) are **noisy, indirect** measurements of a hidden
  congestion state — they do not fully describe the system.
- Roads are **not** independent; congestion propagates through the network.
- Historical patterns inform priors but are **not** assumed to always repeat.
