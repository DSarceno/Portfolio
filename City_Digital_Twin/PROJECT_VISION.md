# Project Vision

## Core vision

Build a **Mobility Digital Twin** for Guatemala City that models urban transportation as a
dynamic system with **partially observable hidden states**.

The entire system preserves one architecture, at all times:

```
Observation → State Estimation → Dynamics → Simulation
```

## What this IS

- A Digital Twin of city mobility
- A state-space modeling system
- A scientific machine-learning platform
- A mobility simulation engine

## What this is NOT

- A dashboard or BI platform
- A simple traffic-prediction app
- A generic forecasting or geospatial-visualization notebook

We say **no** to good ideas that do not serve state estimation, dynamics modeling,
simulation, or scientific validation.

## Goal

A production-quality research and engineering project capable of:

1. Ingesting urban mobility data
2. Estimating hidden transportation states
3. Simulating city-wide traffic dynamics
4. Forecasting future mobility conditions
5. Evaluating hypothetical interventions
6. Serving as a foundation for transportation research

## Scope

**In scope:** state-space models, Kalman / Extended Kalman filters, hidden-state
estimation, traffic forecasting, mobility simulation, Neural ODEs, transportation-network
modeling, scenario simulation, geospatial analysis, noisy-observation modeling, forecast
validation, intervention testing.

**Out of scope:** LLM chatbots, generative-AI features, blockchain/crypto, social features,
mobile apps, dashboard-first development, BI reporting, generic descriptive analytics.

## Operating principles

Priority order: **Correctness → Architecture → Maintainability → Scientific validity →
Performance → UX.**

- Work iteratively in small, validated steps.
- Prefer the smallest scientifically valid solution first. Do not over-engineer.
- Avoid microservices, distributed systems, premature production infrastructure, and
  fancy dashboards.
- Explicit assumptions over hidden ones; reproducible experiments over ad-hoc results.

## Project-specific decisions

- **Geographic scope:** the full Guatemala City metropolitan area is the *system scope*.
  Phase 1 operates on a **strategic subgraph** (major roads only) to keep the state vector
  tractable, then densifies in later phases.
- **Data strategy:** **synthetic-first.** Phase 1 generates noisy, partial observations on
  the *real* OSM road network. This provides ground truth to measure estimation and
  forecast error — something real feeds cannot give on day one. Real feeds (Google Routes
  API the likely first integration) attach later through a stable adapter interface without
  changing the estimation or dynamics layers. See [DATASETS.md](DATASETS.md).

## Definition of Done

A task is done only when it is **implemented, tested, documented, scientifically
validated, and consistent with the Digital Twin architecture.** No validation, no tests,
architecture drift, or dashboard-first thinking are unacceptable outcomes.

## Phases

| Phase | Goal | Success |
|-------|------|---------|
| **0 — Foundation** *(done)* | Vision, architecture, datasets, modeling philosophy, skeleton | Foundation docs and package structure exist; no system logic yet |
| **1 — Minimal Scientific Prototype** *(done)* | Network, observation schema, hidden state, basic estimation, simple dynamics, simulation loop | System estimates hidden state from noisy observations, forecasts its evolution, and simulates a simple intervention |
| **2 — Data Engineering & Storage** | Reliable ingestion + storage (PostgreSQL/TimescaleDB), validation, weather/incidents | Data can be ingested, validated, stored, and retrieved for modeling |
| **3 — Dynamics & Scientific ML** *(done)* | Neural ODEs, hybrid physics-informed, network-aware dynamics | Dynamics model improves forecasts while preserving interpretability and validation |
| **4 — Simulation Engine** *(done)* | Closures, accidents, rainfall, bottleneck propagation, saturation | System can evaluate hypothetical interventions |
| **5 — API & Visualization** | FastAPI endpoints, Streamlit prototype, scenario comparison | Visualization helps interpret estimation and simulation results — without becoming dashboard-first |
