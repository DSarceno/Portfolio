Guatemala City Mobility Digital Twin — Claude Code Workflow Prompt
Core Vision

Build a Mobility Digital Twin for Guatemala City.

This project models urban transportation as a dynamic system with partially observable hidden states.

Preserve this architecture at all times:

Observation → State Estimation → Dynamics → Simulation

This is NOT:

A dashboard
A BI platform
A simple traffic prediction app
A generic forecasting notebook
A geospatial visualization project

This IS:

A Digital Twin
A state-space modeling system
A scientific machine learning platform
A mobility simulation engine

Say no to good ideas that do not serve state estimation, dynamics modeling, simulation, or scientific validation.

Project Goal

Help build a production-quality research and engineering project capable of:

Ingesting urban mobility data
Estimating hidden transportation states
Simulating city-wide traffic dynamics
Forecasting future mobility conditions
Evaluating hypothetical interventions
Serving as a foundation for transportation research

The system should eventually answer questions like:

What will traffic conditions look like in 30, 60, and 120 minutes?
What happens if a major avenue is partially closed?
What is the expected impact of a road accident?
Which zones are approaching congestion saturation?
How does rainfall affect network dynamics?
What mobility interventions produce the largest improvements?
Target Stack

Start from a new codebase.

Use this stack unless there is a strong reason not to:

Backend
Python 3.12
FastAPI
Data Engineering
Pandas
Polars
SQLAlchemy
Storage
PostgreSQL
TimescaleDB
Machine Learning
PyTorch
PyTorch Lightning
Scikit-Learn
Scientific Computing
NumPy
SciPy
Dynamic Systems
torchdiffeq
TorchDyn
Geospatial
GeoPandas
OSMnx
NetworkX
Orchestration
Prefect
Visualization
Streamlit for Phase 1 only
Next.js only as a possible future phase
Infrastructure
Docker
Docker Compose
Required First Step

Before implementation, inspect or create these documents:

README.md
PROJECT_VISION.md
ARCHITECTURE.md
DATASETS.md
MODELING.md

If these files do not exist, create minimal but clear first versions before writing application code.

Do not start with infrastructure, dashboards, or API endpoints before the project vision and modeling architecture are clear.

Operating Principles

Work iteratively in small validated steps.

Priority order:

Correctness
Architecture
Maintainability
Scientific validity
Performance
UX

Prefer the smallest scientifically valid solution first.

Do not over-engineer.

Avoid complexity that does not improve the scientific or simulation quality of the system.

Avoid:

Microservices
Distributed systems
Complex event-driven architecture
Premature production infrastructure
Fancy dashboards
AI assistants
Reporting features disconnected from the Digital Twin
Generic ML pipelines without state estimation
Kaggle-style notebooks as the core system
Scope
In Scope
State-space models
Kalman Filters
Extended Kalman Filters
Hidden state estimation
Traffic forecasting
Mobility simulation
Neural ODEs
Transportation network modeling
Scenario simulation
Geospatial analysis
Noisy observation modeling
Forecast validation
Intervention testing
Out of Scope
LLM chatbots
Generative AI features
Blockchain
Cryptocurrency
Social features
Mobile applications
Dashboard-first development
BI reporting platforms
Generic descriptive analytics
Non-Negotiable Architecture

The architecture must preserve this separation:

Data Ingestion
→ Observation Layer
→ State Estimation Layer
→ Dynamics / Forecasting Layer
→ Simulation Layer
→ Scenario Evaluation Layer
→ Visualization / API Layer

Keep strict separation between:

Data ingestion
State estimation
Modeling
Simulation
Visualization

Do not mix simulation logic into API routes.

Do not mix visualization logic into modeling code.

Do not let data ingestion define mathematical assumptions.

Do not casually alter the Digital Twin philosophy or the core architecture.

Sensitive Areas

Be especially careful with:

Simulation engine
State-estimation layer
Model definitions
Core mathematical assumptions
Dataset interpretation
Forecasting assumptions

Ask before changing these if the change affects the scientific foundation of the project.

Mathematical Assumptions

Safe assumptions:

Urban observations contain noise
Important transportation states are hidden
Traffic dynamics evolve continuously over time
Congestion can propagate through the network
Road network structure matters
Weather, incidents, and closures may affect dynamics

Do NOT assume:

Observed traffic metrics fully describe the system
All roads behave independently
Historical patterns always repeat
Dashboard metrics are enough to represent mobility dynamics
A single model is sufficient for all city conditions
Workflow for Every Task

For every task, follow this process:

Understand the goal.
Inspect relevant docs and code first.
Summarize what you found.
Identify the next minimal useful task.
Propose up to 3 options.
Recommend one option.
Wait for approval before implementation when architectural or mathematical assumptions are affected.
Implement incrementally.
Validate scientifically.
Run tests and checks.
Document changes.
Provide a copyable Follow-Up Prompt for the next step.

Ask questions only when assumptions affect:

Mathematical models
System architecture
Dataset interpretation
Scientific validity

For routine implementation details, make reasonable assumptions and proceed.

Phase-Based Development

Use staged development.

Phase 0 — Project Foundation

Goal:

Establish vision, architecture, dataset assumptions, and modeling philosophy.

Deliverables:

README.md
PROJECT_VISION.md
ARCHITECTURE.md
DATASETS.md
MODELING.md
Initial project structure

Do not build complex infrastructure in this phase.

Phase 1 — Minimal Scientific Prototype

Goal:

Build the smallest scientifically valid Digital Twin prototype.

Focus:

Road network representation
Observation schema
Hidden state definition
Basic state estimation
Simple dynamics model
Basic simulation loop

Possible implementation:

OSMnx road network ingestion
Synthetic or small sample observations
Kalman Filter or Extended Kalman Filter baseline
Simple congestion state variable
Basic forecast horizon: 30, 60, 120 minutes

Success:

The system can estimate hidden traffic state from noisy observations.
The system can forecast state evolution.
The system can simulate a simple intervention.
Phase 2 — Data Engineering and Storage

Goal:

Create reliable ingestion and storage for mobility observations.

Focus:

PostgreSQL / TimescaleDB schema
Data validation
Historical observation storage
Weather / rainfall integration
Incidents or closure data if available
Prefect pipeline only if useful

Success:

Data can be ingested, validated, stored, and retrieved for modeling.
Phase 3 — Dynamics and Scientific ML

Goal:

Improve state transition modeling.

Focus:

Neural ODEs
TorchDyn / torchdiffeq
Hybrid physics-informed models
Network-aware dynamics
Forecast validation

Success:

Dynamics model improves forecast quality while preserving interpretability and validation.
Phase 4 — Simulation Engine

Goal:

Build scenario simulation for interventions.

Focus:

Road closures
Accidents
Rainfall effects
Bottleneck propagation
Congestion saturation
Zone-level stress

Success:

The system can evaluate hypothetical mobility interventions.
Phase 5 — API and Visualization

Goal:

Expose useful outputs without turning the project into a dashboard-first product.

Focus:

FastAPI endpoints
Streamlit prototype
Scenario comparison
Forecast inspection
State estimation diagnostics

Success:

Visualization helps interpret state estimation and simulation results.
Claude Code Features

Use Claude Code’s built-in behavior and tools.

Do not override Claude Code’s identity or core system behavior.

Use these features only when helpful.

Subagents

Use subagents for focused review or implementation support:

code-reviewer for architecture and maintainability checks
debugger for failing tests or runtime issues
test-writer for validation coverage
docs-writer for technical documentation
MCP

Potential MCP integrations:

GitHub
PostgreSQL
Supabase
Notion

Only use MCP if it directly helps the current task.

Do not add MCP complexity prematurely.

Slash Commands

Use slash commands when helpful:

/init
/memory
/agents
/compact
/review
Memory

If recurring project rules emerge, propose updates to CLAUDE.md.

Examples of good CLAUDE.md memory:

Preserve Observation → State Estimation → Dynamics → Simulation
Do not prioritize dashboards before modeling validity
Validate every modeling change scientifically
Keep implementation incremental and minimal
Hooks

If deterministic checks become useful, suggest hooks for:

Formatting
Linting
Type checking
Running tests
Blocking accidental edits to sensitive modeling or simulation files

Do not add hooks unless they reduce repeated manual checking.

Validation Commands

When applicable, run:

pytest
ruff check .
mypy .
docker compose up
pre-commit run --all-files

If a command is unavailable, explain why and suggest the next minimal setup step.

Do not claim validation passed unless commands were actually run.

Scientific Validation Requirements

Every modeling or simulation feature must include validation.

Check for:

Noisy observations
Missing observations
Hidden state assumptions
Edge traffic scenarios
Congestion saturation
Bottleneck propagation
Rainfall effects when relevant
Incident effects when relevant
Forecast stability
Simulation sanity
Sensitivity to bad or sparse data

For forecasting, validate:

30-minute horizon
60-minute horizon
120-minute horizon

For simulation, validate:

Baseline scenario
Road closure scenario
Accident scenario
Rainfall scenario when data exists
Testing Requirements

Add meaningful tests for:

Data validation
Observation schema
State estimation
Dynamics model
Simulation loop
Scenario configuration
Forecast output shape
Missing data behavior
Edge traffic conditions

Prefer deterministic fixtures.

Avoid tests that only check that code runs without validating behavior.

Documentation Requirements

Update documentation whenever architectural or modeling decisions change.

Important documents:

README.md
PROJECT_VISION.md
ARCHITECTURE.md
DATASETS.md
MODELING.md

Documentation should explain:

Why the chosen model is appropriate
What hidden states are being estimated
What observations are used
What assumptions are being made
How validation is performed
What limitations remain
Definition of Done

A task is done only when it is:

Implemented
Tested
Documented
Scientifically validated
Consistent with the Digital Twin architecture

Unacceptable results:

No validation
No tests
Architecture drift
Dashboard-first thinking
Features disconnected from state estimation or simulation
Infrastructure complexity without scientific benefit
Generic ML without state-space reasoning
Decision Rules

When choosing between options:

Prefer:

Scientific validity over visual polish
Simple validated models over complex unvalidated models
Clear architecture over clever abstractions
Small incremental progress over large risky rewrites
Explicit assumptions over hidden assumptions
Reproducible experiments over ad hoc results

Reject:

Features that only make the project look impressive
Architecture that increases cognitive load without improving modeling
Dashboards before the model is meaningful
Data pipelines that do not support state estimation
ML models that cannot be validated against the Digital Twin objective
Final Response Format

At the end of each phase or task, respond with:

Summary

Brief summary of what was done.

Files Changed
path/to/file: what changed
Decisions Made
Decision and reason
Scientific Rationale

Explain the modeling or simulation reasoning.

Commands Run

Command:

command

Result:

result summary
Risks / Tradeoffs
Risk or tradeoff
Next Minimal Task

Describe the smallest useful next step.

Follow-Up Prompt

Paste-ready prompt for the next Claude Code step.

First Task to Execute

Start with Phase 0.

Inspect the current workspace.

If the project foundation docs do not exist, create:

README.md
PROJECT_VISION.md
ARCHITECTURE.md
DATASETS.md
MODELING.md

Keep them concise but useful.

Do not implement the full system yet.

Focus only on establishing the scientific and architectural foundation.

After Phase 0, stop and provide:

Summary
Files changed
Decisions made
Risks / tradeoffs
Next minimal task
Follow-Up Prompt