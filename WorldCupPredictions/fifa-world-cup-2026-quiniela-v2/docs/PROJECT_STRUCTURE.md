# Project Structure

```
fifa-world-cup-2026-quiniela-v2/
├── src/                         # Application code
│   ├── data/                    # Ingestion + canonical table (+ squad_value_client)
│   ├── ratings/                 # Elo, PI, form, ensemble, shrinkage (mixture prior)
│   ├── features/                # Feature engineering (+ squad_value_features)
│   ├── models/                  # Outcome and scoreline models
│   ├── ensemble/                # Probability blending + pick optimizer
│   ├── simulation/              # Group stage, bracket, knockout, simulator
│   ├── prediction/              # Match predictor, scoreline, quiniela, pairwise builder
│   ├── training/                # Trainer, evaluator, cross-tournament backtester
│   ├── api/                     # FastAPI service
│   └── utils/                   # Logging, config, IO, metrics, plotting
├── tests/
│   ├── conftest.py              # Shared pytest fixtures
│   └── unit/                    # Unit tests for every layer
├── scripts/                     # 13 CLI entry points
├── notebooks/                   # Portfolio analysis notebooks + shared nb_style.py
├── config/                      # YAML configs (and .env example)
├── data/
│   ├── raw/                     # Immutable raw snapshots (incl. squad_values/)
│   ├── interim/                 # Unified canonical table
│   └── processed/               # Feature matrix
├── models/                      # Pickled model artifacts
├── outputs/
│   ├── predictions/             # Per-match probabilities + scorelines
│   ├── simulations/             # Aggregated tournament probabilities
│   ├── picks/                   # Quiniela pick sheets per profile
│   └── diagnostics/             # Calibration, feature importance, ratings
├── logs/
│   ├── pipeline/                # ETL + feature build logs
│   ├── training/                # Training logs
│   ├── prediction/              # Prediction logs
│   ├── updates/                 # Matchday update logs
│   └── errors/                  # Error logs
├── docker/                      # Dockerfile + docker-compose
├── reports/
│   ├── academic/                # LaTeX academic report
│   └── dashboard/               # LaTeX dashboard report
├── docs/                        # Documentation set (this folder)
├── run_all.bat                  # Full pipeline (Windows)
├── update_matchday.bat          # Quick matchday refresh (Windows)
├── Makefile                     # CLI shortcuts
├── requirements.txt             # Python dependencies
├── setup.py / pyproject.toml    # Packaging
├── README.md                    # Project overview
└── ARCHITECTURE.md              # High-level architecture notes
```

Each top-level directory is fully populated by the bootstrap pipeline; the `*.gitkeep`-style placeholders are replaced as soon as you run the scripts.
