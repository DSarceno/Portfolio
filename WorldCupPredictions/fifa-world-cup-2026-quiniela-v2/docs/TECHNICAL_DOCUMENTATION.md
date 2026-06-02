# Technical Documentation

## 1. System architecture

```
┌───────────────┐    ┌─────────────┐    ┌──────────────┐    ┌────────────┐
│  data sources │ -> │   data      │ -> │   ratings    │ -> │  features  │
│ (FIFA, FD,    │    │ (loader,    │    │ (Elo, PI,    │    │ (team,     │
│  StatsBomb)   │    │ validator,  │    │  form,       │    │  match,    │
│               │    │ updater)    │    │  ensemble)   │    │  fatigue,  │
│               │    │             │    │              │    │  market)   │
└───────────────┘    └─────────────┘    └──────────────┘    └────────────┘
                                                                  │
                                                                  v
                       ┌───────────────────────────────────┐
                       │             models                │
                       │ (multinomial, XGBoost, Poisson)   │
                       │ + calibration + blender           │
                       └───────────────────────────────────┘
                                       │
                                       v
                       ┌───────────────────────────────────┐
                       │  prediction + strategy + sim       │
                       │  (predictor, score_predictor,      │
                       │   quiniela_strategy, simulator)    │
                       └───────────────────────────────────┘
                                       │
                                       v
                       ┌───────────────────────────────────┐
                       │            API + scripts          │
                       └───────────────────────────────────┘
```

See [diagrams/system_architecture.md](diagrams/system_architecture.md) for a Mermaid version.

## 2. Module responsibilities

| Module          | Responsibility                                                            |
| --------------- | ------------------------------------------------------------------------- |
| `src/data/`     | Ingestion, validation, canonical table, tournament-state aggregations.    |
| `src/ratings/`  | Elo, PI, rolling form, and composite z-score ensemble.                    |
| `src/features/` | Feature engineering across team, match, fatigue, market, tournament.     |
| `src/models/`   | Outcome classifiers, Poisson scoreline, calibration.                      |
| `src/ensemble/` | Probability blending and risk-profile pick optimization.                  |
| `src/simulation/`| Group stage, bracket generation, knockout, tournament Monte-Carlo.       |
| `src/prediction/`| Match-level predictor, score recommendations, quiniela strategy, daily update. |
| `src/training/` | Trainer, Evaluator, Backtester, rolling-origin splits.                    |
| `src/api/`      | FastAPI endpoints exposing prediction, simulation, strategy, updates.    |
| `src/utils/`    | Logging, config, IO, dates, metrics, plotting, constants.                |

## 3. Data flow

1. **Ingest** — `ResultsCollector` invokes `FootballDataClient`, `StatsBombClient`, and `FifaRankingsClient`. Raw snapshots are persisted under `data/raw/`.
2. **Unify** — Raw rows are normalized into a canonical match table at `data/interim/matches_unified.csv`.
3. **Validate** — `validate_match_dataframe` checks schema, missing columns, negative scores, duplicates.
4. **Rate** — `RatingEnsemble().fit(matches)` produces composite team strengths.
5. **Engineer** — `build_match_feature_matrix` joins team and match features, plus tournament/fatigue/market features.
6. **Split** — `temporal_split` produces train/validation/test sets without leakage.
7. **Train** — `Trainer().fit(...)` fits multinomial, XGBoost, and Poisson models. Calibrator is fitted on validation probabilities.
8. **Predict** — `MatchPredictor.predict_proba()` blends and (optionally) calibrates probabilities.
9. **Strategize** — `QuinielaStrategy.generate_all(...)` outputs per-profile pick sheets.
10. **Simulate** — `TournamentSimulator.run(n_runs)` aggregates Monte-Carlo statistics.

## 4. Error handling

- Filesystem reads use `FileNotFoundError`-aware helpers (`load_csv`, `load_pickle`).
- External HTTP calls catch `httpx.HTTPError` and fall back to manual CSV.
- Model loading in `MatchPredictor` is permissive: missing artifacts disable a contribution without breaking the pipeline.
- All exceptions are logged with module-level loggers (`src.utils.logging_config`).

## 5. Logging architecture

`get_logger(__name__)` auto-configures the root logger with a console handler and a rotating file handler (10 MB × 5). Per-script log files live under `logs/<area>/...`.

## 6. Scalability

- XGBoost training is bound by feature-matrix size; 50K rows of historical matches fit in <1 GB.
- The Monte-Carlo simulator is embarrassingly parallel; the current implementation is single-process but can be parallelised by chunking `n_runs`.
- The API is stateless and can be scaled horizontally behind a load balancer.

## 7. Future extension points

- Live in-play probability updating from minute-by-minute event feeds.
- Player-level injury and suspension feeds plugged into `fatigue_features.py`.
- Bookmaker odds APIs for refining the public-bias proxy.
- Pool-specific quiniela scoring optimisation.
