# FIFA WORLD CUP 2026 QUINIELA PREDICTOR V2 - AGGRESSIVE PROJECT GENERATION GUIDE FOR CLAUDE CODE

## ⚠️ CRITICAL INSTRUCTIONS FOR CLAUDE CODE

Read this ENTIRE file first. This is the ONLY context you need.

You will:
1. Create ALL files in the correct directory structure
2. Generate COMPLETE Python code (no truncation)
3. Generate config files (YAML)
4. Generate notebooks, reports, and support scripts
5. Generate a production-ready prediction system oriented to **winning quinielas**
6. NOT ask questions - generate everything based on this specification

When you finish, the project should be 100% functional.

---

## 🎯 PROJECT OVERVIEW

**Project**: FIFA World Cup 2026 Quiniela Predictor V2  
**Goal**: Predict **all matches** of the 2026 FIFA World Cup, maximize practical quiniela performance, and update forecasts dynamically as the tournament progresses  
**Scope**:
- Pre-tournament power ratings
- Match-by-match outcome probabilities
- Scoreline probabilities
- Knockout advancement probabilities
- Bracket simulation
- Quiniela-oriented pick strategy
- Aggressive upset detection
- Daily re-training / incremental updating with newly played World Cup matches

**Tournament assumptions**:
- FIFA World Cup 2026
- 48 teams
- 12 groups of 4
- 104 total matches
- Top 2 from each group + 8 best third-placed teams advance
- Round of 32 → Round of 16 → Quarter-finals → Semi-finals → Third-place match → Final

---

## 🧠 PHILOSOPHY: THIS IS NOT A PURELY ACADEMIC MODEL

This project is **not** optimized only for log-loss or generic ML metrics.

It is optimized for **quinielas**, which means:
1. The best pick is not always the most probable pick
2. A slightly less likely upset can have higher strategic value
3. Confidence calibration matters
4. Tournament state matters
5. Public bias must be modeled when possible
6. Picks should be tiered into:
   - safe picks
   - balanced picks
   - aggressive picks
   - contrarian picks

The system must support both:
- **best probability prediction**
- **best game-theory quiniela pick**

---

## 🏆 SUCCESS CRITERIA

The final project must allow the user to:

1. Ingest historical international football data
2. Build team strength ratings before the tournament
3. Predict every World Cup 2026 match
4. Update models after each matchday using the newly played tournament matches
5. Produce recommended quiniela picks with different risk profiles
6. Simulate the entire tournament thousands of times
7. Generate bracket probabilities and likely paths
8. Track model performance throughout the World Cup
9. Re-rank teams after every new result
10. Export clean prediction tables for manual quiniela entry

---

## 📁 DIRECTORY STRUCTURE TO CREATE

````
fifa-world-cup-2026-quiniela-v2/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── sources.py
│   │   ├── fifa_rankings_client.py
│   │   ├── football_data_client.py
│   │   ├── statsbomb_client.py
│   │   ├── results_collector.py
│   │   ├── data_loader.py
│   │   ├── data_validator.py
│   │   ├── tournament_updater.py
│   │   └── data_splitter.py
│   ├── ratings/
│   │   ├── __init__.py
│   │   ├── elo.py
│   │   ├── pi_rating.py
│   │   ├── form_rating.py
│   │   └── rating_ensemble.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── build_features.py
│   │   ├── match_features.py
│   │   ├── team_features.py
│   │   ├── tournament_features.py
│   │   ├── market_features.py
│   │   └── fatigue_features.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_model.py
│   │   ├── multinomial_model.py
│   │   ├── poisson_model.py
│   │   ├── xgboost_model.py
│   │   ├── calibration.py
│   │   └── model_factory.py
│   ├── ensemble/
│   │   ├── __init__.py
│   │   ├── blender.py
│   │   └── pick_optimizer.py
│   ├── simulation/
│   │   ├── __init__.py
│   │   ├── group_stage.py
│   │   ├── knockout.py
│   │   ├── tournament_simulator.py
│   │   └── bracket_generator.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── evaluator.py
│   │   ├── backtester.py
│   │   └── cross_validation.py
│   ├── prediction/
│   │   ├── __init__.py
│   │   ├── predictor.py
│   │   ├── score_predictor.py
│   │   ├── quiniela_strategy.py
│   │   └── daily_update.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py
│   └── utils/
│       ├── __init__.py
│       ├── logging_config.py
│       ├── config.py
│       ├── io.py
│       ├── dates.py
│       ├── metrics.py
│       ├── plotting.py
│       └── constants.py
├── tests/
│   ├── __init__.py
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_ratings.py
│   │   ├── test_features.py
│   │   ├── test_models.py
│   │   ├── test_simulation.py
│   │   └── test_strategy.py
│   └── conftest.py
├── scripts/
│   ├── bootstrap_historical_data.py
│   ├── build_ratings.py
│   ├── run_pipeline.py
│   ├── train_models.py
│   ├── predict_group_stage.py
│   ├── predict_knockout.py
│   ├── simulate_tournament.py
│   ├── update_after_matchday.py
│   └── export_quiniela_sheet.py
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_ratings_diagnostics.ipynb
│   ├── 03_model_comparison.ipynb
│   └── 04_quiniela_strategy.ipynb
├── config/
│   ├── config.yaml
│   ├── features.yaml
│   ├── model_params.yaml
│   ├── strategy.yaml
│   └── .env.example
├── data/
│   ├── raw/
│   │   ├── fifa_rankings/
│   │   ├── football_data/
│   │   ├── statsbomb/
│   │   └── tournament_updates/
│   ├── interim/
│   └── processed/
├── models/
├── logs/
│   ├── pipeline/
│   ├── training/
│   ├── prediction/
│   ├── updates/
│   └── errors/
├── reports/
│   ├── academic/
│   │   ├── main.tex
│   │   └── sections/
│   └── dashboard/
│       ├── main.tex
│       └── sections/
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── outputs/
│   ├── predictions/
│   ├── simulations/
│   ├── picks/
│   └── diagnostics/
├── .env
├── .gitignore
├── README.md
├── ARCHITECTURE.md
├── Makefile
├── requirements.txt
├── setup.py
└── pyproject.toml
````

---

## 🔥 WHAT MAKES V2 AGGRESSIVE

This version must be more aggressive than a standard predictor because it is designed for **competitive quinielas**, not only statistical purity.

### Aggressive requirement 1: Separate outcome prediction from pick recommendation
Build:
- a model for true probabilities
- a strategy layer for actual picks

The strategy layer must decide when to:
- follow the favorite
- fade the favorite
- target draw probability
- exploit close matches
- select a calculated upset

### Aggressive requirement 2: Use multiple risk profiles
For each match generate:
- `safe_pick`
- `balanced_pick`
- `aggressive_pick`
- `contrarian_pick`

### Aggressive requirement 3: Detect upset windows
The project must explicitly flag matches where:
- favorite win probability is not dominant
- underdog recent form is improving
- favorite has unstable defense
- travel/rest burden is worse
- tournament incentive is asymmetric
- knockout volatility is high

### Aggressive requirement 4: Simulate bracket and future consequences
Predictions must not be isolated by match.
The system must estimate:
- bracket paths
- likely opponents
- rest day asymmetries
- incentive distortion in final group matches
- penalty shootout variance in knockouts

### Aggressive requirement 5: Tournament updates must be immediate
After each World Cup matchday:
- ingest official result
- update tournament form features
- update Elo / PI / recent form ratings
- update goal models
- refresh probabilities for remaining matches
- export new quiniela sheet

### Aggressive requirement 6: Optimize for quiniela scoring
Implement evaluation for custom quiniela scoring:
- 1X2 correctness
- exact score bonus
- upset bonus
- confidence-weighted pick value
- simulated pool-winning rate

### Aggressive requirement 7: Optional crowd/public-bias proxy
If configured, support public-bias approximation using:
- bookmaker implied probabilities (optional)
- FIFA ranking overreaction
- historical blue-blood bias
- team reputation bias
- host-nation overpricing
- recency overreaction

If no live odds source is available, approximate crowd bias from ranking gaps and reputation priors.

---

## 📊 DATA SOURCES STRATEGY

### Historical data sources
Implement modular extractors for these sources:

1. **FIFA Men’s World Ranking snapshots**
   - Use for official ranking history and ranking points
   - Store ranking date, team, rank, points

2. **StatsBomb Open Data**
   - Use when available for event-level or advanced historical match data
   - Extract shots, xG proxies if present, lineups, cards, possession-related proxies

3. **football-data.org**
   - Use for historical fixtures, teams, scores, competition metadata, standings when available
   - Also use as a machine-readable feed for current fixtures and results

4. **Tournament-specific manual CSV fallback**
   - If a source is missing a competition, create importers for manually maintained CSV files
   - Claude must generate templates for these CSVs

### During the World Cup
The project must support a priority order:

1. **Official FIFA fixtures/results pages as canonical truth**
2. **football-data.org API as machine-readable ingestion source**
3. **Local manual CSV override for emergency correction**
4. **Persist every update in `data/raw/tournament_updates/`**

Never hardcode match results.
Always preserve raw snapshots with timestamps.

---

## ⚙️ MODELING APPROACH

This project must combine multiple modeling families:

### 1. Team ratings
Implement:
- Elo rating
- PI rating
- rolling form rating
- home/host advantage adjustment
- confederation strength adjustment

### 2. Match outcome model
Predict:
- home win / draw / away win
- for neutral-site tournaments, still preserve team_a / team_b ordering and host-country features

Use:
- multinomial logistic regression
- XGBoost multiclass classifier
- calibrated ensemble

### 3. Scoreline model
Implement Poisson or Dixon-Coles-inspired score model using:
- attack strength
- defense strength
- recent goals for/against
- rest differential
- tournament pressure
- knockout vs group-stage context

### 4. Ensemble/blending layer
Blend:
- ratings-based probabilities
- multinomial classifier probabilities
- scoreline-implied probabilities
- recent tournament form probabilities

### 5. Calibration layer
Must implement:
- isotonic calibration
- Platt / sigmoid calibration
- reliability diagnostics

### 6. Strategy layer
Convert probabilities into picks:
- maximize raw expected correctness
- maximize upset upside
- maximize pool-winning rate in simulation
- support custom scoring rules

---

## 🧮 FEATURES TO ENGINEER

### Team strength features
- fifa_rank_points
- fifa_rank_position
- elo_pre_match
- pi_rating_pre_match
- rolling_form_points_last_5
- rolling_goal_diff_last_5
- rolling_xg_proxy_last_5
- attack_strength
- defense_strength
- clean_sheet_rate
- concede_rate
- draw_rate
- first_goal_rate
- comeback_rate
- penalty_shootout_history
- tournament_experience
- world_cup_experience
- host_flag
- same_confederation_flag
- confederation_strength_index

### Match-context features
- stage_group
- stage_knockout
- round_of_tournament
- rest_days_team_a
- rest_days_team_b
- rest_diff
- travel_burden_proxy_team_a
- travel_burden_proxy_team_b
- temperature_proxy
- altitude_proxy
- host_continent_advantage
- must_win_team_a
- must_win_team_b
- qualification_scenario_pressure
- elimination_risk_team_a
- elimination_risk_team_b

### Recent-tournament update features
- world_cup_points_so_far
- world_cup_goal_diff_so_far
- world_cup_goals_for_so_far
- world_cup_goals_against_so_far
- world_cup_clean_sheets
- world_cup_form_last_1
- world_cup_form_last_2
- world_cup_minutes_behind
- world_cup_minutes_ahead
- cards_accumulated
- suspension_risk_proxy

### Aggressive strategy features
- upset_window_score
- volatility_index
- draw_trap_score
- public_bias_proxy
- favorite_fragility_score
- underdog_live_value_score
- match_entropy
- confidence_gap
- penalty_risk_score

### Score-model features
- expected_goals_for_team_a
- expected_goals_for_team_b
- poisson_home_lambda
- poisson_away_lambda
- low_scoring_match_flag
- both_teams_to_score_proxy
- exact_score_concentration

---

## 🧪 TRAINING AND VALIDATION

### Split strategy
Use strict temporal validation only.
Never leak future information.

Suggested split:
- train: historical years before 2025
- validation: 2025
- test/backtest: pre-2026 and tournament-specific mock holdout windows

### Backtesting
Must include:
- past World Cup tournaments if available in historical data
- recent continental tournaments if available
- rolling-origin validation
- calibration tests
- upset-pick performance
- draw-pick performance
- simulated quiniela scoring

### Metrics
Track:
- log_loss
- brier_score
- accuracy_1x2
- macro_f1_1x2
- calibration_error
- top_pick_accuracy
- aggressive_pick_accuracy
- exact_score_accuracy
- average_points_under_custom_quiniela
- simulated_first_place_rate
- upset_hit_rate
- draw_hit_rate

---

## 🎲 SIMULATION ENGINE

Implement a full tournament simulator with:
- group-stage match simulation
- official tiebreak ordering
- best-third-place logic
- bracket generation for Round of 32
- knockout simulation with extra-time / penalties proxy
- full tournament repetition (at least 10,000 runs configurable)

Outputs:
- group qualification probabilities
- first-place and second-place probabilities
- best-third advancement probabilities
- round reached probabilities
- championship probabilities
- likely knockout paths
- match-up frequency matrices

---

## 🧾 QUINIELA STRATEGY MODULE

Create a dedicated strategy module.

For each upcoming match, export:
- most likely result
- safe pick
- balanced pick
- aggressive pick
- contrarian pick
- confidence score
- upset score
- draw danger score
- recommended scoreline
- rationale

Also export tournament-level strategy:
- safest matches
- best upset opportunities
- best draw opportunities
- overvalued favorites
- underrated teams
- matches to avoid overcommitting on

Support custom scoring formats such as:
- 1 point for correct 1X2
- 3 points for exact score
- bonus for unique upset
- weighted confidence selections

---

## 🌐 API REQUIREMENTS

Create FastAPI endpoints for:
- `/health`
- `/teams`
- `/matches/upcoming`
- `/matches/completed`
- `/predict/match`
- `/predict/day`
- `/predict/tournament`
- `/simulate/tournament`
- `/strategy/quiniela`
- `/update/results`
- `/diagnostics/calibration`
- `/diagnostics/feature-importance`

Use Pydantic models for all requests and responses.

---

## 📦 DEPENDENCIES

Create `requirements.txt` with at least:
````
numpy==1.24.4
pandas==2.1.4
scipy==1.11.4
scikit-learn==1.4.2
xgboost==2.0.3
statsmodels==0.14.2
fastapi==0.110.0
uvicorn==0.29.0
pydantic==2.6.4
python-dotenv==1.0.1
pyyaml==6.0.1
requests==2.31.0
httpx==0.27.0
tqdm==4.66.2
joblib==1.3.2
matplotlib==3.8.3
pytest==8.1.1
pytest-cov==4.1.0
black==24.3.0
isort==5.13.2
pylint==3.1.0
jupyter==1.0.0
notebook==7.1.2
````

---

## 📋 FILE: `.env`
````
ENV=development
DEBUG=true
LOG_LEVEL=INFO
RANDOM_SEED=42

DATA_DIR=./data
RAW_DATA_DIR=./data/raw
INTERIM_DATA_DIR=./data/interim
PROCESSED_DATA_DIR=./data/processed
MODELS_DIR=./models
OUTPUTS_DIR=./outputs

FIFA_RANKINGS_SOURCE=official
FOOTBALL_DATA_API_KEY=
STATSBOMB_OPEN_DATA_ENABLED=true
MANUAL_OVERRIDES_DIR=./data/raw/manual

TOURNAMENT_YEAR=2026
SIMULATION_RUNS=10000
DEFAULT_MODEL=ensemble
DEFAULT_RISK_PROFILE=balanced

API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=true

CONFIG_PATH=./config/config.yaml
FEATURES_CONFIG_PATH=./config/features.yaml
MODEL_PARAMS_PATH=./config/model_params.yaml
STRATEGY_CONFIG_PATH=./config/strategy.yaml
````

---

## 📋 FILE: `config/config.yaml`
````yaml
project:
  name: "FIFA World Cup 2026 Quiniela Predictor V2"
  version: "0.2.0"
  description: "Aggressive World Cup prediction system optimized for quinielas"

data:
  historical_start_year: 2014
  historical_end_year: 2026
  raw_dir: "data/raw"
  interim_dir: "data/interim"
  processed_dir: "data/processed"
  sources:
    fifa_rankings:
      enabled: true
      mode: "official_snapshot"
    football_data:
      enabled: true
      use_api: true
      require_api_key: false
    statsbomb_open:
      enabled: true
      use_open_data_only: true
    manual_csv:
      enabled: true
      dir: "data/raw/manual"

tournament:
  year: 2026
  teams: 48
  groups: 12
  group_size: 4
  total_matches: 104
  progression:
    auto_qualify_per_group: 2
    best_third_places: 8

training:
  random_seed: 42
  split_strategy: "temporal"
  validation_year: 2025
  backtest_windows: 5
  retrain_on_update: true

simulation:
  n_runs: 10000
  extra_time_enabled: true
  penalties_enabled: true

evaluation:
  metrics:
    - "log_loss"
    - "brier_score"
    - "accuracy_1x2"
    - "macro_f1_1x2"
    - "calibration_error"
    - "top_pick_accuracy"
    - "aggressive_pick_accuracy"
    - "exact_score_accuracy"
    - "upset_hit_rate"
    - "draw_hit_rate"
    - "simulated_first_place_rate"

logging:
  level: "INFO"
  format: "%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s"
  file: "logs/world_cup_quiniela_v2.log"

paths:
  models_dir: "models"
  outputs_dir: "outputs"
  reports_dir: "reports"
  logs_pipeline: "logs/pipeline"
  logs_training: "logs/training"
  logs_prediction: "logs/prediction"
  logs_updates: "logs/updates"
  logs_errors: "logs/errors"
````

---

## 📋 FILE: `config/features.yaml`
````yaml
feature_groups:
  team_strength:
    enabled: true
    features:
      - fifa_rank_points
      - fifa_rank_position
      - elo_pre_match
      - pi_rating_pre_match
      - rolling_form_points_last_5
      - rolling_goal_diff_last_5
      - attack_strength
      - defense_strength
      - clean_sheet_rate
      - concede_rate
      - draw_rate
      - tournament_experience
      - world_cup_experience
      - confederation_strength_index
      - host_flag

  match_context:
    enabled: true
    features:
      - stage_group
      - stage_knockout
      - round_of_tournament
      - rest_days_team_a
      - rest_days_team_b
      - rest_diff
      - travel_burden_proxy_team_a
      - travel_burden_proxy_team_b
      - must_win_team_a
      - must_win_team_b
      - elimination_risk_team_a
      - elimination_risk_team_b
      - qualification_scenario_pressure

  tournament_updates:
    enabled: true
    features:
      - world_cup_points_so_far
      - world_cup_goal_diff_so_far
      - world_cup_goals_for_so_far
      - world_cup_goals_against_so_far
      - world_cup_clean_sheets
      - world_cup_form_last_1
      - world_cup_form_last_2
      - cards_accumulated
      - suspension_risk_proxy

  aggressive_strategy:
    enabled: true
    features:
      - upset_window_score
      - volatility_index
      - draw_trap_score
      - public_bias_proxy
      - favorite_fragility_score
      - underdog_live_value_score
      - match_entropy
      - confidence_gap
      - penalty_risk_score

selection:
  method: "model_based"
  max_features_to_keep: 50
  remove_high_correlation: true
  correlation_threshold: 0.95
````

---

## 📋 FILE: `config/model_params.yaml`
````yaml
models:
  multinomial:
    enabled: true
    primary: false
    hyperparameters:
      C: 1.0
      max_iter: 1000
      class_weight: "balanced"

  xgboost:
    enabled: true
    primary: true
    hyperparameters:
      objective: "multi:softprob"
      num_class: 3
      max_depth: 5
      learning_rate: 0.05
      n_estimators: 400
      subsample: 0.85
      colsample_bytree: 0.85
      reg_alpha: 0.0
      reg_lambda: 1.0
      random_state: 42
      n_jobs: -1

  poisson:
    enabled: true
    primary: false
    hyperparameters:
      max_goals: 8
      use_dixon_coles_adjustment: true
      low_score_correlation: true

calibration:
  enabled: true
  methods:
    - "isotonic"
    - "sigmoid"

ensemble:
  enabled: true
  method: "weighted_blend"
  weights:
    ratings: 0.25
    multinomial: 0.20
    xgboost: 0.30
    poisson: 0.25
````

---

## 📋 FILE: `config/strategy.yaml`
````yaml
quiniela:
  default_risk_profile: "balanced"

  risk_profiles:
    safe:
      favorite_threshold: 0.58
      upset_tolerance: 0.05
      draw_bias: 0.90

    balanced:
      favorite_threshold: 0.50
      upset_tolerance: 0.10
      draw_bias: 1.00

    aggressive:
      favorite_threshold: 0.43
      upset_tolerance: 0.18
      draw_bias: 1.10

    contrarian:
      favorite_threshold: 0.38
      upset_tolerance: 0.25
      draw_bias: 1.15

  upset_detection:
    min_entropy: 0.90
    min_underdog_probability: 0.22
    max_favorite_probability: 0.58
    min_fragility_score: 0.50

  draw_detection:
    min_draw_probability: 0.26
    low_goal_total_bonus: true

  export:
    include_rationale: true
    include_confidence: true
    include_scoreline: true
````

---

## 🔧 CODE STANDARDS

### Python requirements
- Type hints in 100% of functions
- Google-style docstrings in all classes and functions
- Logging everywhere with `logging`
- No `print()` in production code
- Specific exception handling
- Black-compatible formatting
- stdlib > third-party > local imports
- Keep functions focused and reasonably short

---

## 🧱 MODULE REQUIREMENTS

### `src/data/`
Claude must implement robust data connectors:
- `fifa_rankings_client.py`
  - load ranking snapshots
  - parse ranking tables
  - cache raw ranking data
- `football_data_client.py`
  - pull fixtures, results, standings where available
  - support API-key and local cached modes
- `statsbomb_client.py`
  - download/open local StatsBomb open datasets
  - normalize match-level data
- `results_collector.py`
  - unify historical and live match data
- `tournament_updater.py`
  - append new World Cup matches
  - compute tournament state updates
  - trigger re-rating and re-prediction
- `data_loader.py`
  - create canonical match table
- `data_validator.py`
  - validate schemas and missingness
- `data_splitter.py`
  - strict temporal split

### `src/ratings/`
Implement:
- Elo with configurable K factor and tournament multiplier
- PI rating with home/away-neutral support
- rolling form rating
- ensemble of ratings into a composite team power score

### `src/features/`
Implement builders for:
- match features
- team features
- tournament-state features
- market/public-bias proxy features
- fatigue/rest/travel features

### `src/models/`
Implement:
- abstract base model
- multinomial classifier
- Poisson score model
- XGBoost outcome model
- calibration utilities
- model factory

### `src/ensemble/`
Implement:
- probability blending
- pick optimization logic
- risk-profile based pick generation

### `src/simulation/`
Implement:
- group table updates
- FIFA tiebreak ordering
- best-third ranking
- knockout bracket builder
- tournament simulation engine

### `src/prediction/`
Implement:
- single-match prediction
- day slate prediction
- exact score recommendation
- quiniela pick recommendation
- daily update workflow

---

## 📜 ROOT FILES TO GENERATE

Generate complete versions of:
- `.env`
- `.gitignore`
- `README.md`
- `ARCHITECTURE.md`
- `Makefile`
- `requirements.txt`
- `setup.py`
- `pyproject.toml`

Also create:
- Docker files
- test files
- notebooks
- LaTeX reports
- CSV templates for manual overrides

---

## 📈 REQUIRED OUTPUT FILES

The project must export:

1. `outputs/predictions/group_stage_predictions.csv`
2. `outputs/predictions/knockout_predictions.csv`
3. `outputs/picks/quiniela_safe.csv`
4. `outputs/picks/quiniela_balanced.csv`
5. `outputs/picks/quiniela_aggressive.csv`
6. `outputs/picks/quiniela_contrarian.csv`
7. `outputs/simulations/tournament_probabilities.csv`
8. `outputs/simulations/bracket_paths.csv`
9. `outputs/diagnostics/calibration_report.csv`
10. `outputs/diagnostics/model_comparison.csv`

---

## 🧾 REQUIRED SCRIPTS

Create these scripts:

1. `scripts/bootstrap_historical_data.py`
   - build historical database

2. `scripts/build_ratings.py`
   - compute all pre-tournament ratings

3. `scripts/run_pipeline.py`
   - full ETL + feature pipeline

4. `scripts/train_models.py`
   - train and save models

5. `scripts/predict_group_stage.py`
   - predict all current/future group-stage matches

6. `scripts/predict_knockout.py`
   - predict knockout matches once bracket is known

7. `scripts/simulate_tournament.py`
   - run tournament Monte Carlo

8. `scripts/update_after_matchday.py`
   - ingest just-played matches and refresh outputs

9. `scripts/export_quiniela_sheet.py`
   - export human-readable picks for friends/pools

---

## 🧪 TESTING REQUIREMENTS

Write unit tests for:
- rating updates
- feature generation
- probability constraints summing to 1
- scoreline generation
- group table ranking rules
- best-third selection
- bracket generation
- strategy pick logic
- daily update pipeline

Target:
- meaningful tests
- deterministic fixtures
- minimum 80% coverage structure

---

## 📝 README REQUIREMENTS

README must include:
- project goal
- setup
- data sources
- how updates work during the World Cup
- how to export quiniela picks
- explanation of safe/balanced/aggressive/contrarian profiles
- examples of expected output
- limitations and disclaimer that football remains uncertain

---

## 🏗️ ARCHITECTURE REQUIREMENTS

ARCHITECTURE.md must explain:
- why ratings + ML + Poisson are combined
- why calibration matters
- why tournament updates matter
- why quiniela strategy is separate from pure forecasting
- why simulation is needed for bracket-aware decisions

---

## ⚠️ CRITICAL NOTES

**DO NOT:**
- Ask for clarification
- Create incomplete files
- Truncate code
- Skip files
- Assume static tournament state

**DO:**
- Create every listed file
- Make all code complete
- Generate all configs
- Generate tests
- Generate reports
- Implement update-after-matchday workflow
- Preserve raw snapshots of every external data pull
- Keep the code runnable

---

## ✅ FINAL VALIDATION

When finished, the project must:
- Predict all World Cup 2026 matches
- Update itself as new matches are played
- Output multiple quiniela risk-profile pick sheets
- Simulate the full tournament
- Recompute team strengths after each matchday
- Provide API endpoints and scripts
- Be ready to run end-to-end after dependency installation

---

## 📚 ADDITIONAL DOCUMENTATION REQUIREMENTS

In addition to all previously requested files, generate a complete documentation package.

### Create the following files

```text
docs/
├── USER_GUIDE.md
├── TECHNICAL_DOCUMENTATION.md
├── INSTALLATION_GUIDE.md
├── OPERATIONS_RUNBOOK.md
├── API_DOCUMENTATION.md
├── DATA_DICTIONARY.md
├── MODELING_GUIDE.md
├── RETRAINING_GUIDE.md
├── TROUBLESHOOTING.md
├── DEPLOYMENT_GUIDE.md
├── PROJECT_STRUCTURE.md
└── diagrams/
    ├── system_architecture.md
    ├── data_flow.md
    └── model_pipeline.md
```

### USER_GUIDE.md

Must explain:

* Project purpose
* Quick start
* Installation
* Configuration
* Running the full pipeline
* Training models
* Running simulations
* Generating quiniela picks
* Updating after matchdays
* Exporting reports
* Understanding outputs
* Common workflows
* Examples with commands and expected outputs

### TECHNICAL_DOCUMENTATION.md

Must explain:

* Complete system architecture
* Design decisions
* Module responsibilities
* Internal dependencies
* Data flow
* Feature engineering pipeline
* Rating systems implementation
* Modeling layer
* Ensemble logic
* Simulation engine
* Strategy engine
* Update workflow
* Error handling
* Logging architecture
* Scalability considerations
* Future extension points

### API_DOCUMENTATION.md

For every endpoint include:

* Purpose
* Request schema
* Response schema
* Example request
* Example response
* Error responses

### DATA_DICTIONARY.md

Document every dataset and feature:

* Name
* Type
* Source
* Description
* Update frequency
* Nullable status

Include all engineered features.

### MODELING_GUIDE.md

Explain:

* Elo implementation
* PI Rating implementation
* Form rating
* Poisson model
* Multinomial model
* XGBoost model
* Calibration methods
* Ensemble weighting
* Quiniela optimization logic
* Backtesting methodology

### RETRAINING_GUIDE.md

Explain:

* How to refresh data
* How to retrain models
* How to update ratings
* How to rebuild features
* How to regenerate predictions
* How to recover from failed updates

### OPERATIONS_RUNBOOK.md

Provide procedures for:

* Daily World Cup updates
* Matchday updates
* Monitoring
* Validation checks
* Backup procedures
* Recovery procedures
* Production maintenance

### TROUBLESHOOTING.md

Include:

* Common installation errors
* API issues
* Data ingestion failures
* Model training failures
* Simulation failures
* Prediction failures
* Recommended fixes

### DEPLOYMENT_GUIDE.md

Include:

* Local deployment
* Docker deployment
* Server deployment
* Environment variables
* Production recommendations

### PROJECT_STRUCTURE.md

Explain every major directory and file.

### Diagram Documentation

Generate Mermaid diagrams for:

* System architecture
* Data flow
* Training pipeline
* Prediction pipeline
* Tournament update workflow
* API interactions

### Documentation Quality Requirements

* Documentation must be production-grade.
* Every document must be complete.
* Include examples, command snippets, and expected outputs.
* Assume a new engineer must be able to maintain the system using only the documentation.
* Assume a non-technical user must be able to generate quiniela predictions using only the user documentation.
* Do not leave placeholders or TODO sections.
* Generate fully written documentation.

---

# 🚨 LARGE PROJECT GENERATION EXECUTION PROTOCOL

This project is intentionally very large and may exceed a single response window.

You MUST follow the execution protocol below.

## PRIMARY OBJECTIVE

The objective is NOT to explain the project.

The objective is to GENERATE THE ENTIRE REPOSITORY.

A partially generated repository is considered a failure.

---

## RESPONSE CONTINUATION RULE

If output limits are reached:

* DO NOT summarize
* DO NOT stop with placeholders
* DO NOT omit files
* DO NOT write "remaining files omitted"
* DO NOT write "continue generation manually"

Instead:

1. Finish the current file completely.
2. Record generation state internally.
3. Automatically continue with the next files.
4. Continue producing repository files until every required file has been generated.

Treat all responses as a continuous repository-generation session.

---

## NO PLACEHOLDER POLICY

Forbidden:

```python
# TODO
pass

# Implementation omitted

# Placeholder

# Simplified version

# Pseudo-code
```

Every file must contain working code.

Every class must be implemented.

Every function must be implemented.

Every configuration file must be complete.

---

## COMPLETENESS VALIDATION

Before considering generation complete, verify internally that:

* Every directory exists
* Every file listed in the specification exists
* Every import resolves
* Every configuration file exists
* Every script exists
* Every notebook exists
* Every report exists
* Every API endpoint exists
* Every test file exists
* Every required output schema is implemented
* Documentation files exist
* Docker files exist

Only finish after all required artifacts have been generated.

---

## FILE GENERATION ORDER

Generate files in this order:

1. Root files
2. Configuration
3. Utilities
4. Data layer
5. Ratings layer
6. Feature layer
7. Models
8. Ensemble layer
9. Simulation layer
10. Prediction layer
11. Training layer
12. API layer
13. Scripts
14. Tests
15. Notebooks
16. Reports
17. Documentation
18. Docker
19. Final repository validation

Never skip an earlier layer.

---

## TOKEN BUDGET OPTIMIZATION

Avoid unnecessary prose.

Minimize explanations.

Prioritize code generation over commentary.

Generate only what is required to build the repository.

---

## SELF-AUDIT PHASE

After generating all files, perform a repository audit and generate:

```text
REPOSITORY_COMPLETENESS_REPORT.md
```

This report must include:

* Total files generated
* Total modules generated
* Total scripts generated
* Total tests generated
* Missing files (if any)
* Dependency validation
* Import validation
* Configuration validation
* API validation
* Documentation validation

If any required file is missing, generate it before ending.

---

## FINAL SUCCESS CONDITION

The generation is complete only when:

* The repository can be created directly from the generated output.
* No required file remains ungenerated.
* No placeholder code exists.
* Documentation is complete.
* Tests are present.
* API is present.
* Training pipeline is present.
* Tournament update workflow is present.
* Quiniela strategy layer is present.
* Simulation engine is present.

Do not stop before reaching this state.

---

## CLAUDE CODE FILE CREATION MODE

You are running inside Claude Code.

Do NOT output repository contents into chat unless necessary.

Instead:

* Create files directly on disk.
* Write code directly into the repository structure.
* Create every required directory.
* Create every required file.
* Overwrite existing files when necessary.
* Use the filesystem as the primary output channel.
* Use chat only for progress reporting and final validation.

The deliverable is the generated repository itself, not the conversation output.

Success means all files exist in the workspace.
Failure means any required file is missing.

---

## 🎯 FINAL COMMAND TO FOLLOW

**GENERATE ALL FILES NOW. NO QUESTIONS. COMPLETE PROJECT GENERATION.**
