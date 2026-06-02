# FIFA WORLD CUP 2026 PREDICTOR - PROJECT GENERATION GUIDE FOR CLAUDE CODE

## ⚠️ CRITICAL INSTRUCTIONS FOR CLAUDE CODE

Read this ENTIRE file first. This is the ONLY context you need.

You will:
1. Create ALL files in the correct directory structure
2. Generate COMPLETE Python code (no truncation)
3. Generate config files (YAML)
4. Generate LaTeX reports
5. Generate requirements.txt and support files
6. Generate tests and reproducible pipelines
7. NOT ask questions - generate everything based on this specification

When you finish, the project should be 100% functional and ready to run locally.

---

## 🎯 PROJECT OVERVIEW

**Project**: FIFA World Cup 2026 Predictor  
**Goal**: Predict every FIFA World Cup 2026 match using ML + rating systems + simulation  
**Primary Use Case**: Help a user generate pre-match probabilities, likely scorelines, progression paths, and bracket predictions for quinielas  
**Tournament Context**: FIFA World Cup 2026 is the first edition with **48 teams**, **104 matches**, and is hosted by **Canada, Mexico and the United States**. The official tournament format uses **12 groups of 4**, where the **top two teams in each group plus the eight best third-placed teams** advance to the Round of 32. citeturn849321search1turn849321search7turn849321search20

**Critical Requirement**: As World Cup matches are played, those results MUST be ingested automatically into the project so the models update features, ratings, calibration and future predictions before the next matches.

**Outputs**:
- Match win/draw/loss probabilities
- Expected goals and likely scorelines
- Qualification probabilities from group stage
- Knockout bracket advancement probabilities
- Team strength rankings
- Daily refreshed predictions during the tournament
- API responses, CSV outputs, JSON outputs, and PDF reports

---

## 🧠 MODELING PHILOSOPHY

This project is NOT a single-model toy script.

It must combine:
1. **Rating systems**: Elo, form-adjusted Elo, attack/defense ratings
2. **Machine learning**: XGBoost / LightGBM / CatBoost / scikit-learn models
3. **Probabilistic modeling**: Poisson / Dixon-Coles style expected-goals framework
4. **Tournament simulation**: Monte Carlo simulation for groups and knockout rounds
5. **Online updates**: Incremental feature refresh after every newly played World Cup match
6. **Ensembling**: Blend model probabilities into a calibrated final prediction

The project must be opinionated, production-style, and fully reproducible.

---

## 📁 DIRECTORY STRUCTURE TO CREATE
````
world-cup-2026-predictor/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── fifa_client.py
│   │   ├── football_data_client.py
│   │   ├── data_loader.py
│   │   ├── historical_builder.py
│   │   ├── tournament_updater.py
│   │   ├── data_validator.py
│   │   └── data_splitter.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── build_features.py
│   │   ├── team_features.py
│   │   ├── match_features.py
│   │   ├── tournament_features.py
│   │   ├── rating_features.py
│   │   └── rest_travel_features.py
│   ├── ratings/
│   │   ├── __init__.py
│   │   ├── elo.py
│   │   ├── form_elo.py
│   │   ├── attack_defense.py
│   │   └── rating_manager.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_model.py
│   │   ├── baseline.py
│   │   ├── xgboost_model.py
│   │   ├── lightgbm_model.py
│   │   ├── catboost_model.py
│   │   ├── poisson_model.py
│   │   ├── ensemble_model.py
│   │   └── model_factory.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── evaluator.py
│   │   ├── calibrator.py
│   │   └── cross_validation.py
│   ├── simulation/
│   │   ├── __init__.py
│   │   ├── scoreline.py
│   │   ├── group_stage.py
│   │   ├── knockout.py
│   │   └── tournament_simulator.py
│   ├── prediction/
│   │   ├── __init__.py
│   │   ├── predictor.py
│   │   ├── daily_refresh.py
│   │   └── scenario_runner.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py
│   └── utils/
│       ├── __init__.py
│       ├── logging_config.py
│       ├── config.py
│       ├── metrics.py
│       ├── plotting.py
│       ├── dates.py
│       ├── io.py
│       └── seeds.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_data_loader.py
│   │   ├── test_features.py
│   │   ├── test_ratings.py
│   │   ├── test_models.py
│   │   ├── test_simulation.py
│   │   └── test_api.py
│   └── integration/
│       ├── __init__.py
│       ├── test_pipeline.py
│       └── test_daily_refresh.py
├── scripts/
│   ├── run_pipeline.py
│   ├── train_models.py
│   ├── predict_matches.py
│   ├── simulate_tournament.py
│   ├── refresh_after_matchday.py
│   ├── backfill_world_cup_results.py
│   └── serve_api.py
├── config/
│   ├── config.yaml
│   ├── features.yaml
│   ├── model_params.yaml
│   ├── tournament.yaml
│   └── .env.example
├── data/
│   ├── raw/
│   │   ├── fifa/
│   │   ├── football_data/
│   │   ├── rankings/
│   │   └── cached_responses/
│   ├── interim/
│   ├── processed/
│   └── external/
├── models/
│   ├── artifacts/
│   ├── calibrated/
│   └── snapshots/
├── logs/
│   ├── pipeline/
│   ├── training/
│   ├── prediction/
│   ├── refresh/
│   └── errors/
├── reports/
│   ├── academic/
│   │   ├── main.tex
│   │   ├── sections/
│   │   │   ├── 01_abstract.tex
│   │   │   ├── 02_introduction.tex
│   │   │   ├── 03_data.tex
│   │   │   ├── 04_methodology.tex
│   │   │   ├── 05_results.tex
│   │   │   └── 06_conclusions.tex
│   │   └── figures/
│   └── dashboard/
│       ├── main.tex
│       └── sections/
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── .env
├── .gitignore
├── README.md
├── ARCHITECTURE.md
├── Makefile
├── requirements.txt
├── setup.py
├── pyproject.toml
└── outputs/
    ├── predictions/
    ├── simulations/
    ├── reports/
    └── leaderboards/
````

---

## ⚙️ CORE PRODUCT REQUIREMENTS

### 1. Historical Scope
Build a training dataset using:
- International matches from **2010 onward** by default
- Stronger weighting for recent matches
- Extra emphasis on:
  - World Cup
  - continental tournaments
  - Nations League / competitive qualifiers
  - World Cup qualifiers
  - matches against top-30 opponents

### 2. Match Prediction Targets
The project must predict at minimum:
- `home_win_probability`
- `draw_probability`
- `away_win_probability`
- `expected_goals_home`
- `expected_goals_away`
- `most_likely_scoreline`
- `both_teams_to_score_probability`
- `over_2_5_goals_probability`
- `team_to_qualify_probability` for knockout ties

### 3. Tournament Simulation
The simulator must:
- Simulate the full group stage
- Correctly apply group ranking tie-breakers
- Identify the 8 best third-placed teams
- Build the Round of 32 bracket correctly
- Simulate knockout matches with extra time + penalties logic
- Output probabilities for each team to reach:
  - Round of 32
  - Round of 16
  - Quarter-finals
  - Semi-finals
  - Final
  - Champion

### 4. Incremental Updates During the World Cup
After each played World Cup match, the project MUST:
- Fetch official result data
- Append the match into the processed tournament dataset
- Update Elo and attack/defense ratings
- Recompute rolling form features
- Recalibrate probabilities if configured
- Regenerate predictions for all remaining matches
- Save a model/rating snapshot with timestamp
- Create new daily JSON and CSV outputs

This is mandatory.

### 5. Practical Quiniela Focus
Besides raw model accuracy, optimize for:
- picking likely winners
- identifying high-confidence draws
- highlighting upset risk
- ranking matches by confidence
- generating one “safe” and one “aggressive” quiniela suggestion

---

## 🔧 CODE STANDARDS (FOLLOW THESE EXACTLY)

### Python Code Requirements
- **Type Hints**: 100% in ALL functions
- **Docstrings**: Google style, detailed for all functions and classes
- **Logging**: Use logging module, NOT print()
- **Error Handling**: Try-catch with specific exceptions
- **Code Style**: Black formatter (88 char line length)
- **Imports**: Group as stdlib > third-party > local
- **Max Function Length**: 50 lines when reasonably possible
- **Max Class**: No limit, but keep methods focused
- **No notebooks**: everything must be production Python modules and scripts
- **No hidden magic**: all parameters configurable in YAML

### Example Structure
````python
"""Module description."""

from pathlib import Path
from typing import Dict, List, Optional

import logging
import pandas as pd

logger = logging.getLogger(__name__)


class MyClass:
    """Class description.

    Detailed explanation if needed.
    """

    def __init__(self, name: str) -> None:
        """Initialize object.

        Args:
            name: Resource name.
        """
        self.name = name
        logger.info("Initialized MyClass with name=%s", name)

    def run(self, data: pd.DataFrame) -> Dict[str, float]:
        """Execute operation.

        Args:
            data: Input data.

        Returns:
            Dictionary with summary values.

        Raises:
            ValueError: If data is empty.
        """
        if data.empty:
            raise ValueError("Input data cannot be empty.")

        try:
            return {"rows": float(len(data))}
        except Exception as exc:
            logger.error("Run failed: %s", exc, exc_info=True)
            raise
````

---

## 📦 DEPENDENCIES

Create `requirements.txt`:
````
# Core
numpy==1.26.4
pandas==2.2.2
scipy==1.13.1
pyarrow==16.1.0

# ML
scikit-learn==1.5.1
xgboost==2.1.1
lightgbm==4.5.0
catboost==1.2.7
statsmodels==0.14.2

# API / validation
fastapi==0.115.0
uvicorn==0.30.6
pydantic==2.9.2

# HTTP / scraping / parsing
requests==2.32.3
httpx==0.27.2
beautifulsoup4==4.12.3
lxml==5.3.0

# Config / utils
python-dotenv==1.0.1
pyyaml==6.0.2
tqdm==4.66.5
joblib==1.4.2

# Visualization
matplotlib==3.9.2
plotly==5.24.1

# Testing
pytest==8.3.3
pytest-cov==5.0.0
pytest-mock==3.14.0

# Code Quality
black==24.8.0
isort==5.13.2
pylint==3.3.1
mypy==1.11.2
````

---

## 📋 FILE: `.env` (in root)
````bash
# General
ENV=development
DEBUG=true
LOG_LEVEL=INFO
RANDOM_SEED=42

# Data
DATA_DIR=./data
RAW_DATA_DIR=./data/raw
INTERIM_DATA_DIR=./data/interim
PROCESSED_DATA_DIR=./data/processed
EXTERNAL_DATA_DIR=./data/external
CACHE_DIR=./data/raw/cached_responses

# Historical data scope
START_YEAR=2010
END_YEAR=2026
TOURNAMENT_NAME=FIFA World Cup 2026

# Model
MODEL_DIR=./models
DEFAULT_MODEL=ensemble
TEST_SIZE=0.15
VALIDATION_SIZE=0.15
CV_FOLDS=5

# API
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=2
API_RELOAD=true

# Paths
CONFIG_PATH=./config/config.yaml
FEATURES_CONFIG_PATH=./config/features.yaml
MODEL_PARAMS_PATH=./config/model_params.yaml
TOURNAMENT_CONFIG_PATH=./config/tournament.yaml
REPORTS_DIR=./reports
OUTPUTS_DIR=./outputs

# Optional keys
FOOTBALL_DATA_API_KEY=
ANTHROPIC_API_KEY=
````

---

## 📋 FILE: `config/.env.example`
````bash
# Copy this file to .env and update values

ENV=development
DEBUG=true
LOG_LEVEL=INFO
RANDOM_SEED=42

START_YEAR=2010
END_YEAR=2026

FOOTBALL_DATA_API_KEY=
ANTHROPIC_API_KEY=

API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=2
API_RELOAD=true

DATA_DIR=./data
MODEL_DIR=./models
REPORTS_DIR=./reports
OUTPUTS_DIR=./outputs
````

---

## 📋 FILE: `config/config.yaml`
````yaml
project:
  name: "FIFA World Cup 2026 Predictor"
  version: "0.1.0"
  description: "Predict every FIFA World Cup 2026 match using ratings, ML and simulation"

competition:
  tournament_name: "FIFA World Cup 2026"
  hosts:
    - "Canada"
    - "Mexico"
    - "United States"
  year: 2026

history:
  start_year: 2010
  end_year: 2026
  preferred_competitions:
    - "FIFA World Cup"
    - "FIFA World Cup qualification"
    - "UEFA Euro"
    - "Copa America"
    - "UEFA Nations League"
    - "AFC Asian Cup"
    - "Africa Cup of Nations"
    - "CONCACAF Gold Cup"
  recent_weight_half_life_days: 365
  max_matches_per_team: 150

data:
  raw_dir: "data/raw"
  interim_dir: "data/interim"
  processed_dir: "data/processed"
  external_dir: "data/external"
  cached_responses_dir: "data/raw/cached_responses"
  allow_scraping_fallback: true
  validation:
    max_null_percentage: 0.25
    min_rows_for_training: 1000
    require_world_cup_matches_file: true

training:
  test_size: 0.15
  validation_size: 0.15
  cv_folds: 5
  random_seed: 42
  split_strategy: "temporal"
  class_target: "match_outcome"

probabilities:
  calibration_method: "isotonic"
  renormalize_after_calibration: true
  max_scoreline_goals: 7

refresh:
  enabled: true
  auto_update_after_new_result: true
  save_snapshots: true
  snapshot_prefix: "wc2026"
  regenerate_remaining_predictions: true

logging:
  level: "INFO"
  format: "%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s"
  date_format: "%Y-%m-%d %H:%M:%S"
  file: "logs/world_cup_predictor.log"
  max_bytes: 10485760
  backup_count: 5

paths:
  models_dir: "models"
  outputs_dir: "outputs"
  reports_dir: "reports"
  logs_pipeline: "logs/pipeline"
  logs_training: "logs/training"
  logs_prediction: "logs/prediction"
  logs_refresh: "logs/refresh"
  logs_errors: "logs/errors"
````

---

## 📋 FILE: `config/features.yaml`
````yaml
feature_groups:
  team_features:
    enabled: true
    features:
      - fifa_rank_proxy
      - elo_pre_match
      - form_elo_pre_match
      - attack_rating_pre_match
      - defense_rating_pre_match
      - rolling_points_last_5
      - rolling_goal_diff_last_5
      - rolling_xg_proxy_last_5
      - rolling_xga_proxy_last_5
      - clean_sheet_rate_last_10
      - scoring_rate_last_10
      - non_penalty_goals_rate_last_10
      - set_piece_goal_rate_last_10
      - top30_opponent_points_rate
      - tournament_experience_score
      - world_cup_experience_score
      - penalty_shootout_strength
      - manager_stability_score
      - squad_continuity_score
      - confederation_strength_score

  match_features:
    enabled: true
    features:
      - is_host_team
      - same_confederation_match
      - neutral_site_flag
      - rest_days_difference
      - travel_burden_difference
      - altitude_proxy_difference
      - temperature_proxy_difference
      - market_strength_gap
      - elo_gap
      - form_elo_gap
      - attack_defense_interaction
      - goals_for_rate_gap
      - goals_against_rate_gap
      - draw_rate_combined
      - upset_risk_index

  tournament_features:
    enabled: true
    features:
      - stage_group
      - stage_round32
      - stage_round16
      - stage_quarterfinal
      - stage_semifinal
      - stage_final
      - must_win_flag
      - qualification_pressure_score
      - prior_group_points
      - prior_group_goal_diff
      - best_third_place_pressure
      - knockout_extra_time_enabled

  rating_features:
    enabled: true
    features:
      - elo_momentum
      - attack_trend
      - defense_trend
      - result_streak_score
      - weighted_recent_performance
      - opponent_strength_adjusted_form

  rest_travel_features:
    enabled: true
    features:
      - days_since_last_match_home
      - days_since_last_match_away
      - timezone_shift_home
      - timezone_shift_away
      - distance_since_last_venue_home
      - distance_since_last_venue_away
      - travel_fatigue_gap

selection:
  method: "model_importance_and_correlation"
  correlation_threshold: 0.95
  max_features_to_keep: 45
  min_features_to_keep: 20

missing_values:
  strategy: "groupwise"
  defaults:
    numeric: "median"
    categorical: "most_frequent"
    rating_features: "forward_fill"

scaling:
  method: "standard"
  apply_to_linear_or_distance_models_only: true

engineering:
  create_interactions: true
  create_lag_features: true
  lag_periods: [1, 3, 5, 10]
  add_confederation_strength: true
  add_time_decay_weights: true
````

---

## 📋 FILE: `config/model_params.yaml`
````yaml
models:
  baseline:
    enabled: true
    primary: false
    strategy: "elo_rule_based"

  xgboost:
    enabled: true
    primary: false
    hyperparameters:
      objective: "multi:softprob"
      num_class: 3
      max_depth: 6
      learning_rate: 0.05
      n_estimators: 400
      subsample: 0.9
      colsample_bytree: 0.9
      reg_alpha: 0.0
      reg_lambda: 1.0
      min_child_weight: 3
      random_state: 42
      n_jobs: -1

  lightgbm:
    enabled: true
    primary: false
    hyperparameters:
      objective: "multiclass"
      num_class: 3
      learning_rate: 0.05
      n_estimators: 500
      num_leaves: 31
      max_depth: -1
      subsample: 0.9
      colsample_bytree: 0.9
      random_state: 42

  catboost:
    enabled: true
    primary: false
    hyperparameters:
      loss_function: "MultiClass"
      depth: 6
      learning_rate: 0.05
      iterations: 500
      random_seed: 42
      verbose: false

  poisson:
    enabled: true
    primary: false
    hyperparameters:
      max_goals: 7
      home_advantage: 0.0
      dixon_coles_rho: -0.05

  ensemble:
    enabled: true
    primary: true
    weights:
      xgboost: 0.25
      lightgbm: 0.20
      catboost: 0.20
      poisson: 0.25
      baseline: 0.10
    blend_method: "weighted_average"
    calibrate_final_output: true

calibration:
  enabled: true
  method: "isotonic"
  apply_to:
    - "xgboost"
    - "lightgbm"
    - "catboost"
    - "ensemble"

cross_validation:
  enabled: true
  method: "temporal_grouped"
  n_splits: 5
  group_key: "match_date"

simulation:
  n_runs_default: 20000
  use_scoreline_model: true
  use_penalty_resolution: true

model_saving:
  save_best_model: true
  save_all_models: true
  save_dir: "models/artifacts"
  snapshot_dir: "models/snapshots"
````

---

## 📋 FILE: `config/tournament.yaml`
````yaml
tournament:
  name: "FIFA World Cup 2026"
  year: 2026
  teams: 48
  groups: 12
  teams_per_group: 4
  total_matches: 104
  hosts:
    - "Canada"
    - "Mexico"
    - "United States"

advancement:
  group_stage:
    automatic_qualifiers_per_group: 2
    best_third_place_slots: 8

points:
  win: 3
  draw: 1
  loss: 0

group_tiebreakers:
  - "points"
  - "goal_difference"
  - "goals_scored"
  - "head_to_head_points"
  - "head_to_head_goal_difference"
  - "head_to_head_goals_scored"
  - "fair_play"
  - "drawing_of_lots"

knockout:
  extra_time: true
  penalties: true

outputs:
  save_group_tables: true
  save_bracket_probabilities: true
  save_matchday_predictions: true
````

---

## 🧱 ROOT FILES TO GENERATE

### FILE: `setup.py`
````python
from setuptools import find_packages, setup

setup(
    name="world-cup-2026-predictor",
    version="0.1.0",
    description="ML pipeline for predicting FIFA World Cup 2026 matches",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.26.0",
        "pandas>=2.2.0",
        "scikit-learn>=1.5.0",
        "xgboost>=2.1.0",
        "lightgbm>=4.5.0",
        "catboost>=1.2.0",
        "statsmodels>=0.14.0",
        "fastapi>=0.115.0",
        "uvicorn>=0.30.0",
        "pydantic>=2.9.0",
        "requests>=2.32.0",
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.3.0",
            "pytest-cov>=5.0.0",
            "pytest-mock>=3.14.0",
            "black>=24.0.0",
            "isort>=5.13.0",
            "pylint>=3.3.0",
            "mypy>=1.11.0",
        ]
    },
)
````

### FILE: `pyproject.toml`
````toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "world-cup-2026-predictor"
version = "0.1.0"
description = "Predict FIFA World Cup 2026 matches using ratings, ML and simulation"
requires-python = ">=3.10"

[tool.black]
line-length = 88
target-version = ['py310']

[tool.isort]
profile = "black"
line_length = 88

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --cov=src"
````

### FILE: `Makefile`
````makefile
.PHONY: help install install-dev test lint format clean pipeline train predict simulate refresh api

help:
	@echo "World Cup 2026 Predictor - Available Commands"
	@grep -E '^[a-zA-Z_-]+:' $(MAKEFILE_LIST) | sed 's/:.*//g'

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install pytest pytest-cov pytest-mock black isort pylint mypy

test:
	pytest tests/ -v --cov=src

lint:
	pylint src/ --disable=R0903

format:
	black src/ tests/ scripts/
	isort src/ tests/ scripts/

clean:
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
	find . -type d -name '.pytest_cache' -delete

pipeline:
	python scripts/run_pipeline.py

train:
	python scripts/train_models.py

predict:
	python scripts/predict_matches.py

simulate:
	python scripts/simulate_tournament.py

refresh:
	python scripts/refresh_after_matchday.py

api:
	python scripts/serve_api.py
````

### FILE: `.gitignore`
````
__pycache__/
*.py[cod]
*.so
.Python
build/
dist/
*.egg-info/
venv/
ENV/
env/
.vscode/
.idea/
*.swp
.DS_Store
.env
.env.local
.pytest_cache/
.mypy_cache/
.pylint.d/
coverage.xml
htmlcov/

# Data
/data/raw/
/data/interim/
/data/processed/
/data/external/

# Models
/models/artifacts/
/models/calibrated/
/models/snapshots/
*.pkl
*.joblib

# Logs
/logs/
*.log

# Outputs
/outputs/

# Reports
reports/academic/*.pdf
reports/dashboard/*.pdf
````

---

## 📋 FILE: `README.md`

Create a complete README in English with:
- project objective
- architecture summary
- installation steps
- environment setup
- pipeline steps
- training instructions
- prediction instructions
- tournament simulation instructions
- incremental refresh instructions
- API usage examples
- output examples
- test instructions
- Docker usage
- realistic disclaimer: no guaranteed winnings, only probabilistic forecasts

Also include example commands:
````bash
python scripts/run_pipeline.py
python scripts/train_models.py
python scripts/predict_matches.py --stage group
python scripts/simulate_tournament.py --n-runs 20000
python scripts/refresh_after_matchday.py --source official
python scripts/serve_api.py
````

---

## 📋 FILE: `ARCHITECTURE.md`

Create a detailed architecture document covering:
- why temporal splits are mandatory
- why ratings + ML + Poisson are combined
- how data flows from source APIs to prediction outputs
- online update design after tournament matches
- snapshot strategy
- calibration strategy
- simulation engine design
- API design
- testing strategy
- risk controls against data leakage

Include diagrams in markdown code blocks.

---

## 🐳 FILE: `docker/Dockerfile`
````dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p \
    data/raw \
    data/interim \
    data/processed \
    data/external \
    models/artifacts \
    models/calibrated \
    models/snapshots \
    logs \
    outputs

EXPOSE 8000

CMD ["python", "scripts/serve_api.py"]
````

### FILE: `docker/docker-compose.yml`
````yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./logs:/app/logs
      - ./outputs:/app/outputs
    environment:
      - PYTHONUNBUFFERED=1
      - ENV=production
      - LOG_LEVEL=INFO
    command: python scripts/serve_api.py
````

---

## 🔄 DATA SOURCE REQUIREMENTS

Build the project with a **provider abstraction layer**.

### Provider priority
1. **Official FIFA source** for World Cup 2026 fixtures/results if available
2. Secondary football match provider (configured by API key)
3. Local cached CSV/JSON fallback

### Required provider modules
- `src/data/fifa_client.py`
- `src/data/football_data_client.py`

### Data ingestion requirements
The code must support:
- downloading historical international fixtures/results
- downloading current World Cup 2026 schedule/results
- normalizing team names across providers
- caching raw responses
- retry logic with exponential backoff
- local fallback if APIs are unavailable

### Important reliability rule
Do NOT hardcode a single fragile endpoint deep in business logic.
Wrap all external calls in small service classes.

---

## 🧮 FEATURE REQUIREMENTS

### `src/features/team_features.py`
Generate all team-level rolling features, including:
- recent form
- recent goals for/against
- performance vs strong opponents
- major tournament experience
- World Cup experience
- penalty shootout history proxy
- manager continuity proxy
- squad continuity proxy

### `src/features/match_features.py`
Generate match-context features, including:
- host advantage
- confederation matchup type
- rest differential
- travel differential
- rating gaps
- upset index
- draw propensity

### `src/features/tournament_features.py`
Generate stage-specific features, including:
- group-stage state
- must-win pressure
- qualification pressure
- prior points and goal difference
- best-third-place pressure
- knockout tie context

### `src/features/rating_features.py`
Generate rating-derived features from rating modules.

### `src/features/rest_travel_features.py`
Compute rest days, distance approximations and timezone shifts using venue metadata.

### `src/features/build_features.py`
Orchestrate all feature generation and save processed feature matrix to disk.

---

## 📈 RATING SYSTEM REQUIREMENTS

### `src/ratings/elo.py`
Implement standard Elo with configurable K-factor and competition weights.

### `src/ratings/form_elo.py`
Implement a recent-form weighted Elo.

### `src/ratings/attack_defense.py`
Estimate team attack and defense strengths from historical results.

### `src/ratings/rating_manager.py`
Central orchestrator that:
- fits all ratings on historical data
- updates ratings with new World Cup results
- exports per-team rating snapshots

---

## 🤖 MODEL REQUIREMENTS

### `src/models/base_model.py`
Abstract interface with methods:
- `fit`
- `predict`
- `predict_proba`
- `save`
- `load`
- `get_feature_importance`

### `src/models/baseline.py`
Simple baseline using Elo gap + draw heuristics.

### `src/models/xgboost_model.py`
Multi-class classifier for home/draw/away.

### `src/models/lightgbm_model.py`
Multi-class classifier.

### `src/models/catboost_model.py`
Multi-class classifier with native categorical support where applicable.

### `src/models/poisson_model.py`
Expected-goals style model that produces:
- home goal distribution
- away goal distribution
- outcome probabilities
- likely scoreline
- BTS and over/under probabilities

### `src/models/ensemble_model.py`
Blend all model outputs and expose a unified interface.
Must support configurable weights and probability normalization.

### `src/models/model_factory.py`
Factory for all model types.

---

## 🏋️ TRAINING REQUIREMENTS

### `src/training/trainer.py`
Must:
- load processed feature matrix
- perform temporal train/validation/test split
- train all enabled models
- evaluate all models
- save artifacts
- save prediction samples
- choose best primary model

### `src/training/evaluator.py`
Compute at least:
- log_loss
- multiclass_brier_score
- accuracy
- balanced_accuracy
- top_confidence_accuracy
- calibration_error
- ranked_probability_score
- outcome_confidence_bins

### `src/training/calibrator.py`
Calibrate multi-class probabilities using configured method.

### `src/training/cross_validation.py`
Temporal CV only. Never random split.

---

## 🎲 SIMULATION REQUIREMENTS

### `src/simulation/scoreline.py`
Produce scoreline probability matrix from expected goals.

### `src/simulation/group_stage.py`
Must:
- simulate all remaining group matches
- update tables correctly
- resolve tie-breakers
- determine best third-placed teams

### `src/simulation/knockout.py`
Must:
- simulate Round of 32 through Final
- handle extra time / penalties
- use regulation probabilities + scoreline model

### `src/simulation/tournament_simulator.py`
Single high-level interface to simulate the full tournament.

Outputs:
- team advancement probabilities
- champion probabilities
- expected finishing stage
- simulated bracket frequency tables

---

## 🔮 PREDICTION REQUIREMENTS

### `src/prediction/predictor.py`
Produce predictions for:
- a single match
- a matchday
- the full remaining schedule

Each prediction row must include:
- teams
- date
- stage
- all three outcome probabilities
- expected goals
- most likely scoreline
- confidence score
- upset risk
- recommended quiniela pick

### `src/prediction/daily_refresh.py`
Must:
- detect newly completed World Cup matches
- update datasets and ratings
- regenerate all remaining match predictions
- save timestamped outputs

### `src/prediction/scenario_runner.py`
Allow custom scenario simulations, e.g. “if Team A beats Team B”.

---

## 🌐 API REQUIREMENTS

Create `src/api/main.py` using FastAPI.

Required endpoints:
- `GET /health`
- `GET /teams`
- `GET /matches/upcoming`
- `POST /predict/match`
- `POST /predict/matchday`
- `GET /predictions/latest`
- `POST /simulate/tournament`
- `POST /refresh`
- `GET /team/{team_name}`
- `GET /reports/leaderboard`

Use Pydantic request/response models.
Load model artifacts once at startup.
Handle missing data gracefully.

---

## 🧪 TEST REQUIREMENTS

Generate complete tests.

### Unit tests
- data loading
- data validation
- feature engineering
- Elo updates
- Poisson outputs sum correctly
- ensemble probabilities sum to 1
- simulator bracket logic
- API health endpoint

### Integration tests
- full pipeline run on small synthetic dataset
- refresh flow after adding a completed match
- prediction generation for remaining schedule

Provide fixtures in `tests/conftest.py`.

---

## 📊 REPORT REQUIREMENTS

### `reports/academic/main.tex`
Create a complete academic-style report covering:
- abstract
- introduction
- data sources
- methodology
- ratings
- ML models
- calibration
- tournament simulation
- validation results
- limitations
- conclusions

### `reports/dashboard/main.tex`
Create an executive-style PDF report with:
- strongest teams
- title probabilities
- most likely finalists
- top upset alerts
- safest quiniela picks
- most uncertain matches

---

## 🧰 SCRIPT REQUIREMENTS

### `scripts/run_pipeline.py`
End-to-end ETL and feature generation.

### `scripts/train_models.py`
Train all models and save artifacts.

### `scripts/predict_matches.py`
CLI for predicting matches. Support filters by stage, team, date.

### `scripts/simulate_tournament.py`
Run full tournament simulations.

### `scripts/refresh_after_matchday.py`
Ingest newly played official matches and regenerate predictions.

### `scripts/backfill_world_cup_results.py`
Allow manual backfill from CSV or JSON for robustness.

### `scripts/serve_api.py`
Launch uvicorn server.

All scripts must use argparse and logging.

---

## 📦 DATA LEAKAGE RULES (MANDATORY)

1. Never use future matches to build pre-match features.
2. Never use post-match outcomes inside predictor features for that same match.
3. Use temporal splits only.
4. When refreshing after a played World Cup match:
   - the completed match becomes available only for future fixtures
   - never refit using later matches beyond the current simulated time
5. Group table features must reflect only matches already played before each fixture.

---

## 📊 OUTPUT FILE REQUIREMENTS

Save these outputs automatically:
- `outputs/predictions/latest_predictions.csv`
- `outputs/predictions/latest_predictions.json`
- `outputs/predictions/matchday_predictions.csv`
- `outputs/simulations/tournament_probabilities.csv`
- `outputs/simulations/champion_probabilities.csv`
- `outputs/leaderboards/quiniela_safe.csv`
- `outputs/leaderboards/quiniela_aggressive.csv`
- `outputs/reports/model_evaluation_summary.json`

Also save dated snapshots after refresh:
- `outputs/predictions/YYYYMMDD_HHMM_remaining_matches.json`
- `models/snapshots/YYYYMMDD_HHMM_ratings.csv`

---

## 🎯 FINAL PROJECT OBJECTIVE

The final project should let the end user do exactly this:
1. build the historical dataset
2. train ratings and predictive models
3. predict all World Cup 2026 matches before the tournament
4. refresh predictions after each matchday as new official results come in
5. simulate the full tournament thousands of times
6. export practical quiniela suggestions and confidence rankings

---

## 📋 GENERATE ALL FILES IN THIS ORDER

### PHASE 1: CORE UTILS
1. `src/utils/logging_config.py`
2. `src/utils/config.py`
3. `src/utils/metrics.py`
4. `src/utils/plotting.py`
5. `src/utils/dates.py`
6. `src/utils/io.py`
7. `src/utils/seeds.py`
8. `src/utils/__init__.py`

### PHASE 2: DATA MODULE
1. `src/data/__init__.py`
2. `src/data/fifa_client.py`
3. `src/data/football_data_client.py`
4. `src/data/data_loader.py`
5. `src/data/historical_builder.py`
6. `src/data/tournament_updater.py`
7. `src/data/data_validator.py`
8. `src/data/data_splitter.py`

### PHASE 3: FEATURES MODULE
1. `src/features/__init__.py`
2. `src/features/build_features.py`
3. `src/features/team_features.py`
4. `src/features/match_features.py`
5. `src/features/tournament_features.py`
6. `src/features/rating_features.py`
7. `src/features/rest_travel_features.py`

### PHASE 4: RATINGS MODULE
1. `src/ratings/__init__.py`
2. `src/ratings/elo.py`
3. `src/ratings/form_elo.py`
4. `src/ratings/attack_defense.py`
5. `src/ratings/rating_manager.py`

### PHASE 5: MODELS MODULE
1. `src/models/__init__.py`
2. `src/models/base_model.py`
3. `src/models/baseline.py`
4. `src/models/xgboost_model.py`
5. `src/models/lightgbm_model.py`
6. `src/models/catboost_model.py`
7. `src/models/poisson_model.py`
8. `src/models/ensemble_model.py`
9. `src/models/model_factory.py`

### PHASE 6: TRAINING MODULE
1. `src/training/__init__.py`
2. `src/training/trainer.py`
3. `src/training/evaluator.py`
4. `src/training/calibrator.py`
5. `src/training/cross_validation.py`

### PHASE 7: SIMULATION MODULE
1. `src/simulation/__init__.py`
2. `src/simulation/scoreline.py`
3. `src/simulation/group_stage.py`
4. `src/simulation/knockout.py`
5. `src/simulation/tournament_simulator.py`

### PHASE 8: PREDICTION & API
1. `src/prediction/__init__.py`
2. `src/prediction/predictor.py`
3. `src/prediction/daily_refresh.py`
4. `src/prediction/scenario_runner.py`
5. `src/api/__init__.py`
6. `src/api/main.py`

### PHASE 9: SCRIPTS
1. `scripts/run_pipeline.py`
2. `scripts/train_models.py`
3. `scripts/predict_matches.py`
4. `scripts/simulate_tournament.py`
5. `scripts/refresh_after_matchday.py`
6. `scripts/backfill_world_cup_results.py`
7. `scripts/serve_api.py`

### PHASE 10: TESTS
1. `tests/__init__.py`
2. `tests/conftest.py`
3. `tests/unit/__init__.py`
4. `tests/unit/test_data_loader.py`
5. `tests/unit/test_features.py`
6. `tests/unit/test_ratings.py`
7. `tests/unit/test_models.py`
8. `tests/unit/test_simulation.py`
9. `tests/unit/test_api.py`
10. `tests/integration/__init__.py`
11. `tests/integration/test_pipeline.py`
12. `tests/integration/test_daily_refresh.py`

### PHASE 11: REPORTS
1. `reports/academic/main.tex`
2. `reports/dashboard/main.tex`

### PHASE 12: ROOT FILES
1. `.env`
2. `.gitignore`
3. `README.md`
4. `ARCHITECTURE.md`
5. `Makefile`
6. `requirements.txt`
7. `setup.py`
8. `pyproject.toml`
9. all config files

---

## ⚠️ CRITICAL NOTES

**DO NOT:**
- Ask for clarification
- Create incomplete files
- Truncate code
- Skip files
- Use notebooks instead of Python modules
- Ignore refresh/update requirements

**DO:**
- Create EVERY file listed
- Make files complete and executable
- Use type hints everywhere
- Add detailed Google-style docstrings
- Add logging in all important operations
- Add robust error handling
- Make the project runnable end-to-end
- Ensure probability outputs are calibrated and normalized
- Ensure all remaining tournament predictions can be regenerated after new results

---

## ✅ FINAL VALIDATION CHECKLIST

When finished, the project must:
- Have no truncated files
- Have all imports working
- Have type hints on 100% of functions
- Have docstrings on 100% of classes/methods/functions
- Have logging in all important operations
- Have error handling in all external I/O and training steps
- Support full training and prediction pipeline
- Support tournament simulation
- Support refresh after played World Cup matches
- Save artifacts, outputs and snapshots
- Be runnable with:
  - `python scripts/run_pipeline.py`
  - `python scripts/train_models.py`
  - `python scripts/predict_matches.py`
  - `python scripts/simulate_tournament.py`
  - `python scripts/refresh_after_matchday.py`

---

## 🚨 FINAL INSTRUCTION TO CLAUDE CODE

GENERATE ALL FILES NOW. NO QUESTIONS. COMPLETE PROJECT GENERATION.
