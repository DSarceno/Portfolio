# F1 RACE PREDICTOR - PROJECT GENERATION GUIDE FOR CLAUDE CODE

## ⚠️ CRITICAL INSTRUCTIONS FOR CLAUDE CODE

Read this ENTIRE file first. This is the ONLY context you need.

You will:
1. Create ALL files in the correct directory structure
2. Generate COMPLETE Python code (no truncation)
3. Generate config files (YAML)
4. Generate LaTeX reports
5. Generate requirements.txt and other support files
6. NOT ask questions - generate everything based on this specification

When you finish, the project should be 100% functional.

---

## 🎯 PROJECT OVERVIEW

**Project**: F1 Race Predictor
**Goal**: Predict Formula 1 race positions using ML (XGBoost, PyTorch, scikit-learn)
**Data**: FastF1 API (2020-2025)
**Output**: Position predictions, lap-by-lap analysis, academic + dashboard reports

---

## 📁 DIRECTORY STRUCTURE TO CREATE
````
f1-race-predictor/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── fastf1_client.py
│   │   ├── data_loader.py
│   │   ├── data_validator.py
│   │   └── data_splitter.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── build_features.py
│   │   ├── lap_features.py
│   │   ├── driver_features.py
│   │   ├── constructor_features.py
│   │   └── weather_features.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_model.py
│   │   ├── baseline.py
│   │   ├── xgboost_model.py
│   │   ├── neural_net.py
│   │   └── model_factory.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── evaluator.py
│   │   └── cross_validation.py
│   ├── prediction/
│   │   ├── __init__.py
│   │   └── predictor.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py
│   └── utils/
│       ├── __init__.py
│       ├── logging_config.py
│       ├── config.py
│       ├── metrics.py
│       └── plotting.py
├── tests/
│   ├── __init__.py
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_data_loader.py
│   │   ├── test_features.py
│   │   └── test_models.py
│   └── conftest.py
├── scripts/
│   ├── run_pipeline.py
│   ├── train_model.py
│   └── predict_race.py
├── config/
│   ├── config.yaml
│   ├── features.yaml
│   ├── model_params.yaml
│   └── .env.example
├── data/
│   ├── raw/
│   │   └── fastf1_cache/
│   ├── interim/
│   └── processed/
├── logs/
│   ├── pipeline/
│   ├── training/
│   ├── prediction/
│   └── errors/
├── reports/
│   ├── academic/
│   │   ├── main.tex
│   │   ├── sections/
│   │   │   ├── 01_abstract.tex
│   │   │   ├── 02_introduction.tex
│   │   │   ├── 03_methodology.tex
│   │   │   ├── 04_results.tex
│   │   │   └── 05_conclusions.tex
│   │   └── figures/
│   └── dashboard/
│       ├── main.tex
│       └── sections/
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── .env
├── .gitignore
├── .env.example
├── README.md
├── ARCHITECTURE.md
├── Makefile
├── requirements.txt
├── setup.py
├── pyproject.toml
└── outputs/
````

---

## 🔧 CODE STANDARDS (FOLLOW THESE EXACTLY)

### Python Code Requirements
- **Type Hints**: 100% in ALL functions
- **Docstrings**: Google style, detailed for all functions and classes
- **Logging**: Use logging module, NOT print()
- **Error Handling**: Try-catch with specific exceptions
- **Code Style**: Black formatter (88 char line length)
- **Imports**: Group as stdlib > third-party > local
- **Max Function Length**: 50 lines
- **Max Class**: No limit, but keep methods focused

### Example Structure:
````python
"""Module description."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class MyClass:
    """Class description.
    
    Detailed explanation if needed.
    """
    
    def __init__(self, param: str) -> None:
        """Initialize.
        
        Args:
            param: Description.
        """
        self.param = param
        logger.info(f"MyClass initialized with {param}")
    
    def my_method(self, data: pd.DataFrame) -> Dict[str, float]:
        """Process data.
        
        Args:
            data: Input dataframe.
            
        Returns:
            Dictionary with results.
            
        Raises:
            ValueError: If data is empty.
        """
        if data.empty:
            raise ValueError("Data cannot be empty")
        
        try:
            result = self._process(data)
            logger.info(f"Processed {len(data)} rows")
            return result
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            raise
    
    def _process(self, data: pd.DataFrame) -> Dict[str, float]:
        """Internal processing."""
        pass
````

---

## 📦 DEPENDENCIES

Create `requirements.txt`:
````
# Core
numpy==1.24.3
pandas==2.0.3
scipy==1.11.1

# ML
scikit-learn==1.3.0
xgboost==2.0.0
torch==2.0.1
tensorflow==2.13.0

# Data
fastf1==0.5.0

# API
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.4.2

# Utils
python-dotenv==1.0.0
pyyaml==6.0.1
tqdm==4.66.1

# Visualization
matplotlib==3.8.0
seaborn==0.13.0

# Testing
pytest==7.4.2
pytest-cov==4.1.0
pytest-mock==3.12.0

# Code Quality
black==23.10.0
isort==5.12.0
pylint==3.0.2
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
FASTF1_CACHE_DIR=./data/raw/fastf1_cache
FASTF1_CACHE_ENABLED=true
MAX_SEASONS=5
START_SEASON=2020
END_SEASON=2025

# Model
MODEL_DIR=./models
DEFAULT_MODEL=xgboost
TEST_SIZE=0.2
CV_FOLDS=5

# API
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
API_RELOAD=true

# Paths
CONFIG_PATH=./config/config.yaml
FEATURES_CONFIG_PATH=./config/features.yaml
MODEL_PARAMS_PATH=./config/model_params.yaml
REPORTS_DIR=./reports

# Optional
ANTHROPIC_API_KEY=
````

---

## 📋 FILE: `.env.example` (in config/)
````bash
# Environment variables template
# Copy to .env and fill in your values

ANTHROPIC_API_KEY=sk-ant-v1-XXXXX
FASTF1_CACHE_DIR=./data/raw/fastf1_cache
FASTF1_CACHE_ENABLED=true

DB_HOST=localhost
DB_PORT=5432
DB_NAME=f1_predictor
DB_USER=postgres
DB_PASSWORD=password

LOG_LEVEL=INFO
LOG_FILE=logs/f1_predictor.log

API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
API_DEBUG=false

RANDOM_SEED=42
ENVIRONMENT=development
VERBOSE=true
````

---

## 📋 FILE: `config/config.yaml`
````yaml
# =============================================================================
# F1 Race Predictor - Main Configuration
# =============================================================================

project:
  name: "F1 Race Predictor"
  version: "0.1.0"
  description: "Predict Formula 1 race positions using ML"

data:
  start_season: 2020
  end_season: 2025
  max_seasons: 5
  cache_dir: "data/raw/fastf1_cache"
  raw_dir: "data/raw"
  interim_dir: "data/interim"
  processed_dir: "data/processed"

  session_types:
    - "R"    # Race
    - "Q"    # Qualifying
    - "FP1"  # Free Practice 1
    - "FP2"  # Free Practice 2
    - "FP3"  # Free Practice 3

  validation:
    max_null_percentage: 0.3
    min_laps_per_race: 10
    min_drivers_per_race: 10

training:
  test_size: 0.2
  val_size: 0.15
  cv_folds: 5
  random_seed: 42
  shuffle: true
  split_strategy: "temporal"

evaluation:
  metrics:
    - "mae"
    - "rmse"
    - "top_3_accuracy"
    - "top_5_accuracy"
    - "spearman_corr"
    - "ndcg"
    - "exact_accuracy"

  dnf_metrics:
    - "accuracy"
    - "precision"
    - "recall"
    - "f1_score"

logging:
  level: "INFO"
  format: "%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s"
  date_format: "%Y-%m-%d %H:%M:%S"
  file: "logs/f1_predictor.log"
  max_bytes: 10485760
  backup_count: 5

paths:
  logs_pipeline: "logs/pipeline"
  logs_training: "logs/training"
  logs_prediction: "logs/prediction"
  logs_errors: "logs/errors"
  models_dir: "models"
  outputs_dir: "outputs"
  reports_dir: "reports"
````

---

## 📋 FILE: `config/features.yaml`
````yaml
# =============================================================================
# Feature Engineering Configuration
# =============================================================================

feature_groups:
  
  lap_features:
    enabled: true
    features:
      - avg_lap_time_last_5_laps
      - median_lap_time
      - best_lap_time
      - lap_time_std
      - pace_trend
      - consistency_score
      - max_speed
      - avg_speed
      - throttle_usage
      - brake_usage
      - cornering_speed
      - straight_speed
      - fuel_consumption_rate
      - tire_degradation
      - drs_usage_count

  driver_features:
    enabled: true
    features:
      - driver_experience_years
      - total_races
      - avg_finish_position_all_time
      - win_rate
      - podium_rate
      - points_per_race
      - dnf_rate
      - avg_finish_position_5races
      - points_per_race_5races
      - avg_grid_to_finish_delta_5races
      - team_car_mate_comparison
      - team_avg_points
      - best_circuit
      - best_circuit_avg_points
      - pole_position_count
      - team_change_recent
      - seasons_with_current_team
      - is_rookie
      - avg_qualify_position
      - avg_race_position

  constructor_features:
    enabled: true
    features:
      - constructor_avg_points
      - constructor_dnf_rate
      - constructor_avg_finish_pos
      - constructor_points_5races
      - constructor_win_count
      - constructor_recent_form
      - engine_max_speed_rank
      - reliability_score
      - new_regulations_season
      - mid_season_upgrades

  weather_features:
    enabled: true
    critical: true
    features:
      - track_temp_celsius
      - air_temp_celsius
      - humidity_percent
      - wind_speed_kmh
      - is_raining
      - track_status
      - temp_diff
      - weather_aggressiveness

  circuit_features:
    enabled: true
    features:
      - circuit_type
      - elevation
      - lap_length
      - number_of_turns
      - avg_speed_previous_years

selection:
  method: "variance_threshold"
  variance_threshold: 0.01
  correlation_threshold: 0.95
  max_features_to_keep: 30
  min_features_to_keep: 15

missing_values:
  strategy: "variable"
  
  defaults:
    lap_features: "mean"
    driver_features: "team_mean"
    weather_features: "forward_fill"
    circuit_features: "drop"

scaling:
  method: "standard"
  apply_to_all: true

engineering:
  create_interactions: false
  create_polynomials: false
  create_lag_features: true
  lag_periods: [1, 3, 5]
````

---

## 📋 FILE: `config/model_params.yaml`
````yaml
# =============================================================================
# Model Parameters Configuration
# =============================================================================

models:
  
  xgboost:
    enabled: true
    primary: true
    
    hyperparameters:
      objective: "multi:softprob"
      num_class: 20
      max_depth: 6
      min_child_weight: 1
      gamma: 0
      alpha: 0
      lambda: 1
      learning_rate: 0.1
      n_estimators: 200
      subsample: 0.8
      colsample_bytree: 0.8
      colsample_bylevel: 0.8
      tree_method: "auto"
      random_state: 42
      n_jobs: -1
    
    training:
      early_stopping_rounds: 50
      eval_metric: "mlogloss"
      verbose: 100
      batch_size: 256
  
  neural_net:
    enabled: true
    primary: false
    
    architecture:
      input_features: null
      hidden_layers: [256, 128, 64]
      activation: "relu"
      output_classes: 20
      dropout_rate: 0.3
      batch_norm: true
      
    hyperparameters:
      learning_rate: 0.001
      optimizer: "adam"
      loss: "cross_entropy"
      weight_decay: 0.0001
      
    training:
      epochs: 100
      batch_size: 32
      early_stopping: 20
      early_stopping_metric: "val_loss"
      validation_split: 0.2
      shuffle: true
      random_state: 42
      device: "cpu"
  
  baseline:
    enabled: true
    primary: false
    
    strategy: "grid_position"
    
    parameters:
      use_historical_delta: true
      dnf_threshold: 0.2
      max_position_change: 5

cross_validation:
  enabled: true
  method: "temporal"
  n_splits: 5

model_saving:
  save_best_model: true
  save_all_models: false
  save_frequency: "best"
  save_dir: "models"

metrics_thresholds:
  min_accuracy: 0.5
  min_top3_accuracy: 0.8
  max_mae: 5.0
  min_dnf_f1: 0.6
````

---

## 📋 FILE: `setup.py`
````python
from setuptools import setup, find_packages

setup(
    name="f1-race-predictor",
    version="0.1.0",
    description="ML pipeline for predicting F1 race outcomes",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/f1-race-predictor",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "fastf1>=0.5.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "torch>=2.0.0",
        "tensorflow>=2.13.0",
        "xgboost>=2.0.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.12.0",
            "black>=23.0.0",
            "isort>=5.13.0",
            "pylint>=3.0.0",
        ]
    }
)
````

---

## 📋 FILE: `pyproject.toml`
````toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "f1-race-predictor"
version = "0.1.0"
description = "ML pipeline for predicting F1 race outcomes"
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

---

## 📋 FILE: `Makefile`
````makefile
.PHONY: help install install-dev test lint format clean run-pipeline train predict api

help:
	@echo "F1 Race Predictor - Available Commands"
	@echo "======================================"
	@grep -E '^[a-zA-Z_-]+:' $(MAKEFILE_LIST) | sed 's/:.*//g'

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install pytest pytest-cov black isort pylint

test:
	pytest tests/ -v --cov=src

lint:
	pylint src/ --disable=R0903

format:
	black src/ tests/
	isort src/ tests/

clean:
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
	find . -type d -name '.pytest_cache' -delete

run-pipeline:
	python scripts/run_pipeline.py

train:
	python scripts/train_model.py

predict:
	python scripts/predict_race.py

api:
	uvicorn src.api.main:app --reload

docker-build:
	docker build -t f1-predictor .

docker-run:
	docker-compose up

all: format lint test
````

---

## 📋 FILE: `.gitignore`
````
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Data
data/raw/
data/interim/
data/processed/
data/raw/fastf1_cache/

# Models
models/
*.pkl
*.pt
*.pth

# Logs
logs/
*.log

# Environment
.env
.env.local

# Reports
reports/academic/*.pdf
reports/dashboard/*.pdf

# Cache
.pytest_cache/
.mypy_cache/
.pylint.d/

# OS
Thumbs.db
.DS_Store

# Outputs
outputs/
````

---

## 📋 FILE: `README.md`
````markdown
# F1 Race Predictor 🏎️

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)]()

**Predict Formula 1 race outcomes using machine learning and historical data.**

## 🎯 Objective

Predict final race positions for each F1 driver using:
- Historical race data (2020-2025)
- Driver & constructor performance metrics
- Weather & circuit conditions
- Multiple ML models (XGBoost, Neural Networks, Baseline)

## 📊 Expected Performance

| Model | Accuracy | Top-3 Accuracy | Top-5 Accuracy |
|-------|----------|----------------|----------------|
| **XGBoost** | 72-75% | 90-95% | 95-98% |
| **Neural Net** | 65-70% | 88-92% | 92-96% |
| **Baseline** | 45-50% | 75-80% | 85-90% |

## 🚀 Quick Start

### 1. Installation
```bash
# Clone or create project
mkdir f1-race-predictor
cd f1-race-predictor

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration
```bash
# Copy environment file
cp config/.env.example .env

# Edit .env with your settings
```

### 3. Run Pipeline
```bash
# Extract and prepare data
python scripts/run_pipeline.py

# Train models
python scripts/train_model.py

# Make predictions
python scripts/predict_race.py --season 2025 --round 1
```

### 4. Serve API
```bash
# Start API server
python -m uvicorn src.api.main:app --reload

# API docs: http://localhost:8000/docs
```

## 📁 Project Structure

- `src/data/` - Data extraction and loading
- `src/features/` - Feature engineering
- `src/models/` - ML models (XGBoost, PyTorch, Baseline)
- `src/training/` - Training and evaluation
- `src/prediction/` - Race predictions
- `src/api/` - FastAPI server
- `src/utils/` - Utilities and helpers
- `tests/` - Unit tests
- `scripts/` - Pipeline and training scripts
- `config/` - Configuration files (YAML)
- `reports/` - LaTeX reports (academic + dashboard)
- `docker/` - Docker configuration

## 🧪 Testing
```bash
# Run all tests
pytest tests/ -v --cov=src

# Run specific test
pytest tests/unit/test_data_loader.py -v
```

## 📋 Code Quality
```bash
# Format code
make format

# Lint code
make lint

# All checks
make all
```

## 📚 Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) - Design decisions
- [reports/academic/main.pdf](reports/academic/main.pdf) - Full research paper
- [reports/dashboard/main.pdf](reports/dashboard/main.pdf) - Executive summary

## 🐳 Docker
```bash
# Build image
docker build -t f1-predictor .

# Run container
docker-compose up

# API available at: http://localhost:8000
```

## 📦 Key Dependencies

- **Data**: FastF1, pandas, numpy
- **ML**: scikit-learn, XGBoost, PyTorch, TensorFlow
- **API**: FastAPI, Pydantic, Uvicorn
- **Testing**: pytest, pytest-cov
- **Code Quality**: Black, isort, pylint

## 📝 License

MIT License - see LICENSE file for details

## 🤝 Contributing

1. Create feature branch (`git checkout -b feature/AmazingFeature`)
2. Commit changes (`git commit -m 'Add AmazingFeature'`)
3. Push to branch (`git push origin feature/AmazingFeature`)
4. Open Pull Request

## 📞 Contact

Your Name - [Your Email](mailto:your.email@example.com)

---

**¿Quieres ver the full research paper or need help setting up?**
````

---

## 📋 FILE: `ARCHITECTURE.md`
````markdown
# Architecture Documentation

## Design Decisions

### 1. Temporal Data Split
- NOT random split (would leak future information)
- Train: 2020-2023, Validation: 2024, Test: 2025
- Respects realistic prediction scenario

### 2. Feature Engineering Strategy
- Generate 50+ features initially
- Feature selection reduces to 20-30 best
- Features split by category: lap, driver, constructor, weather
- Weather ALWAYS included (critical for F1)

### 3. Multiple Models
- **XGBoost**: Best with tabular data, interpretable
- **Neural Network**: Potential for complex patterns
- **Baseline**: Lower bound, simple grid position predictor
- Comparison: Validates results, shows model capability

### 4. Logging Strategy
- Logging module (not MLflow, not print)
- Centralized logging config
- Per-module loggers
- File + console output

### 5. Data Pipeline
- Extract → Validate → Transform → Feature Engineering → Split
- Each step documented and testable
- Caching to avoid re-downloading

### 6. API Design
- REST endpoints (not WebSocket)
- Request validation with Pydantic
- Stateless service (models loaded at startup)
- Docker-ready

## Class Hierarchy
````
BaseModel (ABC)
├── XGBoostModel
├── NeuralNetModel
└── BaselineModel

DataPipelineStep (ABC)
├── ValidationStep
├── FeatureEngineeringStep
└── ...

FeatureExtractor (ABC)
├── LapFeaturesExtractor
├── DriverFeaturesExtractor
├── ConstructorFeaturesExtractor
└── WeatherFeaturesExtractor
````

## Data Flow
````
FastF1 API
    ↓
data/raw/ (cached)
    ↓
FastF1Client.get_session()
    ↓
DataLoader.load_race()
    ↓
DataValidator.validate()
    ↓
FeatureBuilder.build_features()
    ↓
DataSplitter.temporal_split()
    ↓
data/processed/
    ↓
Trainer.train_single_model()
    ↓
models/best_model.pkl
    ↓
Predictor.predict_race()
    ↓
outputs/predictions.json
````

## Key Patterns

1. **Factory Pattern**: ModelFactory for creating model instances
2. **Strategy Pattern**: Different feature extractors
3. **Pipeline Pattern**: Sequential data processing
4. **Singleton**: Config loaded once
5. **Dependency Injection**: Services receive dependencies

## Error Handling

- Specific exception types
- Logging all errors
- Graceful degradation
- User-friendly messages

## Performance Considerations

- Caching FastF1 data locally
- Vectorized operations (numpy/pandas)
- Batch processing for models
- Early stopping for neural networks

## Testing Strategy

- Unit tests for each module
- Fixtures for test data
- Mocking external APIs
- Coverage target: 80%+

---

**For questions, see src/models/base_model.py for pattern examples.**
````

---

## 🐳 FILE: `docker/Dockerfile`
````dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Create directories
RUN mkdir -p data/raw data/interim data/processed logs models outputs

# Expose API port
EXPOSE 8000

# Run API
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
````

---

## 🐳 FILE: `docker-compose.yml`
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
    command: uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
````

---

## 🔄 NOW GENERATE ALL PYTHON FILES

### KEY REQUIREMENTS FOR ALL PYTHON FILES:

1. **Type Hints**: 100% coverage
````python
   def function(param: str) -> Dict[str, float]:
````

2. **Docstrings**: Google style
````python
   """Short description.
   
   Longer description if needed.
   
   Args:
       param: Description.
       
   Returns:
       Description of return value.
       
   Raises:
       ValueError: When this happens.
   """
````

3. **Logging**: Every important operation
````python
   logger = logging.getLogger(__name__)
   logger.info("Starting process")
   logger.error("Error occurred", exc_info=True)
````

4. **Error Handling**: Specific exceptions
````python
   try:
       result = operation()
   except SpecificError as e:
       logger.error(f"Failed: {e}")
       raise
````

5. **Line Length**: Max 88 characters (Black)

6. **Imports**: stdlib > third-party > local
````python
   import os
   import sys
   from pathlib import Path
   
   import pandas as pd
   import numpy as np
   
   from src.utils.config import Config
````

---

## 📊 GENERATE ALL FILES IN THIS ORDER

### PHASE 1: CORE UTILS (3 files)
1. `src/utils/logging_config.py` - Logging setup
2. `src/utils/config.py` - Configuration manager
3. `src/utils/__init__.py` - Empty

### PHASE 2: DATA MODULE (5 files)
1. `src/data/__init__.py` - Empty
2. `src/data/fastf1_client.py` - FastF1 API client
3. `src/data/data_loader.py` - Data loading
4. `src/data/data_validator.py` - Validation
5. `src/data/data_splitter.py` - Train/val/test split

### PHASE 3: FEATURES MODULE (6 files)
1. `src/features/__init__.py` - Empty
2. `src/features/build_features.py` - Feature orchestrator
3. `src/features/lap_features.py` - Lap features (15 features)
4. `src/features/driver_features.py` - Driver features (20 features)
5. `src/features/constructor_features.py` - Constructor features (10 features)
6. `src/features/weather_features.py` - Weather features (8 features)

### PHASE 4: MODELS MODULE (6 files)
1. `src/models/__init__.py` - Empty
2. `src/models/base_model.py` - Abstract base class
3. `src/models/baseline.py` - Baseline model
4. `src/models/xgboost_model.py` - XGBoost implementation
5. `src/models/neural_net.py` - PyTorch implementation
6. `src/models/model_factory.py` - Factory pattern

### PHASE 5: TRAINING MODULE (4 files)
1. `src/training/__init__.py` - Empty
2. `src/training/trainer.py` - Training orchestrator
3. `src/training/evaluator.py` - Evaluation metrics
4. `src/training/cross_validation.py` - Temporal K-fold

### PHASE 6: PREDICTION & API (5 files)
1. `src/prediction/__init__.py` - Empty
2. `src/prediction/predictor.py` - Race predictor
3. `src/utils/metrics.py` - Custom metrics
4. `src/utils/plotting.py` - Visualization
5. `src/api/__init__.py` - Empty
6. `src/api/main.py` - FastAPI application

### PHASE 7: SCRIPTS (3 files)
1. `scripts/run_pipeline.py` - ETL orchestrator
2. `scripts/train_model.py` - Training script
3. `scripts/predict_race.py` - Prediction script

### PHASE 8: TESTS (4 files)
1. `tests/__init__.py` - Empty
2. `tests/conftest.py` - Pytest configuration
3. `tests/unit/__init__.py` - Empty
4. `tests/unit/test_data_loader.py` - Data tests
5. `tests/unit/test_features.py` - Feature tests
6. `tests/unit/test_models.py` - Model tests

### PHASE 9: REPORTS (LaTeX)
1. `reports/academic/main.tex` - Full research paper
2. `reports/dashboard/main.tex` - Executive summary

### PHASE 10: ROOT FILES (Already provided above)
1. `.env` - Environment variables
2. `.gitignore`
3. `README.md`
4. `ARCHITECTURE.md`
5. `Makefile`
6. `requirements.txt`
7. `setup.py`
8. `pyproject.toml`
9. All config files (config/*.yaml)

---

## ⚠️ CRITICAL NOTES

**DO NOT:**
- Ask for clarification
- Create incomplete files
- Truncate code
- Skip files

**DO:**
- Create EVERY file listed
- Make files 100% complete
- Include all methods/functions mentioned
- Follow type hint requirements
- Add logging to all operations
- Include comprehensive docstrings
- Handle errors properly

**WHEN YOU FINISH:**
- All 30+ Python files created ✅
- All config files created ✅
- All test files created ✅
- All script files created ✅
- LaTeX reports created ✅
- requirements.txt created ✅
- setup.py created ✅
- Makefile created ✅
- Project is 100% ready to use ✅

---

## 🎯 FINAL VALIDATION

When you finish, the project should:
- ✅ Have no truncated files
- ✅ Have all imports working
- ✅ Have type hints on 100% of functions
- ✅ Have docstrings on 100% of classes/methods
- ✅ Have logging in all operations
- ✅ Have error handling in all functions
- ✅ Have requirements.txt updated
- ✅ Have all directories created
- ✅ Be ready to run: `python scripts/run_pipeline.py`

---

**GENERATE ALL FILES NOW. NO QUESTIONS. COMPLETE PROJECT GENERATION.**