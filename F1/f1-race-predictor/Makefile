.PHONY: help install dev-install format lint type-check test test-cov clean data train predict api docker-build docker-up docker-down all-checks

# =============================================================================
# F1 Race Predictor - Makefile
# =============================================================================

PYTHON := python
PIP := pip
SRC := src
TESTS := tests
BLACK := black
ISORT := isort
PYLINT := pylint
MYPY := mypy
PYTEST := pytest

# Colors for terminal output
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m  # No Color

help: ## Show this help message
	@echo "$(GREEN)F1 Race Predictor - Available Commands$(NC)"
	@echo "========================================"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "$(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'

# --- Setup ---

install: ## Install production dependencies
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	@echo "$(GREEN)✓ Production dependencies installed$(NC)"

dev-install: ## Install all dependencies (including dev)
	$(PIP) install -r requirements.txt
	$(PIP) install -e ".[dev]"
	pre-commit install || true
	@echo "$(GREEN)✓ Development dependencies installed$(NC)"

setup-dirs: ## Create project directory structure
	mkdir -p data/raw data/interim data/processed
	mkdir -p data/raw/fastf1_cache
	mkdir -p models
	mkdir -p reports/academic/sections reports/dashboard/sections
	mkdir -p logs
	@echo "$(GREEN)✓ Directory structure created$(NC)"

setup: dev-install setup-dirs ## Full project setup
	cp -n .env.example .env 2>/dev/null || true
	@echo "$(GREEN)✓ Project setup complete$(NC)"

# --- Code Quality ---

format: ## Format code with Black and isort
	$(BLACK) $(SRC)/ $(TESTS)/
	$(ISORT) $(SRC)/ $(TESTS)/
	@echo "$(GREEN)✓ Code formatted$(NC)"

lint: ## Run pylint
	$(PYLINT) $(SRC)/ --output-format=colorized
	@echo "$(GREEN)✓ Linting complete$(NC)"

type-check: ## Run mypy type checking
	$(MYPY) $(SRC)/
	@echo "$(GREEN)✓ Type checking complete$(NC)"

all-checks: format lint type-check ## Run all code quality checks
	@echo "$(GREEN)✓ All checks passed$(NC)"

# --- Testing ---

test: ## Run tests
	$(PYTEST) $(TESTS)/ -v

test-cov: ## Run tests with coverage report
	$(PYTEST) $(TESTS)/ -v --cov=$(SRC) --cov-report=term-missing --cov-report=html
	@echo "$(GREEN)✓ Coverage report generated in htmlcov/$(NC)"

test-unit: ## Run unit tests only
	$(PYTEST) $(TESTS)/unit/ -v

test-integration: ## Run integration tests only
	$(PYTEST) $(TESTS)/integration/ -v -m integration

# --- Pipeline ---

data: ## Run data extraction pipeline
	$(PYTHON) scripts/run_pipeline.py
	@echo "$(GREEN)✓ Data pipeline complete$(NC)"

train: ## Train models
	$(PYTHON) scripts/train_model.py
	@echo "$(GREEN)✓ Model training complete$(NC)"

predict: ## Generate predictions
	$(PYTHON) scripts/predict_race.py
	@echo "$(GREEN)✓ Predictions generated$(NC)"

pipeline: data train predict ## Run full pipeline (data → train → predict)
	@echo "$(GREEN)✓ Full pipeline complete$(NC)"

# --- API ---

api: ## Start FastAPI server
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# --- Docker ---

docker-build: ## Build Docker image
	docker-compose -f docker/docker-compose.yml build
	@echo "$(GREEN)✓ Docker image built$(NC)"

docker-up: ## Start Docker containers
	docker-compose -f docker/docker-compose.yml up -d
	@echo "$(GREEN)✓ Containers started$(NC)"

docker-down: ## Stop Docker containers
	docker-compose -f docker/docker-compose.yml down
	@echo "$(GREEN)✓ Containers stopped$(NC)"

# --- Cleanup ---

clean: ## Remove build artifacts and caches
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info htmlcov/ .coverage
	@echo "$(GREEN)✓ Cleaned build artifacts$(NC)"

clean-data: ## Remove all cached and processed data
	rm -rf data/raw/fastf1_cache/*
	rm -rf data/interim/*
	rm -rf data/processed/*
	@echo "$(YELLOW)⚠ Data directories cleared$(NC)"

clean-all: clean clean-data ## Remove everything (artifacts + data)
	@echo "$(RED)✓ Full cleanup complete$(NC)"