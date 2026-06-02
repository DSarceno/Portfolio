.PHONY: help install install-dev clean lint format test test-cov bootstrap train predict simulate update api docker-build docker-up

PY ?= python

help:
	@echo "FIFA World Cup 2026 Quiniela Predictor V2"
	@echo ""
	@echo "Targets:"
	@echo "  install       Install runtime dependencies"
	@echo "  install-dev   Install runtime + development dependencies"
	@echo "  clean         Remove build artifacts and caches"
	@echo "  lint          Run pylint on source"
	@echo "  format        Run black + isort"
	@echo "  test          Run pytest"
	@echo "  test-cov      Run pytest with coverage"
	@echo "  bootstrap     Bootstrap historical data"
	@echo "  ratings       Build pre-tournament ratings"
	@echo "  pipeline      Run full ETL + feature pipeline"
	@echo "  train         Train models"
	@echo "  predict       Predict group-stage matches"
	@echo "  simulate      Simulate the tournament"
	@echo "  update        Update after a matchday"
	@echo "  picks         Export quiniela sheet"
	@echo "  api           Run the FastAPI server"
	@echo "  docker-build  Build the Docker image"
	@echo "  docker-up     Run docker-compose up"

install:
	$(PY) -m pip install -r requirements.txt

install-dev: install
	$(PY) -m pip install -e .

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .coverage htmlcov
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

lint:
	pylint src tests

format:
	black src tests scripts
	isort src tests scripts

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

bootstrap:
	$(PY) scripts/bootstrap_historical_data.py

ratings:
	$(PY) scripts/build_ratings.py

pipeline:
	$(PY) scripts/run_pipeline.py

train:
	$(PY) scripts/train_models.py

predict:
	$(PY) scripts/predict_group_stage.py

simulate:
	$(PY) scripts/simulate_tournament.py

update:
	$(PY) scripts/update_after_matchday.py

picks:
	$(PY) scripts/export_quiniela_sheet.py

api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -f docker/Dockerfile -t wc2026-quiniela:latest .

docker-up:
	docker-compose -f docker/docker-compose.yml up --build
