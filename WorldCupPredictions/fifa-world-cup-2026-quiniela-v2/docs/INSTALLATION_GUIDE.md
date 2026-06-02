# Installation Guide

## 1. Requirements

- Python 3.10 or 3.11.
- Git.
- 4+ GB of RAM (XGBoost training and Monte-Carlo simulations).
- Optional: Docker 24+ and docker-compose v2 for the containerised deployment.

## 2. Clone

```bash
git clone <repo-url>
cd fifa-world-cup-2026-quiniela-v2
```

## 3. Virtual environment

```bash
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
.venv\Scripts\activate             # Windows
```

## 4. Install Python dependencies

```bash
pip install -r requirements.txt
```

Optional: install the project in editable mode to make `import src.*` work everywhere:

```bash
pip install -e .
```

## 5. Configure environment variables

Copy the example file and edit values as needed:

```bash
cp config/.env.example .env
```

Variables you may want to set:

- `FOOTBALL_DATA_API_KEY` — needed only if you want live ingestion from `football-data.org`.
- `LOG_LEVEL` — `DEBUG` / `INFO` / `WARNING`.
- `SIMULATION_RUNS` — override the default 10,000 Monte-Carlo runs.

## 6. Bootstrap data

```bash
python scripts/bootstrap_historical_data.py
```

This will create timestamped raw snapshots under `data/raw/`. If a source is unavailable a CSV template is produced for you to fill manually.

## 7. Verify the install

```bash
pytest -q
```

You should see green tests across the rating, feature, model, simulation and strategy suites.

## 8. Optional: Docker

```bash
make docker-build
make docker-up
```

The API is exposed at `http://localhost:8000`.

## 9. Common installation issues

| Symptom                                       | Fix                                                         |
| --------------------------------------------- | ----------------------------------------------------------- |
| `ImportError: libgomp.so.1`                   | Install `libgomp1` system package (Debian/Ubuntu)           |
| XGBoost fails on macOS Apple Silicon          | `pip install xgboost --no-binary xgboost` or `brew install libomp` |
| `pyyaml` build error                          | Upgrade `pip` and `setuptools`                              |
| `pandas` import error                         | Recreate the virtual environment with Python 3.10/3.11      |
