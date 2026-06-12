# Installation Guide

## 1. Requirements

- Python 3.10–3.12 (developed on 3.11/3.12).
- Git.
- 4+ GB of RAM (XGBoost training and Monte-Carlo simulations).
- Optional: Docker 24+ and docker-compose v2 for the containerised deployment.
- Optional (analysis notebooks): `seaborn`/`jupyter` are already in `requirements.txt`.

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

### Optional Kaggle datasets

- **Results history** — `martj42/international-football-results-from-1872-to-2017` →
  `data/raw/kaggle/results.csv`, imported by `scripts/import_kaggle_history.py`.
- **Squad values (A.2)** — `davidcariboo/player-scores` → `data/raw/kaggle/players-scores/`
  (needs `players.csv` + `player_valuations.csv`). Then build the snapshots:

  ```bash
  python scripts/build_squad_values.py
  ```

  This is optional — without it the pipeline degrades cleanly without the
  squad-value features.

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
