# User Guide

This guide walks a non-technical user from a fresh clone to a complete set of quiniela picks for the 2026 FIFA World Cup.

## 1. Purpose

The system forecasts every match of the 2026 World Cup, simulates the tournament thousands of times, and exports four risk-tiered pick sheets:

- `outputs/picks/quiniela_safe.csv`
- `outputs/picks/quiniela_balanced.csv`
- `outputs/picks/quiniela_aggressive.csv`
- `outputs/picks/quiniela_contrarian.csv`

Picks update dynamically as new matchday results arrive.

## 2. Quick start

```bash
git clone <repo-url>
cd fifa-world-cup-2026-quiniela-v2
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
.venv\Scripts\activate             # Windows
pip install -r requirements.txt
cp config/.env.example .env
python scripts/bootstrap_historical_data.py
python scripts/build_ratings.py
python scripts/run_pipeline.py
python scripts/train_models.py
python scripts/predict_group_stage.py
python scripts/export_quiniela_sheet.py
python scripts/simulate_tournament.py
```

## 3. Installation

See [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md) for a full walkthrough.

## 4. Configuration

All knobs live in `config/`:

- `config.yaml` — project paths, tournament metadata, simulation runs.
- `features.yaml` — feature groups and selection.
- `model_params.yaml` — hyperparameters for every model.
- `strategy.yaml` — risk profile thresholds and quiniela scoring rules.

Environment variables override paths via `.env`.

## 5. Running the full pipeline

```bash
make pipeline       # ETL + features
make train          # Train every model
make predict        # Group-stage predictions
make picks          # Export quiniela sheets
make simulate       # Tournament Monte-Carlo
make api            # Launch FastAPI on http://localhost:8000
```

## 6. Updating after matchdays

```bash
python scripts/update_after_matchday.py --date 2026-06-15
```

Or use a manual CSV under `data/raw/manual/`:

```bash
python scripts/update_after_matchday.py --manual-csv data/raw/manual/matchday_5.csv
```

## 7. Output reference

| Path                                                  | Description                                |
| ----------------------------------------------------- | ------------------------------------------ |
| `outputs/predictions/group_stage_predictions.csv`     | Per-match H/D/A probabilities              |
| `outputs/predictions/knockout_predictions.csv`        | Knockout-match probabilities               |
| `outputs/picks/quiniela_safe.csv`                     | Conservative picks                         |
| `outputs/picks/quiniela_balanced.csv`                 | Expected-value picks                       |
| `outputs/picks/quiniela_aggressive.csv`               | Upset-tilted picks                         |
| `outputs/picks/quiniela_contrarian.csv`               | Public-bias-fading picks                   |
| `outputs/simulations/tournament_probabilities.csv`    | Qualification probabilities                |
| `outputs/simulations/championship_probabilities.csv`  | Title probabilities                        |
| `outputs/simulations/bracket_paths.csv`               | Per-run champion/runner-up/third-place log |
| `outputs/diagnostics/composite_ratings.csv`           | Composite team strength index              |
| `outputs/diagnostics/feature_importance.csv`          | XGBoost feature importance                 |

## 8. Common workflows

- **Daily refresh during the tournament**: `make update && make picks && make simulate`.
- **Compare risk profiles**: open `notebooks/04_quiniela_strategy.ipynb`.
- **Audit a single match**: `POST /predict/match` to the API.

## 9. Disclaimers

Football is uncertain. No model can guarantee winning a quiniela. Use the picks as a structured aid, not a guarantee.
