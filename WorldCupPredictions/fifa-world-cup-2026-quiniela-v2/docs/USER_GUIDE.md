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
python scripts/build_squad_values.py   # A.2 squad value (optional; needs Kaggle player-scores)
python scripts/run_pipeline.py
python scripts/train_models.py
python scripts/predict_group_stage.py
python scripts/predict_scorelines.py   # most likely scorelines per match
python scripts/export_quiniela_sheet.py
python scripts/simulate_tournament.py
```

> **Squad value (A.2):** `build_squad_values.py` reads the Kaggle
> `davidcariboo/player-scores` dataset (place it under
> `data/raw/kaggle/players-scores/`) and writes
> `data/raw/squad_values/squad_values.csv`. It is **optional** — if the dataset
> is absent the pipeline degrades cleanly without the squad-value features. It is
> idempotent: rerun only when the Kaggle dump or the optional manual override
> (`data/raw/squad_values/squad_values_2026_manual.csv`) changes, **not** every
> matchday.

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

**One-shot (Windows):** `update_matchday.bat` ingests recent results + the knockout
bracket and regenerates ratings → predictions (group + knockout) → scorelines → picks →
simulation in one go, then archives a **dated snapshot** of `outputs/` to
`outputs/snapshots/<date>/`. It does *not* rebuild the static squad-value snapshots, and it does
*not* retrain the models — those are trained pre-tournament and **frozen** for the duration
(retraining mid-tournament is unnecessary and was shown to destabilise the blend):

```bat
update_matchday.bat                       REM default: data\raw\manual\wc2026_results.csv
update_matchday.bat path\to\results.csv   REM a specific results CSV
update_matchday.bat 2026-06-15            REM pull results for a date from football-data.org
```

Add one row per played match to `data/raw/manual/wc2026_results.csv` (canonical
schema: `date,competition,stage,group,team_a,team_b,score_a,score_b,neutral_venue`).
Close any open `outputs/*.csv` (Excel locks them).

**The manual CSV is the source of truth.** The `--manual-csv` ingest runs in
**replace mode** (purge + re-ingest) by default: every existing `tournament_update`
row is replaced by the CSV on each run. So you just edit the CSV freely — add rows,
fix a wrong date, correct a team name — and the canonical table is rebuilt to match
it exactly. Edits propagate and no orphan/duplicate rows survive. (For the *why*, see
[TROUBLESHOOTING](TROUBLESHOOTING.md): editing an already-ingested row changes the
`(date, team_a, team_b)` dedup key, so a plain append would otherwise leave the stale
row behind.) Use `--append-only` to upsert without purging (e.g. a partial CSV).

> **Use the fixture's UTC date.** football-data.org stores kickoff dates in UTC, and
> WC2026 is in North America, so an evening match is often the *next* calendar day in
> UTC. If your date doesn't match the scheduled fixture's UTC date, the result lands on
> a phantom match and the real fixture stays "upcoming". Look up the right date with
> `fixture_date.bat <team>` (e.g. `fixture_date.bat Argentina`) before entering a result.

**Manual equivalents:**

```bash
python scripts/update_after_matchday.py --date 2026-06-15
python scripts/update_after_matchday.py --manual-csv data/raw/manual/wc2026_results.csv
python scripts/update_after_matchday.py --manual-csv data/raw/manual/wc2026_results.csv --append-only
```

### Knockout phase

After the group stage, keep using `update_matchday.bat` — it also handles the bracket:

1. **Enter a knockout result** like any result row, but with `stage=ROUND_OF_32` (or
   `QUARTER_FINAL` / `SEMI_FINAL` / `FINAL` / `THIRD_PLACE_FINAL`), an empty `group`, and
   the **same date + team order** as the bracket fixture. **For a penalty shootout, record only
   the regulation/extra-time score (the draw, e.g. `1,1`) — never add the shootout goals**, or
   you would corrupt the ratings. The team that advanced is captured when you fill the next
   round's matchup in the bracket (step 2): the predictor, the notebook **and the Monte-Carlo
   simulator** all read the advancer from there (the simulator infers a drawn tie's winner from
   the next-round fill via `feeds_winner_into`, so champion odds condition on the real shootout
   result, not a coin flip).
2. **Fill the next round's matchups** in `data/raw/manual/wc2026_knockout_bracket.csv`
   (`date`, `team_a`, `team_b`) as they are decided. `update_matchday.bat` ingests them.
   Re-running is safe — already-played ties are never overwritten.
3. **See the bracket** any time with `fixture_date.bat knockout`. The visual bracket
   (results + projection) lives in `notebooks/05_knockout_bracket.ipynb`.

## 7. Output reference

| Path                                                  | Description                                |
| ----------------------------------------------------- | ------------------------------------------ |
| `outputs/predictions/group_stage_predictions.csv`     | Per-match H/D/A probabilities              |
| `outputs/predictions/scoreline_predictions.csv`       | Per-match outcome + top-3 likely scorelines |
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
| `outputs/snapshots/<date>/`                           | Dated archive of the above + `manifest.json` (one per matchday) |

## 8. Common workflows

- **Daily refresh during the tournament**: `make update && make picks && make simulate`.
- **Compare risk profiles**: open `notebooks/04_quiniela_strategy.ipynb`.
- **Audit a single match**: `POST /predict/match` to the API.

## 9. Disclaimers

Football is uncertain. No model can guarantee winning a quiniela. Use the picks as a structured aid, not a guarantee.
