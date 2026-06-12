# Retraining Guide

## 1. Refresh raw data

```bash
python scripts/bootstrap_historical_data.py
```

Snapshots are timestamped under `data/raw/`. Old snapshots are preserved.

## 2. Rebuild ratings

```bash
python scripts/build_ratings.py
```

The composite rating table is written to `outputs/diagnostics/composite_ratings.csv` and a `RatingEnsemble` pickle to `models/rating_ensemble.pkl`.

## 2b. (Optional) rebuild squad values

```bash
python scripts/build_squad_values.py
```

Only needed when the Kaggle `player-scores` dump or the manual override changes;
squad values are static during the tournament. Writes `data/raw/squad_values/squad_values.csv`.

## 3. Rebuild features

```bash
python scripts/run_pipeline.py
```

`data/processed/feature_matrix.csv` is rewritten with the latest ratings, tournament state, and squad-value diffs (the as-of join loads `squad_values.csv` when present).

## 4. Retrain models

```bash
python scripts/train_models.py
```

The trainer:

1. Temporal-splits the feature matrix.
2. Runs **LASSO** feature selection over the engineered pool (logs the dropped features).
3. Fits `multinomial_model.pkl`, `xgboost_model.pkl`, `poisson_model.pkl` (Poisson shrunk toward confederation priors).
4. Fits `calibrator.pkl` on the validation slice.
5. Persists `feature_columns.pkl` so inference uses the same column order.

## 5. Regenerate predictions and picks

```bash
python scripts/predict_group_stage.py
python scripts/predict_scorelines.py   # most-likely scorelines per match
python scripts/predict_knockout.py     # only after the bracket is fixed
python scripts/export_quiniela_sheet.py
python scripts/simulate_tournament.py
```

## 5b. Validate the retrain (the arbiter)

Any change to weights, K factors, shrinkage, or features must be justified by the
**leakage-free cross-tournament backtest**, not by how the champion list looks:

```bash
python scripts/backtest_tournaments.py --years 2018 2022
```

Compare held-out log-loss/Brier/RPS against the previous config; keep the change
only if it improves (or at least does not worsen) the metrics. For an A/B (e.g.
with vs without a feature), call `run_tournament_backtest(...)` directly with the
two configurations. This is exactly how the mixture prior (A.4) was rejected and
squad value (A.2) was accepted.

## 6. Recover from a failed update

The matchday updater is idempotent on a given date. Re-run with the same arguments:

```bash
python scripts/update_after_matchday.py --date 2026-06-15
```

If the canonical table is corrupted, restore the previous snapshot:

```bash
cp data/raw/tournament_updates/update_YYYYMMDDTHHMMSSZ.csv \
   data/interim/matches_unified.csv
```

## 7. Reset everything

```bash
make clean
rm -rf data/interim/* data/processed/* models/*.pkl outputs/*
python scripts/bootstrap_historical_data.py
python scripts/build_squad_values.py    # if the Kaggle player-scores dump is present
python scripts/run_pipeline.py
python scripts/train_models.py
```
