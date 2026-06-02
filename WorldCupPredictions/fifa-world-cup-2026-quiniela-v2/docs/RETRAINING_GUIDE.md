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

## 3. Rebuild features

```bash
python scripts/run_pipeline.py
```

`data/processed/feature_matrix.csv` is rewritten with the latest ratings and tournament state.

## 4. Retrain models

```bash
python scripts/train_models.py
```

The trainer:

1. Temporal-splits the feature matrix.
2. Fits `multinomial_model.pkl`, `xgboost_model.pkl`, `poisson_model.pkl`.
3. Fits `calibrator.pkl` on the validation slice.
4. Persists `feature_columns.pkl` so inference uses the same column order.

## 5. Regenerate predictions and picks

```bash
python scripts/predict_group_stage.py
python scripts/predict_knockout.py     # only after the bracket is fixed
python scripts/export_quiniela_sheet.py
python scripts/simulate_tournament.py
```

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
python scripts/run_pipeline.py
python scripts/train_models.py
```
