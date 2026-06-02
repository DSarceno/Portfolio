# Operations Runbook

## Daily World Cup update

Recommended cron:

```
0 23 * * * cd /opt/wc2026 && /opt/wc2026/.venv/bin/python scripts/update_after_matchday.py --date $(date +\%F)
```

The updater:

1. Pulls fresh match results for the target date.
2. Appends them to `data/interim/matches_unified.csv` (deduplicated by `(date, team_a, team_b)`).
3. Refits the rating ensemble and persists the composite table.
4. Refreshes the tournament-state aggregates.
5. (Optional) re-trains models if `training.retrain_on_update` is `true` in `config.yaml`.

## Matchday checklist

1. `python scripts/update_after_matchday.py --date <today>` — fetch and append.
2. `python scripts/build_ratings.py` — refresh composite ratings.
3. `python scripts/run_pipeline.py` — rebuild features.
4. `python scripts/train_models.py` — refit models (or warm-start).
5. `python scripts/predict_group_stage.py` / `python scripts/predict_knockout.py`.
6. `python scripts/export_quiniela_sheet.py` — refresh quiniela CSVs.
7. `python scripts/simulate_tournament.py` — refresh Monte-Carlo aggregates.

## Monitoring

- Check `logs/updates/update_after_matchday.log` for the most recent update.
- Check `logs/training/train_models.log` for model fit metrics.
- Inspect `outputs/diagnostics/calibration_report.csv` weekly.

## Validation checks

```bash
pytest tests/ -q
python -c "from src.data.data_validator import validate_match_dataframe; \
           import pandas as pd; \
           df = pd.read_csv('data/interim/matches_unified.csv'); \
           print(validate_match_dataframe(df))"
```

## Backup

Persist these directories nightly:

- `data/raw/` (immutable raw snapshots).
- `data/interim/matches_unified.csv` (canonical truth).
- `models/` (pickled artifacts).
- `outputs/` (predictions, picks, diagnostics).

## Recovery

Replay from `data/raw/`:

```bash
python scripts/bootstrap_historical_data.py
python scripts/run_pipeline.py
python scripts/train_models.py
```

## Maintenance

- Rotate logs via the rotating file handler (already configured, 10 MB × 5).
- Periodically prune stale snapshots: keep at least the latest 30 days under `data/raw/tournament_updates/`.
- Re-deploy the API after any model artifact change.
