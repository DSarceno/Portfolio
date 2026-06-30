# Operations Runbook

## One-shot matchday refresh (Windows)

`update_matchday.bat` ingests recent results + the knockout bracket and regenerates
ratings → predictions (group + knockout) → scorelines → picks → simulation in a single
command. It does **not** rebuild the static squad-value snapshots, and — by design — it
does **not** retrain the predictive models (see "Models are frozen" below):

```bat
update_matchday.bat                       REM default: data\raw\manual\wc2026_results.csv
update_matchday.bat path\to\results.csv   REM a specific results CSV
update_matchday.bat 2026-06-15            REM pull results for a date from football-data.org
```

The 8 steps: (1) ingest results, (2) ingest the knockout bracket
(`ingest_knockout_bracket.py`), (3) rebuild ratings, (4) predict group stage,
(5) predict knockout, (6) predict scorelines, (7) export quinielas, (8) simulate.

Add one row per played match to `data/raw/manual/wc2026_results.csv` (canonical
schema: `date,competition,stage,group,team_a,team_b,score_a,score_b,neutral_venue`).
The `--manual-csv` ingest runs in **replace mode** by default: it purges all existing
`tournament_update` rows and re-ingests the CSV, so the canonical table mirrors the CSV
exactly and edits (corrected dates/teams) propagate without leaving orphan rows. Use the
fixture's **UTC date** (football-data stores kickoff in UTC; look it up with
`fixture_date.bat <team>`). Pass `--append-only` to upsert without purging. Close any
open `outputs/*.csv` first (Excel locks them → `PermissionError`).

### Knockout phase

Once the group stage is over, the workflow is the same `update_matchday.bat`, plus the
knockout bracket:

- **Enter a knockout result** as a normal results row, but with `stage=ROUND_OF_32`
  (or `QUARTER_FINAL` / `SEMI_FINAL` / `FINAL` / `THIRD_PLACE_FINAL`), an empty `group`,
  and the **exact same date + team order** as the bracket fixture (the merge key is
  `(date, team_a, team_b)`; reversing the order or changing the date creates a phantom
  row). **For a penalty shootout, record only the regulation/extra-time score (the draw) —
  never add the shootout goals (it would corrupt the ratings).** The advancing team is
  captured when you fill the next round in the bracket CSV: the predictor, the notebook, and
  the **Monte-Carlo simulator** all read it from there (the sim infers a drawn tie's winner
  from the `feeds_winner_into` next-round fill, so champion odds condition on the real shootout
  result rather than re-flipping it).
- **Add upcoming-round matchups** to `data/raw/manual/wc2026_knockout_bracket.csv` as
  they become known (fill `date`, `team_a`, `team_b` for the `ROUND_OF_16` rows, then
  `QUARTER_FINAL`, etc.). `update_matchday.bat` ingests them automatically. Re-ingesting
  is **safe**: `ingest_knockout_bracket.py` skips any tie that already has a result, so
  it never blanks an entered score. View the current bracket with `fixture_date.bat knockout`.

### Models are frozen during the tournament

The predictive models (XGBoost, Poisson, multinomial, calibrator) are trained and
validated **before** the tournament and are **not** retrained during it.
`update_matchday.bat` deliberately omits `run_pipeline.py` and `train_models.py`.
Reason: refitting on a handful of in-tournament games is unnecessary and destabilising —
a mid-tournament retrain once drove the multinomial model to a degenerate state
(~99% away for every fixture), which poisoned the blend. Ratings (Elo/pi/form) **do**
update each matchday (cheap and stable). To retrain on purpose, run `run_pipeline.py`
then `train_models.py` by hand and re-validate with the backtest (see RETRAINING_GUIDE).

> **Multinomial — fixed and restored (2026-06-29).** A stage-feature train/serve skew once made it
> explode (`stage_*` indicators are constant=0 in historical training but 1 for live knockout fixtures,
> so `StandardScaler` blew them up — see `docs/AUDIT_2026-06-29.md` §2). Fixed in
> `src/models/multinomial_model.py` (drops near-constant features + clips standardized values), then
> the stack was **redeployed via a deliberate backtest-gated re-tune** and the blend weight restored to
> the designed **0.20** (four-model blend). Calibration was also switched isotonic → **sigmoid**
> (`config/model_params.yaml::calibration.strategy`). The matchday loop stays frozen regardless — this
> was a one-off re-tune, not a change to `update_matchday.bat`.

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

> Models are **frozen** during the tournament; the matchday loop does not retrain
> them (see "Models are frozen" above).

## Matchday checklist (mirrors `update_matchday.bat`)

1. `python scripts/update_after_matchday.py --manual-csv data/raw/manual/wc2026_results.csv` — ingest results (replace mode).
2. `python scripts/ingest_knockout_bracket.py` — ingest the knockout bracket (skips already-played ties).
3. `python scripts/build_ratings.py` — refresh composite ratings.
4. `python scripts/predict_group_stage.py` — H/D/A for any remaining group games.
5. `python scripts/predict_knockout.py` — knockout fixtures already in the table.
6. `python scripts/predict_scorelines.py` — most-likely scorelines per match.
7. `python scripts/export_quiniela_sheet.py` — refresh quiniela CSVs.
8. `python scripts/simulate_tournament.py` — refresh Monte-Carlo aggregates (auto knockout-only post-groups, conditioned on played results).
9. `python scripts/snapshot_outputs.py` — archive a dated copy of `outputs/` to `outputs/snapshots/<date>/` with a manifest (git commit, played-match count, championship top-5). The live outputs are overwritten every run; this preserves the matchday's forecast and feeds the pre-tournament-vs-actual comparison. Snapshots are committed (exempted from `.gitignore`). A failure here is non-fatal — it does not abort the matchday. Pass `--label <name>` for a custom snapshot (e.g. `pre_tournament`).

> `run_pipeline.py` and `train_models.py` are **not** in the loop (models frozen).
> `build_squad_values.py` is also out — squad values are static during the tournament;
> rerun only when the Kaggle dump changes.

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

- `data/raw/` (immutable raw snapshots, incl. `squad_values/` and `manual/wc2026_results.csv`).
- `data/interim/matches_unified.csv` (canonical truth).
- `models/` (pickled artifacts).
- `outputs/` (predictions, picks, diagnostics).

## Recovery

Replay from `data/raw/`:

```bash
python scripts/bootstrap_historical_data.py
python scripts/build_squad_values.py    # if the Kaggle player-scores dump is present
python scripts/run_pipeline.py
python scripts/train_models.py
```

## Maintenance

- Rotate logs via the rotating file handler (already configured, 10 MB × 5).
- Periodically prune stale snapshots: keep at least the latest 30 days under `data/raw/tournament_updates/`.
- Re-deploy the API after any model artifact change.
