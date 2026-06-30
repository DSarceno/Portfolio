# Troubleshooting

## Installation

| Issue                                              | Fix                                                                |
| -------------------------------------------------- | ------------------------------------------------------------------ |
| `pip` fails to build wheels                        | Upgrade pip/setuptools: `pip install -U pip setuptools wheel`.     |
| `xgboost` import error (`libomp.dylib not found`)  | macOS: `brew install libomp`. Linux: `apt-get install libgomp1`.   |
| Python version mismatch                            | Use Python 3.10 or 3.11. Recreate the virtual environment.         |

## API ingestion

| Symptom                                  | Fix                                                              |
| ---------------------------------------- | ---------------------------------------------------------------- |
| `403 Forbidden` from football-data.org   | Verify `FOOTBALL_DATA_API_KEY` and competition tier permissions. |
| Empty match table after bootstrap        | Check `data/raw/football_data/`, `data/raw/manual/` and logs.    |
| `httpx.ReadTimeout`                      | Increase `FootballDataClient.timeout`. Retry later.              |

## Matchday updates (manual results CSV)

| Symptom                                                       | Cause / Fix                                                                                                                                                                                                                                                                          |
| ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| A played match still shows as "upcoming" in `scoreline_predictions.csv` | The result row's `(date, team_a, team_b)` doesn't match the scheduled fixture, so it landed on a phantom row while the real fixture stayed `outcome=NaN`. Most common cause: the **UTC date** — football-data stores kickoff in UTC and WC2026 evening matches are often the next UTC day (look it up with `fixture_date.bat <team>`). Also check `team_a`/`team_b` **order and spelling** match the fixture exactly. |
| A team is double-counted in standings/ratings after I fixed a row | Editing a date/team of an already-ingested row changes the `(date, team_a, team_b)` dedup key, leaving the old row as an **orphan**. The `--manual-csv` ingest now runs **replace mode** by default (purges `tournament_update` rows + re-ingests), which removes orphans automatically — just re-run `update_matchday.bat`. (Older orphans from before this change: re-run once and they're gone.) |
| `TypeError: '>' not supported between 'float' and 'str'` during update | A malformed CSV row (e.g. a comma typed as a period like `WC.GROUP_STAGE`) shifted columns, putting a string into `score_a/score_b`. `world_cup_state` now skips non-numeric scores with a warning instead of crashing, but the row is still wrong — fix the CSV so every row has the full 9 comma-separated fields. Validate with `python -c "import pandas as pd; d=pd.read_csv('data/raw/manual/wc2026_results.csv'); print(d.dtypes)"` (scores must be int/float). |

## Data ingestion failures

- Inspect `logs/pipeline/bootstrap_historical.log` and `logs/pipeline/run_pipeline.log`.
- Validate the canonical table with `python -c "from src.data.data_validator import validate_match_dataframe; ..."`.

## Model training failures

| Symptom                                                | Fix                                                                |
| ------------------------------------------------------ | ------------------------------------------------------------------ |
| `ValueError: Training feature matrix is empty`         | Run `scripts/run_pipeline.py` first.                               |
| XGBoost segfault on Apple Silicon                      | Reinstall XGBoost with `pip install --no-binary xgboost xgboost`.  |
| Calibrator not fitted                                  | Provide a non-empty validation slice (`validation_year` config).   |

## Simulation failures

| Symptom                                  | Fix                                                              |
| ---------------------------------------- | ---------------------------------------------------------------- |
| Empty group fixtures                     | Populate `data/raw/manual/manual_fixtures_WC2026.csv`.            |
| Simulator runs forever                   | Reduce `simulation.n_runs` in `config.yaml`.                     |
| Bracket has < 8 pairs                    | Group stage produced fewer than 16 advancing teams. Inspect the standings; ensure all groups are full. |
| Post-groups: only G/J/K/L teams in the champion list | Old bug — the sim re-simulated only *unplayed* group fixtures. Fixed: with no unplayed group games it auto-switches to **knockout-only** mode, seeding the real R32 from `wc2026_knockout_bracket.csv`. Ensure the bracket's 16 R32 ties are filled. |
| `Round-of-32 tie ... is not filled in yet` | Knockout-only sim needs all 16 R32 matchups in the bracket CSV. Fill `team_a`/`team_b` for every `ROUND_OF_32` row. |
| Eliminated team still has championship probability | Make sure its tie's result is in `wc2026_results.csv`. **Decisive score:** the sim forces it via `_played_knockout_winners`. **Penalty shootout (drawn score):** the score can't reveal the winner, so the sim reads the advancer from the bracket's next round — you must also **fill that next-round matchup** in `wc2026_knockout_bracket.csv` with the team that advanced (`_bracket_advancers` infers it via `feeds_winner_into`). Until the next round is filled, that one drawn tie is left to the simulation, so the loser keeps some probability. |

## Prediction failures

| Symptom                                  | Fix                                                              |
| ---------------------------------------- | ---------------------------------------------------------------- |
| Uniform probabilities everywhere         | No models are loaded. Run `scripts/train_models.py` (a one-off, deliberate retrain — not part of the matchday loop). |
| Favourites flat or flipped (e.g. Brazil < Japan, Germany ~50%) | Historically a mid-tournament retrain broke the **multinomial** (saturated ~99% away — the `stage_*` indicators are constant in training but 1 at live knockout serving, so `StandardScaler` exploded them). **Fixed (2026-06-29):** the model drops near-constant features + clips standardized values, the stack was re-tuned, and the weight restored to `0.20`. If you see this again, it means models were retrained with a regression in that guard — restore from a `models_backup_<ts>/`, and never retrain in the matchday loop (`update_matchday.bat` must not run `train_models`). See `docs/AUDIT_2026-06-29.md` §2. |
| A played knockout result reverted to "unplayed" | You re-ran `ingest_knockout_bracket.py` over a tie that already had a result — it now **skips** played ties, so update the script if this recurs. Re-enter the result via `update_after_matchday.py --manual-csv`. |
| Knockout result not merging (duplicate/phantom row) | The results row must use the **same date + team order** as the bracket fixture (`stage=ROUND_OF_32` etc., empty group). Look the date up with `fixture_date.bat <team>` or `fixture_date.bat knockout`. |
| `KeyError: 'team_a'`                     | Missing column in input. Pass a DataFrame with `team_a`/`team_b`.|
| `ModuleNotFoundError: src.*`             | Run from project root or install in editable mode: `pip install -e .`. |
| `PermissionError: ... outputs\\*.csv`    | The output CSV is open in Excel/another app (Windows file lock). Close it and rerun the step. |
| `.bat` errors like `'EM'/'cho'/'oto' is not recognized` (first char of each line eaten) | The batch file has **LF** line endings; `cmd.exe` needs **CRLF**. This happens after editing a `.bat` with a tool/editor that writes Unix endings. Fix: re-save as CRLF (e.g. `python -c "import pathlib,sys; p=pathlib.Path(sys.argv[1]); p.write_bytes(p.read_bytes().replace(b'\r\n',b'\n').replace(b'\r',b'\n').replace(b'\n',b'\r\n'))" update_matchday.bat`). All repo `.bat` files must stay CRLF. |

## Squad values (A.2)

| Symptom                                       | Fix                                                              |
| --------------------------------------------- | ---------------------------------------------------------------- |
| "Squad-value table not found" / features absent | Run `scripts/build_squad_values.py` (needs the Kaggle `player-scores` dump). The pipeline degrades cleanly without it. |
| Unmapped team warnings in `build_squad_values`  | Add the citizenship/country alias to `SQUAD_VALUE_NAME_MAP` in `src/data/squad_value_client.py`. |

## Notebooks

| Symptom                                       | Fix                                                              |
| --------------------------------------------- | ---------------------------------------------------------------- |
| `KeyError: 'pi_combined'` / empty data        | The notebook ran from the wrong cwd. `import nb_style` chdirs to the project root — ensure it runs first. |
| Charts missing squad value / scorelines       | Run `build_squad_values.py` / `predict_scorelines.py` (those cells degrade with a printed notice). |
| Cell timeout during calibration               | The held-out fold trains a model; rerun with `jupyter nbconvert --execute --ExecutePreprocessor.timeout=600`. |
