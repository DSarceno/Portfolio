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

## Prediction failures

| Symptom                                  | Fix                                                              |
| ---------------------------------------- | ---------------------------------------------------------------- |
| Uniform probabilities everywhere         | No models are loaded. Run `scripts/train_models.py`.             |
| `KeyError: 'team_a'`                     | Missing column in input. Pass a DataFrame with `team_a`/`team_b`.|
| `ModuleNotFoundError: src.*`             | Run from project root or install in editable mode: `pip install -e .`. |
| `PermissionError: ... outputs\\*.csv`    | The output CSV is open in Excel/another app (Windows file lock). Close it and rerun the step. |

## Squad values (A.2)

| Symptom                                       | Fix                                                              |
| --------------------------------------------- | ---------------------------------------------------------------- |
| "Squad-value table not found" / features absent | Run `scripts/build_squad_values.py` (needs the Kaggle `player-scores` dump). The pipeline degrades cleanly without it. |
| Unmapped team warnings in `build_squad_values`  | Add the citizenship/country alias to `SQUAD_VALUE_NAME_MAP` in `src/data/squad_value_client.py`. |
| Result added but match still "upcoming"         | The result row's `(date, team_a, team_b)` must match the scheduled fixture exactly (order and spelling) so the dedup replaces it. |

## Notebooks

| Symptom                                       | Fix                                                              |
| --------------------------------------------- | ---------------------------------------------------------------- |
| `KeyError: 'pi_combined'` / empty data        | The notebook ran from the wrong cwd. `import nb_style` chdirs to the project root — ensure it runs first. |
| Charts missing squad value / scorelines       | Run `build_squad_values.py` / `predict_scorelines.py` (those cells degrade with a printed notice). |
| Cell timeout during calibration               | The held-out fold trains a model; rerun with `jupyter nbconvert --execute --ExecutePreprocessor.timeout=600`. |
