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
