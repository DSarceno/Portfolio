# Modeling Guide

## 1. Elo

- **State**: per-team rating, default `1500`.
- **Update**: `R' = R + K * margin * (actual - expected)` with competition-aware `K` (**80 for WC**, 40 continental, 30 qualifier, **8 friendly**). Recalibrated June 2026 so the World Cup weighs more and friendlies less.
- **Margin multiplier**: 1.0 for `|gd| = 1`, 1.5 for `gd = 2`, `(11 + gd) / 8` otherwise.
- **Expected score**: standard `1 / (1 + 10^(-(R_A + HA - R_B)/400))` with home advantage HA = 65 (only when `neutral=False`).
- **Time decay**: each update is weighted by `exp(-xi * days_to_latest)` when `ratings.time_decay.enabled` (default `xi_elo=0.0015`, half-life ~462 days), so recent matches count more (Dixon-Coles 1997).

## 2. PI Rating

- **State**: separate home and away ratings per team.
- **Expected goal difference**: `(10^|d|/c - 1)` with `c = 3` (Constantinou & Fenton).
- **Update**: `R_home += λ * error`, `R_away += γ * λ * error` with `λ = 0.054` and `γ = 0.79`.

## 3. Rolling form

- **Window**: last 5 matches per team.
- **Form score**: `points + 0.25 * goal_diff`.
- Used both as an explicit feature and as a backbone signal in the ensemble.

## 4. Composite rating

- Z-score each rating system and apply weights `(0.55, 0.30, 0.15)` for `(elo, pi, form)` by default. Tunable via the `RatingEnsemble` constructor.

## 5. Multinomial logistic regression

- **Features**: standardised via `StandardScaler`.
- **Near-constant guard (2026-06-29):** before fitting, features whose single most-frequent training
  value covers ≥99% of rows are dropped, and standardized values are clipped to ±8 SD. A standardized
  linear model cannot learn a feature with no training variance, and such a feature is a train/serve
  hazard: the `stage_group`/`stage_knockout` indicators are constant=0 across the historical training
  data but 1 for every live knockout fixture, so without the guard `StandardScaler`'s tiny `scale_`
  turned them into a ~100-SD lever that saturated the softmax. See `docs/AUDIT_2026-06-29.md` §2.
- **Model**: `LogisticRegression(multi_class="multinomial", solver="lbfgs", C=1.0, class_weight="balanced")`.
- **Output**: row-normalised probabilities over `{H, D, A}`.

## 6. XGBoost

- **Tree method**: histogram.
- **Hyperparameters**: see [config/model_params.yaml](../config/model_params.yaml).
- **Output**: `XGBClassifier.predict_proba`, re-normalised to sum to 1.
- **Feature importance**: exported to `outputs/diagnostics/feature_importance.csv`.

## 7. Poisson scoreline

- **Parameters**: per-team attack and defence strengths (mean goals scored / conceded over the average), home advantage scalar.
- **Expected goals**: `λ_A = base * attack_A * defense_B * exp(HA / 2)`, symmetric for `B`.
- **Joint scoreline**: outer product of two Poisson PMFs, Dixon-Coles correction:
  - `(0,0) *= 1 - λ_A λ_B ρ`
  - `(0,1) *= 1 + λ_A ρ`
  - `(1,0) *= 1 + λ_B ρ`
  - `(1,1) *= 1 - ρ`
- **Outcome probabilities (H/D/A)**: by default via **Skellam** (`scipy.stats.skellam` on the goal difference of two Poissons) — cleaner and better-calibrated for draws (Karlis & Ntzoufras 2009). The Dixon-Coles scoreline grid is retained for **exact scorelines** (`top_k_scorelines`, used by `predict_scorelines.py`). Toggle with `models.poisson.hyperparameters.use_skellam`.
  - **Matrix orientation (critical):** `matrix[i, j] = P(team_a scores i, team_b scores j)`, so team_a wins on the lower triangle (`i > j`). A historical bug had this inverted.
- **Time decay**: `fit()` weights matches by `exp(-xi * days_since)` when `time_decay_xi > 0` and a `date` column is present (default `xi_poisson=0.0020`).
- **Shrinkage**: attack/defence are pulled toward confederation priors (see §13) to tame teams with little data.

## 8. Calibration

- **Strategies**: Platt **sigmoid (default since 2026-06-29)** or isotonic regression. Set via
  `config/model_params.yaml::calibration.strategy`, read by `train_models.py`. Switched off isotonic
  because it overfits the small per-tournament calibration set (a holdout A/B on 64 matches gave
  log-loss 1.45 vs sigmoid 1.04) and collapses mid-table ties onto identical step values; sigmoid is
  smooth and far more stable on limited data (Niculescu-Mizil & Caruana 2005).
- **Approach**: one-vs-rest per class; row-normalisation after transform.
- **Validation slice**: last full calendar year before the tournament (`validation_year` from `config.yaml`).
- **Out-of-sample reliability**: `backtester.collect_holdout_predictions(matches, year)` trains the full stack on `< year` and returns `(proba, y_true)` for that tournament, used to draw the leakage-free reliability diagram and ECE in `notebooks/03_model_comparison.ipynb`.

## 9. Blender

- **Weights**: `ratings 0.35, multinomial 0.20, xgboost 0.30, poisson 0.15` (June 2026 — ratings up, the minnow-noisy Poisson down). Read from `config/model_params.yaml::ensemble.weights` via `BlendWeights.from_config()`. The multinomial weight was temporarily `0.0` during the 2026 knockout phase after a retrain exposed a stage-feature train/serve skew; the bug is now fixed (near-constant guard, §5 above) and the weight was **restored to `0.20`** via a deliberate backtest-gated re-tune on 2026-06-29 (a holdout weight grid put 0.0/0.10/0.20 within fold noise, 0.20 best accuracy). See `docs/AUDIT_2026-06-29.md` §2 and the "Models are frozen" policy in OPERATIONS_RUNBOOK (the matchday loop still never retrains).
  - **Note:** weights live in `model_params.yaml`, **not** `config.yaml`; `Config.get()` only reads the latter, so the wiring goes through `config.model_params`. **Every** entry point must build its blender with `BlendWeights.from_config()` — `predict_group_stage`, `predict_knockout`, `predict_scorelines`, `simulate_tournament`, `export_quiniela_sheet`, and the API. (`predict_knockout` was the lone exception until 2026-06-29: it used the `BlendWeights()` defaults, silently blending the degenerate multinomial at 0.20 despite the config — fixed, with regression test `tests/unit/test_predict_knockout_weights.py`.)
- **Operation**: weighted average of probability matrices, re-normalised.
- Models that aren't fitted (or whose feature columns are absent) are silently skipped.

## 10. Quiniela optimisation

- **Risk profile knobs**: `favorite_threshold`, `upset_tolerance`, `draw_bias`.
- **Aggressive trigger**: `upset_window_score >= 0.5` AND `favorite_fragility_score >= 0.5` AND underdog `p >= upset_tolerance`.
- **Draw trigger**: `p_draw >= 0.26` and small home/away gap.
- **Contrarian rule**: fade the team with the higher public-bias proxy.

## 11. Backtesting

- **Rolling-origin** (`Backtester`): `rolling_origin_splits` produces folds with growing training windows; each fits an XGBoost model (fallback `MultinomialOutcomeModel`).
- **Cross-tournament** (`run_tournament_backtest`, `scripts/backtest_tournaments.py`): trains the **full blended stack** on `< year` and evaluates the held-out tournament (`== year`), leakage-free (ratings and squad-value features are derived only from pre-year data via an as-of join). Metrics: log-loss, Brier, ordinal RPS, accuracy, ECE. Output: `outputs/diagnostics/backtest_WC_<years>.csv`.

  ```bash
  python scripts/backtest_tournaments.py --years 2018 2022
  ```

  **This is the arbiter:** any modelling change (weights, K, mixture prior, new features) is accepted only if it improves held-out log-loss here, not by how "realistic" the champion list looks. Baseline (production config): log-loss ~0.99 on 2018+2022 vs uniform `ln 3 ≈ 1.0986`.

## 12. LASSO feature selection

- Before fitting XGBoost/multinomial, `Trainer._lasso_select_features` runs an L1-multinomial logistic over the ~25-feature pool and drops features with no independent signal (Groll-Schauberger-Tutz 2015). Floor at `min_features=5`; logs the dropped set. Config: `training.lasso.{enabled,C}`.

## 13. Shrinkage & mixture prior

- **Bayesian shrinkage** (`src/ratings/shrinkage.py`): pulls Elo/PI/Poisson toward confederation priors with weight `n / (n + K)` so teams with little data don't get inflated ratings. Config: `ratings.shrinkage.{k_elo,k_pi,k_poisson}` (defaults 30/30/35).
- **Mixture prior (A.4)** — *implemented but disabled by default*: splits each confederation into an elite/regular tier (with a `min_matches` gate so a minnow can never be elite). The cross-tournament backtest showed the hand-tuned priors **worsened** log-loss (+2.3%), so `ratings.shrinkage.mixture_prior.enabled: false`. Re-enable only after re-tuning the priors and re-validating.

## 14. Squad value (A.2)

- The most predictive covariate after Elo (Groll et al.). `scripts/build_squad_values.py` aggregates the Kaggle `player-scores` valuations by citizenship as-of date into `data/raw/squad_values/squad_values.csv` (same method for 2018/2022/2026 → consistent scale). `squad_value_features.py` adds date-aware `squad_value_total_diff` and `squad_value_top11_diff`. Validated by the backtest (log-loss 0.9994 → 0.9905). See [the design spec](superpowers/specs/2026-06-11-a2-squad-value-design.md).

## 15. Full-model tournament simulation

- The Monte-Carlo simulator scores every ordered team pair once with the **full blended model** (not just ratings + Poisson). `feature_builder.build_pairwise_feature_matrix` builds synthetic neutral fixtures for all pairs, runs them through the same feature pipeline (squad value as-of + strength diffs + neutral context), and `simulate_tournament.py` looks them up in the hot loop — so the championship distribution reflects talent, not only results.
