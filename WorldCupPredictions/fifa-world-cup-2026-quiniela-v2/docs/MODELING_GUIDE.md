# Modeling Guide

## 1. Elo

- **State**: per-team rating, default `1500`.
- **Update**: `R' = R + K * margin * (actual - expected)` with competition-aware `K` (60 for WC, 40 continental, 30 qualifier, 18 friendly).
- **Margin multiplier**: 1.0 for `|gd| = 1`, 1.5 for `gd = 2`, `(11 + gd) / 8` otherwise.
- **Expected score**: standard `1 / (1 + 10^(-(R_A + HA - R_B)/400))` with home advantage HA = 65 (only when `neutral=False`).

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
- **Outcome probabilities**: derived from the upper triangle, diagonal, and lower triangle.

## 8. Calibration

- **Strategies**: isotonic regression (default) or Platt sigmoid.
- **Approach**: one-vs-rest per class; row-normalisation after transform.
- **Validation slice**: last full calendar year before the tournament (`validation_year` from `config.yaml`).

## 9. Blender

- **Weights** (default): `ratings 0.25, multinomial 0.20, xgboost 0.30, poisson 0.25`.
- **Operation**: weighted average of probability matrices, re-normalised.
- Models that aren't fitted are silently skipped.

## 10. Quiniela optimisation

- **Risk profile knobs**: `favorite_threshold`, `upset_tolerance`, `draw_bias`.
- **Aggressive trigger**: `upset_window_score >= 0.5` AND `favorite_fragility_score >= 0.5` AND underdog `p >= upset_tolerance`.
- **Draw trigger**: `p_draw >= 0.26` and small home/away gap.
- **Contrarian rule**: fade the team with the higher public-bias proxy.

## 11. Backtesting

- `rolling_origin_splits` produces folds with growing training windows.
- For each fold, an XGBoost model is fitted and evaluated; if fitting fails, the fallback is `MultinomialOutcomeModel`.
- Aggregate metrics (mean per fold) are written into `outputs/diagnostics/backtest_aggregate.csv`.
