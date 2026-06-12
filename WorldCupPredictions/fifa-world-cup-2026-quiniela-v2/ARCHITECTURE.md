# Architecture

## Why ratings + ML + Poisson are combined

No single family of models captures every aspect of international football:

- **Ratings (Elo, PI, form)** are good at long-horizon ordering of teams but slow to adapt to one-off surprises. They give a stable backbone.
- **Multinomial / XGBoost classifiers** consume the full feature vector (rest, host advantage, tournament state, fatigue proxies, strategy features, **squad value**) and capture interactions that ratings cannot.
- **Poisson scoreline models** derive H/D/A via **Skellam** (the goal-difference distribution, better-calibrated for draws) and keep the Dixon-Coles grid for the exact scorelines quinielas reward.

Blending them (with config-driven weights) produces probabilities that are simultaneously stable (ratings), expressive (ML), and scoreline-consistent (Poisson).

## Why squad value: talent vs results

Elo and PI are *results-based* — they reward what already happened. They systematically under-rate talent-rich teams in a poor run (e.g. Brazil) and over-rate over-performers (e.g. Morocco). **Squad market value** (Transfermarkt, via the Kaggle `player-scores` dataset, aggregated by citizenship as-of date) is an orthogonal *talent* signal. Added as `squad_value_*_diff` features, it improved held-out log-loss (0.9994 → 0.9905) and pulls the simulated title race toward the genuine favorites. Crucially, the same aggregation method is used for every snapshot (2018/2022/2026) so the feature is on a consistent scale across training and prediction.

## Why calibration matters

Quiniela scoring is non-linear in probability. A pick worth 0.55 implied probability and a pick worth 0.70 implied probability can result in very different pool outcomes. Raw model outputs from XGBoost are notoriously overconfident; without isotonic or sigmoid calibration the strategy layer would pick "fake favorites" and routinely overweight low-entropy matches.

Calibration is applied per-model **before** blending and validated through a **leakage-free, out-of-sample reliability diagram**: `collect_holdout_predictions` trains the full stack on matches before a held-out tournament and compares predicted confidence against observed accuracy (ECE) on that tournament (see `notebooks/03_model_comparison.ipynb`).

## Why tournament updates matter

Pre-tournament ratings encode 4+ years of qualifiers and friendlies. They do not reflect:

- A surprise group-stage upset on matchday 1.
- A key suspension after a red card.
- A goalkeeper change after a poor performance.
- A team's actual fitness under tournament travel load.

Without an update loop the predictor would still be predicting matchday 3 with prior-of-the-tournament information. The `tournament_updater` module ingests every just-played match and refreshes:

1. Elo / PI / form ratings.
2. Tournament-state features (`world_cup_points_so_far`, `world_cup_goal_diff_so_far`, ...).
3. The match probability + scoreline models (warm-started from the previous fit).
4. The simulator's initial state, so remaining round probabilities reflect actual standings.

## Why quiniela strategy is separate from forecasting

The most likely outcome is not always the best pick. In a quiniela:

- Picking the same favorite as everyone else cannot win a pool.
- A correct upset is worth differential equity.
- Scoring bonuses reward exact scores and rare outcomes.
- A 38% underdog with low public probability has higher expected pool equity than a 52% favorite that 80% of competitors will also pick.

`src/prediction/quiniela_strategy.py` consumes calibrated probabilities and turns them into picks under multiple risk profiles. Forecasting and pick selection are deliberately decoupled — the same forecast can drive `safe`, `balanced`, `aggressive`, and `contrarian` sheets.

## Why simulation is needed for bracket-aware decisions

Knockout football is path-dependent:

- Group winners face different opponents than runners-up.
- Best-third teams are assigned by complex tiebreak rules.
- Final-matchday group fixtures have asymmetric incentives that distort effort.
- Rest-day differentials compound through the bracket.

The Monte-Carlo simulator (`src/simulation/tournament_simulator.py`) runs the entire tournament thousands of times to produce:

- Group qualification probabilities.
- Round-reached probabilities per team.
- Championship probabilities.
- Most likely knockout paths.
- Match-up frequency matrices.

Every ordered team pair is pre-scored once with the **full blended model** (`feature_builder.build_pairwise_feature_matrix` builds synthetic neutral fixtures carrying squad value + strength diffs, run through the same feature pipeline), so the simulated title race reflects talent — not just the ratings + Poisson fallback the simulator used to be limited to. The strategy layer uses these to make bracket-aware picks.

## Why the cross-tournament backtest is the arbiter

Champion probabilities are seductive but misleading: a results-over-performer can top the list "correctly". The only honest test of a modelling change is **held-out predictive accuracy**. `run_tournament_backtest` (`scripts/backtest_tournaments.py`) trains the full stack on `< year` and scores the held-out tournament (`== year`) leakage-free, reporting log-loss / Brier / RPS / ECE. Every change — blend weights, K factors, the mixture prior (A.4, rejected), squad value (A.2, accepted) — is judged here, not by how the bracket looks.

## Module dependency graph

```
utils         <-- shared by all modules
data          <-- depends on utils
ratings       <-- depends on data, utils
features      <-- depends on data, ratings, utils
models        <-- depends on features, utils
ensemble      <-- depends on models, utils
simulation    <-- depends on ensemble, features, utils
prediction    <-- depends on ensemble, simulation, features, utils
training      <-- depends on models, features, data, utils
api           <-- depends on prediction, simulation, training, utils
```

See [docs/diagrams/system_architecture.md](docs/diagrams/system_architecture.md) for a Mermaid diagram.

## Logging architecture

All logs flow through `src/utils/logging_config.py`:

- `logs/pipeline/` — ETL and feature builds.
- `logs/training/` — model fits, CV scores, calibration metrics.
- `logs/prediction/` — single-match and slate predictions.
- `logs/updates/` — daily World Cup update runs.
- `logs/errors/` — uncaught exceptions and validation failures.

## Failure modes and recovery

- **API outage (football-data.org).** Manual CSV under `data/raw/manual/` is read as fallback. Raw snapshot is timestamped.
- **Model training failure.** Previous model artifact under `models/` remains the canonical artifact until a successful re-train.
- **Update fails mid-way.** The updater is idempotent on a given matchday; re-running it discards partial state and replays from the canonical match table.

## Future extension points

- Live in-play probability updating using minute-by-minute event feeds.
- Player-level injury / suspension feeds plugged into `fatigue_features.py`.
- Direct integration with bookmaker odds APIs to refine the public-bias proxy.
- Pool-specific scoring optimization for the user's own quiniela rules.
