# FIFA World Cup 2026 Quiniela Predictor V2

Aggressive prediction system for the 2026 FIFA World Cup, optimized for **winning quinielas** (prediction pools) rather than for academic accuracy alone.

## Goal

Generate match-by-match outcome and scoreline forecasts for the 48-team tournament, recompute them dynamically after every matchday, and translate them into multiple risk-tiered quiniela pick sheets (`safe`, `balanced`, `aggressive`, `contrarian`).

## Highlights

- Modular data layer (FIFA rankings, football-data.org, StatsBomb open data, manual CSV fallback, **Transfermarkt squad values via Kaggle**).
- Composite team-strength rating combining Elo, PI rating, and rolling form, with time decay and Bayesian shrinkage toward confederation priors.
- **Squad-value features (A.2)** — talent signal orthogonal to results, the most predictive covariate after Elo.
- Multinomial logistic regression + XGBoost + Poisson scoreline (**Skellam for H/D/A**, Dixon-Coles grid for exact scorelines), with LASSO feature selection.
- Config-driven probability ensemble with isotonic / Platt calibration, validated by an **out-of-sample reliability diagram**.
- Tournament Monte-Carlo simulator with FIFA tiebreak rules and best-third selection — pairings scored with the **full blended model** so the title race reflects talent, not only results.
- **Most-likely scorelines** per match (`scoreline_predictions.csv`).
- **Leakage-free cross-tournament backtester** that is the arbiter for every modelling change.
- Dedicated quiniela strategy layer separating "true probability" from "best pick".
- Daily update workflow (`update_matchday.bat`) that refreshes ratings, features, and predictions after each matchday.
- FastAPI service exposing all key operations; four polished analysis notebooks.

## Quick start

```bash
# 1. Clone and enter the project
git clone <repo-url>
cd fifa-world-cup-2026-quiniela-v2

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
.venv\Scripts\activate             # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp config/.env.example .env

# 5. Bootstrap historical data, squad values and ratings
python scripts/bootstrap_historical_data.py
python scripts/build_squad_values.py        # optional (needs Kaggle player-scores)
python scripts/build_ratings.py

# 6. Build features and train models
python scripts/run_pipeline.py
python scripts/train_models.py

# 7. Predict matches, scorelines and export picks
python scripts/predict_group_stage.py
python scripts/predict_scorelines.py
python scripts/export_quiniela_sheet.py

# 8. Simulate the full tournament
python scripts/simulate_tournament.py

# 9. After each matchday
python scripts/update_after_matchday.py
```

On Windows, `run_all.bat` runs the whole pipeline end-to-end and
`update_matchday.bat` does a one-shot matchday refresh.

## Data sources

| Source                   | Purpose                                       | Mode                 |
| ------------------------ | --------------------------------------------- | -------------------- |
| FIFA ranking snapshots   | Historical ranking points and positions       | Local snapshot       |
| football-data.org        | Fixtures, results, standings                  | API + cached         |
| StatsBomb Open Data      | Event-level historical data when available    | Open data only       |
| Kaggle `player-scores`   | Transfermarkt squad market values (A.2)       | Local dataset        |
| Manual CSV               | Fallback / emergency override / results entry | Local CSV templates  |

Every external pull is persisted under `data/raw/` with a timestamp so results are always reproducible.

## Updating during the World Cup

After matches finish on any given matchday (Windows one-shot: `update_matchday.bat`):

```bash
python scripts/update_after_matchday.py --date 2026-06-15
```

This will:

1. Pull official fixtures/results for that date (or read a manual results CSV).
2. Append them to the canonical match table (dedup replaces the scheduled fixture).
3. Update Elo / PI / form ratings.
4. Refresh tournament-state features.
5. Re-train (or warm-start) outcome and scoreline models.
6. Regenerate probabilities, most-likely scorelines, and remaining-match forecasts.
7. Export new `outputs/picks/quiniela_*.csv` sheets and refresh the simulation.

## Risk profiles

For every upcoming match the predictor exports four picks:

- **safe** — highest single-result probability, conservative draws.
- **balanced** — best expected-value pick using ensemble probabilities.
- **aggressive** — biased toward upsets when fragility / volatility signals fire.
- **contrarian** — fades public bias and reputation; targets pool differentiation.

See [docs/USER_GUIDE.md](docs/USER_GUIDE.md) and [docs/MODELING_GUIDE.md](docs/MODELING_GUIDE.md) for the underlying logic.

## Project layout

See [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) for a complete tour.

## Limitations

- Football is uncertain. No model can guarantee winning a quiniela.
- Penalty shootouts are modeled as Bernoulli draws around a knockout-volatility prior.
- Travel / fatigue features are proxies, not GPS-derived measurements.
- StatsBomb event data is only available for a subset of historical matches.

## License

MIT.
