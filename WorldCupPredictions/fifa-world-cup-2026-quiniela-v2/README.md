# FIFA World Cup 2026 Quiniela Predictor V2

Aggressive prediction system for the 2026 FIFA World Cup, optimized for **winning quinielas** (prediction pools) rather than for academic accuracy alone.

## Goal

Generate match-by-match outcome and scoreline forecasts for the 48-team tournament, recompute them dynamically after every matchday, and translate them into multiple risk-tiered quiniela pick sheets (`safe`, `balanced`, `aggressive`, `contrarian`).

## Highlights

- Modular data layer (FIFA rankings, football-data.org, StatsBomb open data, manual CSV fallback).
- Composite team-strength rating combining Elo, PI rating, and rolling form.
- Multinomial logistic regression + XGBoost + Poisson scoreline + Dixon-Coles low-score adjustment.
- Probability ensemble with isotonic and Platt calibration.
- Tournament Monte-Carlo simulator with FIFA tiebreak rules and best-third selection.
- Dedicated quiniela strategy layer separating "true probability" from "best pick".
- Daily update workflow that refreshes ratings, features, and predictions after each matchday.
- FastAPI service exposing all key operations.

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

# 5. Bootstrap historical data and ratings
python scripts/bootstrap_historical_data.py
python scripts/build_ratings.py

# 6. Train models
python scripts/train_models.py

# 7. Predict group-stage matches and export picks
python scripts/predict_group_stage.py
python scripts/export_quiniela_sheet.py

# 8. Simulate the full tournament
python scripts/simulate_tournament.py

# 9. After each matchday
python scripts/update_after_matchday.py
```

## Data sources

| Source                   | Purpose                                       | Mode                 |
| ------------------------ | --------------------------------------------- | -------------------- |
| FIFA ranking snapshots   | Historical ranking points and positions       | Local snapshot       |
| football-data.org        | Fixtures, results, standings                  | API + cached         |
| StatsBomb Open Data      | Event-level historical data when available    | Open data only       |
| Manual CSV               | Fallback / emergency override                 | Local CSV templates  |

Every external pull is persisted under `data/raw/` with a timestamp so results are always reproducible.

## Updating during the World Cup

After matches finish on any given matchday:

```bash
python scripts/update_after_matchday.py --date 2026-06-15
```

This will:

1. Pull official fixtures/results for that date.
2. Append them to the canonical match table.
3. Update Elo / PI / form ratings.
4. Refresh tournament-state features.
5. Re-train (or warm-start) outcome and scoreline models.
6. Regenerate probabilities for remaining matches.
7. Export new `outputs/picks/quiniela_*.csv` sheets.

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
