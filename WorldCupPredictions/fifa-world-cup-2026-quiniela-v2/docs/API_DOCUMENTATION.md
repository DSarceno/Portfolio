# API Documentation

The FastAPI service is launched with `make api` or:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive OpenAPI documentation is available at `http://localhost:8000/docs`.

## GET `/health`

Service health.

**Response 200**
```json
{ "status": "ok", "version": "0.2.0" }
```

## GET `/teams`

Returns the list of unique teams in the canonical match table.

**Response 200**
```json
{ "teams": ["Argentina", "Brazil", "..."] }
```

## GET `/matches/upcoming?limit=50`

List upcoming (unscored) matches.

## GET `/matches/completed?limit=100`

List completed (scored) matches.

## POST `/predict/match`

Predict a single match outcome.

**Request**
```json
{ "team_a": "Brazil", "team_b": "Serbia", "risk_profile": "balanced" }
```

**Response 200**
```json
{
  "team_a": "Brazil",
  "team_b": "Serbia",
  "p_home": 0.55,
  "p_draw": 0.25,
  "p_away": 0.20,
  "pick": "H",
  "confidence": 0.55,
  "rationale": "profile=balanced, p=[0.55, 0.25, 0.2]",
  "recommended_scoreline": [2, 1]
}
```

## GET `/predict/day?target_date=2026-06-15`

Predict every match scheduled on `target_date`.

## GET `/predict/tournament`

Predict every unplayed match.

## POST `/simulate/tournament?n_runs=1000`

Run an in-memory Monte-Carlo simulation.

**Response 200**
```json
{
  "qualification_probs": [
    { "team": "Brazil", "qualification_prob": 0.94 },
    "..."
  ],
  "championship_probs": [
    { "team": "Brazil", "championship_prob": 0.18 },
    "..."
  ],
  "n_runs": 1000
}
```

## POST `/strategy/quiniela`

Generate a pick sheet.

**Request**
```json
{ "risk_profile": "aggressive", "include_scoreline": true }
```

**Response 200**
```json
{
  "picks": [
    {
      "team_a": "Germany",
      "team_b": "Japan",
      "p_home": 0.48,
      "p_draw": 0.28,
      "p_away": 0.24,
      "pick": "A",
      "confidence": 0.24,
      "rationale": "profile=aggressive, ...",
      "scoreline": [1, 2]
    }
  ]
}
```

## POST `/update/results`

Append new match results.

**Request**
```json
{
  "matches": [
    {
      "date": "2026-06-15",
      "competition": "WC",
      "team_a": "Argentina",
      "team_b": "Mexico",
      "score_a": 2,
      "score_b": 1
    }
  ]
}
```

**Response 200**
```json
{
  "matches_appended": 1,
  "composite_table_path": "outputs/diagnostics/composite_ratings.csv",
  "state_path": "outputs/diagnostics/tournament_state.csv"
}
```

## GET `/diagnostics/calibration`

Return the latest calibration report CSV records.

## GET `/diagnostics/feature-importance`

Return the latest XGBoost feature-importance table.

## Error responses

All endpoints follow FastAPI conventions:

- `400` — bad request payload.
- `404` — resource missing.
- `500` — uncaught exception (logged under `logs/errors/`).
