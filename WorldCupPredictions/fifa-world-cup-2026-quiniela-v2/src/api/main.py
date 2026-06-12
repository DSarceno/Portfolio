"""FastAPI application exposing the prediction stack."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.data.data_loader import DataLoader
from src.data.tournament_updater import TournamentUpdater
from src.ensemble.blender import BlendWeights, ProbabilityBlender
from src.ensemble.pick_optimizer import PickOptimizer
from src.models.base_model import BaseOutcomeModel
from src.models.calibration import ProbabilityCalibrator
from src.models.poisson_model import PoissonScoreModel
from src.prediction.daily_update import DailyUpdater
from src.prediction.predictor import MatchPredictor
from src.prediction.quiniela_strategy import QuinielaStrategy
from src.prediction.score_predictor import ScorePredictor
from src.utils.config import load_config
from src.utils.io import load_pickle
from src.utils.logging_config import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)
config = load_config()

app = FastAPI(
    title=config.get("project.name", "FIFA World Cup 2026 Quiniela Predictor V2"),
    version=config.get("project.version", "0.2.0"),
    description=config.get("project.description", ""),
)

MODELS_DIR = Path(config.get("paths.models_dir", "models"))


class HealthResponse(BaseModel):
    """Health response payload."""

    status: str = "ok"
    version: str


class TeamListResponse(BaseModel):
    """Team list response."""

    teams: list[str]


class FixtureSummary(BaseModel):
    """Fixture summary."""

    date: str
    competition: str
    team_a: str
    team_b: str
    score_a: Optional[float]
    score_b: Optional[float]


class FixturesResponse(BaseModel):
    """Fixture list response."""

    matches: list[FixtureSummary]


class PredictMatchRequest(BaseModel):
    """Single-match prediction request."""

    team_a: str = Field(..., description="First team")
    team_b: str = Field(..., description="Second team")
    risk_profile: Optional[str] = Field(default="balanced")


class PredictMatchResponse(BaseModel):
    """Single-match prediction response."""

    team_a: str
    team_b: str
    p_home: float
    p_draw: float
    p_away: float
    pick: str
    confidence: float
    rationale: str
    recommended_scoreline: Optional[tuple[int, int]] = None


class StrategyRequest(BaseModel):
    """Strategy/quiniela export request."""

    risk_profile: str = "balanced"
    include_scoreline: bool = True


class UpdateRequest(BaseModel):
    """Tournament update request."""

    matches: list[dict] = Field(default_factory=list)


def _load_models() -> dict:
    """Try to load every model artifact, returning whatever is available."""
    artifacts: dict = {}
    try:
        artifacts["xgboost"] = BaseOutcomeModel.load(MODELS_DIR / "xgboost_model.pkl")
    except FileNotFoundError:
        artifacts["xgboost"] = None
    try:
        artifacts["multinomial"] = BaseOutcomeModel.load(MODELS_DIR / "multinomial_model.pkl")
    except FileNotFoundError:
        artifacts["multinomial"] = None
    try:
        artifacts["poisson"] = load_pickle(MODELS_DIR / "poisson_model.pkl")
        if not isinstance(artifacts["poisson"], PoissonScoreModel):
            artifacts["poisson"] = None
    except FileNotFoundError:
        artifacts["poisson"] = None
    try:
        artifacts["calibrator"] = load_pickle(MODELS_DIR / "calibrator.pkl")
        if not isinstance(artifacts["calibrator"], ProbabilityCalibrator):
            artifacts["calibrator"] = None
    except FileNotFoundError:
        artifacts["calibrator"] = None
    return artifacts


def _build_predictor() -> MatchPredictor:
    art = _load_models()
    return MatchPredictor(
        outcome_models={
            "multinomial": art.get("multinomial"),
            "xgboost": art.get("xgboost"),
        },
        poisson_model=art.get("poisson"),
        calibrator=art.get("calibrator"),
        blender=ProbabilityBlender(weights=BlendWeights.from_config(config)),
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return the service health."""
    return HealthResponse(status="ok", version=config.get("project.version", "0.2.0"))


@app.get("/teams", response_model=TeamListResponse)
def teams() -> TeamListResponse:
    """Return the unique team list seen in the canonical match table."""
    df = DataLoader().load_matches()
    if df.empty:
        return TeamListResponse(teams=[])
    s = pd.unique(pd.concat([df["team_a"], df["team_b"]]).dropna())
    return TeamListResponse(teams=sorted(map(str, s)))


@app.get("/matches/upcoming", response_model=FixturesResponse)
def upcoming_matches(limit: int = 50) -> FixturesResponse:
    """List matches with missing scores (treated as upcoming)."""
    df = DataLoader().load_matches()
    if df.empty:
        return FixturesResponse(matches=[])
    upcoming = df[df["score_a"].isna() | df["score_b"].isna()].head(limit)
    return FixturesResponse(matches=_to_fixture_summaries(upcoming))


@app.get("/matches/completed", response_model=FixturesResponse)
def completed_matches(limit: int = 100) -> FixturesResponse:
    """List matches with a final score (treated as completed)."""
    df = DataLoader().load_matches()
    if df.empty:
        return FixturesResponse(matches=[])
    completed = df.dropna(subset=["score_a", "score_b"]).tail(limit)
    return FixturesResponse(matches=_to_fixture_summaries(completed))


@app.post("/predict/match", response_model=PredictMatchResponse)
def predict_match(payload: PredictMatchRequest) -> PredictMatchResponse:
    """Predict a single match outcome with the requested risk profile."""
    predictor = _build_predictor()
    prediction = predictor.predict_single(payload.team_a, payload.team_b)
    optimizer = PickOptimizer()
    pick, confidence, rationale = optimizer.pick(
        proba=np.array([prediction.p_home, prediction.p_draw, prediction.p_away]),
        risk_profile=payload.risk_profile or "balanced",
    )
    scoreline = None
    art = _load_models()
    if art.get("poisson") is not None:
        scoreline = art["poisson"].most_likely_scoreline(payload.team_a, payload.team_b)[0]
    return PredictMatchResponse(
        team_a=payload.team_a,
        team_b=payload.team_b,
        p_home=prediction.p_home,
        p_draw=prediction.p_draw,
        p_away=prediction.p_away,
        pick=pick,
        confidence=confidence,
        rationale=rationale,
        recommended_scoreline=scoreline,
    )


@app.get("/predict/day")
def predict_day(target_date: str) -> dict:
    """Predict every match scheduled on *target_date*."""
    df = DataLoader().load_matches()
    if df.empty:
        return {"matches": []}
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    daily = df[df["date"].dt.date == pd.Timestamp(target_date).date()]
    if daily.empty:
        return {"matches": []}
    predictor = _build_predictor()
    predictions = predictor.predict_matches(daily[["team_a", "team_b"]])
    return {"matches": predictions.to_dict(orient="records")}


@app.get("/predict/tournament")
def predict_tournament() -> dict:
    """Predict every unplayed match."""
    df = DataLoader().load_matches()
    if df.empty:
        return {"matches": []}
    unplayed = df[df["score_a"].isna() | df["score_b"].isna()]
    if unplayed.empty:
        return {"matches": []}
    predictor = _build_predictor()
    predictions = predictor.predict_matches(unplayed[["team_a", "team_b"]])
    return {"matches": predictions.to_dict(orient="records")}


@app.post("/simulate/tournament")
def simulate_tournament(n_runs: int = 1000) -> dict:
    """Run an in-memory tournament simulation."""
    from src.simulation.tournament_simulator import TournamentSimulator

    df = DataLoader().load_matches()
    if df.empty:
        return {"error": "No canonical match table available"}
    fixtures = df[df["stage"].astype(str).str.lower().str.contains("group", na=False)].copy()
    if fixtures.empty:
        raise HTTPException(status_code=400, detail="No group-stage fixtures available")
    if "group" not in fixtures.columns:
        fixtures["group"] = "A"
    predictor = _build_predictor()

    def _predict_pair(a: str, b: str) -> np.ndarray:
        pred = predictor.predict_single(a, b)
        return np.array([pred.p_home, pred.p_draw, pred.p_away])

    simulator = TournamentSimulator(fixtures=fixtures, predict_fn=_predict_pair)
    summary = simulator.run(n_runs=n_runs)
    return {
        "qualification_probs": summary.qualification_probs.head(20).to_dict(orient="records"),
        "championship_probs": summary.championship_probs.head(20).to_dict(orient="records"),
        "n_runs": summary.n_runs,
    }


@app.post("/strategy/quiniela")
def strategy_quiniela(payload: StrategyRequest) -> dict:
    """Generate a quiniela pick sheet for every unplayed match."""
    df = DataLoader().load_matches()
    if df.empty:
        return {"picks": []}
    unplayed = df[df["score_a"].isna() | df["score_b"].isna()]
    predictor = _build_predictor()
    probs = predictor.predict_matches(unplayed[["team_a", "team_b"]])

    art = _load_models()
    poisson = art.get("poisson")
    score_predictor = ScorePredictor(poisson) if poisson is not None else None
    strategy = QuinielaStrategy(score_predictor=score_predictor)
    sheet = strategy.generate(
        probs, profile=payload.risk_profile, include_scoreline=payload.include_scoreline
    )
    return {"picks": sheet.to_dict(orient="records")}


@app.post("/update/results")
def update_results(payload: UpdateRequest) -> dict:
    """Append new match results from the request body."""
    new_matches = pd.DataFrame(payload.matches)
    result = DailyUpdater().update(new_matches=new_matches)
    return {
        "matches_appended": result.matches_appended,
        "composite_table_path": result.composite_table_path,
        "state_path": result.state_path,
    }


@app.get("/diagnostics/calibration")
def calibration_report() -> dict:
    """Return the latest calibration report (if available)."""
    path = (
        Path(config.get("paths.outputs_dir", "outputs")) / "diagnostics" / "calibration_report.csv"
    )
    if not path.exists():
        return {"error": f"{path} not found"}
    return {"records": pd.read_csv(path).to_dict(orient="records")}


@app.get("/diagnostics/feature-importance")
def feature_importance() -> dict:
    """Return the latest feature-importance table (if available)."""
    art = _load_models()
    xgb = art.get("xgboost")
    if xgb is None:
        return {"error": "XGBoost model is not loaded"}
    return {"records": xgb.feature_importance().to_dict(orient="records")}


def _to_fixture_summaries(df: pd.DataFrame) -> list[FixtureSummary]:
    """Convert a fixture DataFrame to the typed response shape."""
    out: list[FixtureSummary] = []
    for _, row in df.iterrows():
        out.append(
            FixtureSummary(
                date=str(row.get("date")),
                competition=str(row.get("competition", "")),
                team_a=str(row.get("team_a", "")),
                team_b=str(row.get("team_b", "")),
                score_a=row.get("score_a"),
                score_b=row.get("score_b"),
            )
        )
    return out
