"""FastAPI application for the F1 Race Predictor."""

import logging
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel as PydanticModel

from src.utils.config import Config
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application startup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="F1 Race Predictor API",
    description="Predict Formula 1 race outcomes using machine learning.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state (loaded at startup)
_predictor = None
_config: Optional[Config] = None


@app.on_event("startup")
async def startup_event() -> None:
    """Load configuration and models on API startup."""
    global _config, _predictor
    setup_logging(level="INFO")
    _config = Config()
    logger.info("F1 Race Predictor API started")


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class PredictionRequest(PydanticModel):
    """Request body for race prediction endpoint."""

    season: int
    round_num: int
    model_name: Optional[str] = None


class DriverPrediction(PydanticModel):
    """Single driver prediction result."""

    driver: str
    team: str
    predicted_position: int
    probability: float
    confidence: float


class PredictionResponse(PydanticModel):
    """Full race prediction response."""

    season: int
    round_num: int
    model_used: str
    predictions: List[DriverPrediction]
    created_at: str


class HealthResponse(PydanticModel):
    """Health check response."""

    status: str
    version: str
    timestamp: str


class ModelInfoResponse(PydanticModel):
    """Available models information."""

    models: List[str]
    default_model: str
    last_trained: Optional[str]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Check API health status.

    Returns:
        Health response with status and timestamp.
    """
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        timestamp=datetime.utcnow().isoformat(),
    )


@app.get("/models", response_model=ModelInfoResponse, tags=["Models"])
async def get_models() -> ModelInfoResponse:
    """List available prediction models.

    Returns:
        Model information including available types.
    """
    from src.models.model_factory import ModelFactory

    available = ModelFactory.get_available_models()
    default = (
        _config.get("DEFAULT_MODEL", "xgboost") if _config else "xgboost"
    )
    return ModelInfoResponse(
        models=available,
        default_model=default,
        last_trained=None,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_race(request: PredictionRequest) -> PredictionResponse:
    """Generate race outcome predictions.

    Args:
        request: PredictionRequest with season, round_num, optional model.

    Returns:
        PredictionResponse with ranked driver predictions.
    """
    try:
        from src.prediction.predictor import RacePredictor

        if _config is None:
            raise HTTPException(status_code=503, detail="Config not loaded")

        predictor = RacePredictor(config=_config._config)
        model_name = request.model_name or "xgboost"

        import os
        model_path = os.path.join(
            _config.get("paths.models_dir", "models"), f"{model_name}.pkl"
        )
        try:
            predictor.load_model(model_path)
        except FileNotFoundError:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_name}' not found. Train models first.",
            )

        predictions_df = predictor.predict_race(request.season, request.round_num)
        driver_preds = [
            DriverPrediction(
                driver=str(row["driver"]),
                team=str(row["team"]),
                predicted_position=int(row["predicted_position"]),
                probability=float(row["probability"]),
                confidence=float(row["confidence"]),
            )
            for _, row in predictions_df.iterrows()
        ]

        return PredictionResponse(
            season=request.season,
            round_num=request.round_num,
            model_used=model_name,
            predictions=driver_preds,
            created_at=datetime.utcnow().isoformat(),
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Prediction failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Prediction failed")


@app.get(
    "/predict/{season}/{round}",
    response_model=PredictionResponse,
    tags=["Predictions"],
)
async def predict_race_get(
    season: int, round: int, model: Optional[str] = None
) -> PredictionResponse:
    """GET endpoint for race predictions.

    Args:
        season: F1 season year.
        round: Race round number.
        model: Optional model type override.

    Returns:
        PredictionResponse with ranked driver predictions.
    """
    request = PredictionRequest(
        season=season, round_num=round, model_name=model
    )
    return await predict_race(request)


@app.get("/seasons", response_model=List[int], tags=["Data"])
async def get_seasons() -> List[int]:
    """Return list of available seasons.

    Returns:
        List of season year integers.
    """
    if _config is None:
        return list(range(2020, 2026))
    start = int(_config.get("data.start_season", 2020))
    end = int(_config.get("data.end_season", 2025))
    return list(range(start, end + 1))


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------


@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle ValueError with 400 status."""
    raise HTTPException(status_code=400, detail=str(exc))
