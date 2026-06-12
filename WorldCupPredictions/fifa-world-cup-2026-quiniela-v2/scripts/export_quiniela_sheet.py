"""Export quiniela pick sheets for every risk profile."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.ensemble.blender import BlendWeights, ProbabilityBlender
from src.ensemble.pick_optimizer import PickOptimizer
from src.models.base_model import BaseOutcomeModel
from src.models.calibration import ProbabilityCalibrator
from src.models.poisson_model import PoissonScoreModel
from src.prediction.feature_builder import (
    build_inference_feature_matrix,
    filter_upcoming,
)
from src.prediction.predictor import MatchPredictor
from src.prediction.quiniela_strategy import QuinielaStrategy
from src.prediction.score_predictor import ScorePredictor
from src.utils.config import load_config
from src.utils.constants import RISK_PROFILES
from src.utils.io import load_pickle, save_csv
from src.utils.logging_config import get_logger, setup_logging


def main() -> int:
    """Generate one CSV per risk profile under ``outputs/picks/``."""
    setup_logging(log_file="logs/prediction/export_quiniela_sheet.log")
    logger = get_logger(__name__)
    config = load_config()
    tournament_year = config.get("tournament.year", 2026)

    feature_matrix = build_inference_feature_matrix(tournament_year=tournament_year)
    if feature_matrix.empty:
        logger.error("Feature matrix is empty; aborting")
        return 1

    fixtures = filter_upcoming(feature_matrix)
    if fixtures.empty:
        logger.warning("No upcoming matches to pick")
        return 0
    logger.info("Generating picks for %d upcoming matches", len(fixtures))

    models_dir = Path("models")
    try:
        xgb = BaseOutcomeModel.load(models_dir / "xgboost_model.pkl")
    except FileNotFoundError:
        xgb = None
    try:
        mn = BaseOutcomeModel.load(models_dir / "multinomial_model.pkl")
    except FileNotFoundError:
        mn = None
    try:
        poisson = load_pickle(models_dir / "poisson_model.pkl")
        if not isinstance(poisson, PoissonScoreModel):
            poisson = None
    except FileNotFoundError:
        poisson = None
    try:
        calibrator = load_pickle(models_dir / "calibrator.pkl")
        if not isinstance(calibrator, ProbabilityCalibrator):
            calibrator = None
    except FileNotFoundError:
        calibrator = None

    predictor = MatchPredictor(
        outcome_models={"multinomial": mn, "xgboost": xgb},
        poisson_model=poisson,
        calibrator=calibrator,
        blender=ProbabilityBlender(weights=BlendWeights.from_config(config)),
    )
    probabilities = predictor.predict_matches(fixtures)

    strategy_features = [
        "upset_window_score",
        "favorite_fragility_score",
        "public_bias_proxy",
    ]
    fixtures_idx = fixtures.reset_index(drop=True)
    for col in strategy_features:
        if col in fixtures_idx.columns:
            probabilities[col] = fixtures_idx[col].values
    if "date" in fixtures_idx.columns:
        probabilities.insert(0, "date", fixtures_idx["date"].astype(str).values)
    if "stage" in fixtures_idx.columns:
        probabilities.insert(1, "stage", fixtures_idx["stage"].astype(str).values)

    score_predictor = ScorePredictor(poisson) if poisson is not None else None
    strategy = QuinielaStrategy(
        optimizer=PickOptimizer(),
        score_predictor=score_predictor,
    )
    sheets = strategy.generate_all(probabilities)
    for profile in RISK_PROFILES:
        sheet = sheets[profile]
        if "date" in probabilities.columns:
            sheet.insert(0, "date", probabilities["date"].values)
        if "stage" in probabilities.columns:
            sheet.insert(1, "stage", probabilities["stage"].values)
        path = f"outputs/picks/quiniela_{profile}.csv"
        save_csv(sheet, path)
        logger.info("Wrote %s (%d picks)", path, len(sheet))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
