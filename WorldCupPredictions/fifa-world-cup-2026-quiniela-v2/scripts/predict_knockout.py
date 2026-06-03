"""Predict knockout-stage matches once the bracket is known."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.base_model import BaseOutcomeModel
from src.models.calibration import ProbabilityCalibrator
from src.models.poisson_model import PoissonScoreModel
from src.prediction.feature_builder import (
    build_inference_feature_matrix,
    filter_upcoming,
)
from src.prediction.predictor import MatchPredictor
from src.utils.config import load_config
from src.utils.io import load_pickle, save_csv
from src.utils.logging_config import get_logger, setup_logging

KNOCKOUT_KEYWORDS = ("round", "knockout", "final", "quarter", "semi", "r16", "r32")


def main() -> int:
    """Predict every knockout fixture currently in the canonical table."""
    setup_logging(log_file="logs/prediction/predict_knockout.log")
    logger = get_logger(__name__)
    config = load_config()
    tournament_year = config.get("tournament.year", 2026)

    feature_matrix = build_inference_feature_matrix(tournament_year=tournament_year)
    if feature_matrix.empty:
        logger.error("Feature matrix is empty; aborting")
        return 1

    upcoming = filter_upcoming(feature_matrix)
    if upcoming.empty:
        logger.warning("No upcoming matches at all; nothing to predict")
        return 0

    stage_lc = upcoming["stage"].astype(str).str.lower()
    knockout_mask = stage_lc.apply(lambda s: any(kw in s for kw in KNOCKOUT_KEYWORDS))
    fixtures = upcoming[knockout_mask].copy()
    if fixtures.empty:
        logger.warning("No knockout fixtures detected; nothing to predict")
        return 0
    logger.info("Predicting %d upcoming knockout fixtures", len(fixtures))

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
    )
    preds = predictor.predict_matches(fixtures)

    if "date" in fixtures.columns:
        preds.insert(0, "date", fixtures["date"].astype(str).values)
    if "stage" in fixtures.columns:
        preds.insert(1, "stage", fixtures["stage"].astype(str).values)

    save_csv(preds, "outputs/predictions/knockout_predictions.csv")
    logger.info("Wrote knockout_predictions.csv (%d rows)", len(preds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
