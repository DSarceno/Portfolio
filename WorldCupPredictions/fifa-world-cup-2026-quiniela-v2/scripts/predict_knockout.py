"""Predict knockout-stage matches once the bracket is known."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.data_loader import DataLoader
from src.models.base_model import BaseOutcomeModel
from src.models.calibration import ProbabilityCalibrator
from src.models.poisson_model import PoissonScoreModel
from src.prediction.predictor import MatchPredictor
from src.utils.io import load_pickle, save_csv
from src.utils.logging_config import get_logger, setup_logging


def main() -> int:
    """Predict every knockout fixture currently in the canonical table."""
    setup_logging(log_file="logs/prediction/predict_knockout.log")
    logger = get_logger(__name__)

    df = DataLoader().load_matches()
    if df.empty:
        logger.warning("No matches available to predict")
        return 0

    stage_str = df["stage"].astype(str).str.lower()
    knockout_mask = stage_str.str.contains("|".join(["round", "knockout", "final", "quarter", "semi"]), regex=True, na=False)
    fixtures = df[knockout_mask & df["score_a"].isna()].copy()
    if fixtures.empty:
        logger.warning("No knockout fixtures detected")
        return 0

    models_dir = Path("models")
    try:
        xgb = BaseOutcomeModel.load(models_dir / "xgboost_model.pkl")
    except FileNotFoundError:
        xgb = None
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
        outcome_models={"xgboost": xgb}, poisson_model=poisson, calibrator=calibrator
    )
    preds = predictor.predict_matches(fixtures[["team_a", "team_b"]])
    save_csv(preds, "outputs/predictions/knockout_predictions.csv")
    logger.info("Wrote knockout_predictions.csv (%d rows)", len(preds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
