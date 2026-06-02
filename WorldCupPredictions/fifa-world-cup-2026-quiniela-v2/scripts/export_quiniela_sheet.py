"""Export quiniela pick sheets for every risk profile."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.data_loader import DataLoader
from src.ensemble.pick_optimizer import PickOptimizer
from src.models.base_model import BaseOutcomeModel
from src.models.calibration import ProbabilityCalibrator
from src.models.poisson_model import PoissonScoreModel
from src.prediction.predictor import MatchPredictor
from src.prediction.quiniela_strategy import QuinielaStrategy
from src.prediction.score_predictor import ScorePredictor
from src.utils.constants import RISK_PROFILES
from src.utils.io import load_pickle, save_csv
from src.utils.logging_config import get_logger, setup_logging


def main() -> int:
    """Generate one CSV per risk profile under ``outputs/picks/``."""
    setup_logging(log_file="logs/prediction/export_quiniela_sheet.log")
    logger = get_logger(__name__)

    df = DataLoader().load_matches()
    if df.empty:
        logger.warning("No matches available to predict")
        return 0
    unplayed = df[df["score_a"].isna() | df["score_b"].isna()]
    if unplayed.empty:
        logger.warning("No unplayed matches to pick")
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
    probabilities = predictor.predict_matches(unplayed[["team_a", "team_b"]])

    score_predictor = ScorePredictor(poisson) if poisson is not None else None
    strategy = QuinielaStrategy(
        optimizer=PickOptimizer(),
        score_predictor=score_predictor,
    )
    sheets = strategy.generate_all(probabilities)
    for profile in RISK_PROFILES:
        sheet = sheets[profile]
        path = f"outputs/picks/quiniela_{profile}.csv"
        save_csv(sheet, path)
        logger.info("Wrote %s (%d picks)", path, len(sheet))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
