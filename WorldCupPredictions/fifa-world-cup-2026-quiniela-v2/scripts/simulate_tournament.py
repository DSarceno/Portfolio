"""Run a Monte-Carlo simulation of the entire tournament."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.data.data_loader import DataLoader
from src.models.base_model import BaseOutcomeModel
from src.models.calibration import ProbabilityCalibrator
from src.models.poisson_model import PoissonScoreModel
from src.prediction.predictor import MatchPredictor
from src.simulation.tournament_simulator import TournamentSimulator
from src.utils.config import load_config
from src.utils.io import load_pickle, save_csv
from src.utils.logging_config import get_logger, setup_logging


def main() -> int:
    """Simulate the tournament and persist aggregated probabilities."""
    setup_logging(log_file="logs/prediction/simulate_tournament.log")
    logger = get_logger(__name__)
    config = load_config()
    n_runs = int(config.get("simulation.n_runs", 10000))

    df = DataLoader().load_matches()
    if df.empty:
        logger.warning("No matches available to simulate")
        return 0
    fixtures = df[df["stage"].astype(str).str.lower().str.contains("group", na=False)].copy()
    if fixtures.empty:
        logger.warning("No group-stage fixtures available; aborting")
        return 0
    if "group" not in fixtures.columns:
        fixtures["group"] = "A"

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

    def _predict_pair(a: str, b: str) -> np.ndarray:
        pred = predictor.predict_single(a, b)
        return np.array([pred.p_home, pred.p_draw, pred.p_away])

    simulator = TournamentSimulator(fixtures=fixtures, predict_fn=_predict_pair)
    summary = simulator.run(n_runs=n_runs)

    save_csv(summary.qualification_probs, "outputs/simulations/tournament_probabilities.csv")
    save_csv(summary.bracket_paths, "outputs/simulations/bracket_paths.csv")
    save_csv(summary.championship_probs, "outputs/simulations/championship_probabilities.csv")
    save_csv(summary.round_reached_probs, "outputs/simulations/round_reached_probabilities.csv")
    logger.info("Simulation finished (n_runs=%d)", n_runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
