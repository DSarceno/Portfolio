"""Predict the most likely scorelines for every upcoming match.

Combines the full blended H/D/A model (ratings + multinomial + XGBoost + Poisson
+ squad value) with the Poisson Dixon-Coles scoreline grid, emitting one row per
match with the predicted outcome and the top-3 most probable scorelines.

Note: the H/D/A probabilities come from the full blended model, while the exact
scorelines come from the Poisson model only. They can occasionally disagree
(e.g. the blend favours team A but the Poisson mode is a draw) — both are shown
so you can judge.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.ensemble.blender import BlendWeights, ProbabilityBlender
from src.models.base_model import BaseOutcomeModel
from src.models.calibration import ProbabilityCalibrator
from src.models.poisson_model import PoissonScoreModel
from src.prediction.feature_builder import build_inference_feature_matrix, filter_upcoming
from src.prediction.predictor import MatchPredictor
from src.prediction.score_predictor import ScorePredictor
from src.utils.config import load_config
from src.utils.io import load_pickle, save_csv
from src.utils.logging_config import get_logger, setup_logging

TOP_K = 3


def main() -> int:
    """Write the most-likely-scorelines sheet for upcoming matches."""
    setup_logging(log_file="logs/prediction/predict_scorelines.log")
    logger = get_logger(__name__)
    config = load_config()
    tournament_year = config.get("tournament.year", 2026)

    feature_matrix = build_inference_feature_matrix(tournament_year=tournament_year)
    if feature_matrix.empty:
        logger.error("Feature matrix is empty; aborting")
        return 1

    fixtures = filter_upcoming(feature_matrix).reset_index(drop=True)
    if fixtures.empty:
        logger.warning("No upcoming matches to predict")
        return 0
    logger.info("Predicting scorelines for %d upcoming matches", len(fixtures))

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

    if poisson is None:
        logger.error("Poisson model not found; cannot recommend scorelines")
        return 1

    predictor = MatchPredictor(
        outcome_models={"multinomial": mn, "xgboost": xgb},
        poisson_model=poisson,
        calibrator=calibrator,
        blender=ProbabilityBlender(weights=BlendWeights.from_config(config)),
    )
    outcomes = predictor.predict_matches(fixtures)
    score_predictor = ScorePredictor(poisson)

    rows: list[dict[str, object]] = []
    for i, fixture in fixtures.iterrows():
        team_a = str(fixture["team_a"])
        team_b = str(fixture["team_b"])
        row: dict[str, object] = {
            "date": str(fixture.get("date", "")),
            "stage": str(fixture.get("stage", "")),
            "team_a": team_a,
            "team_b": team_b,
            "pred_outcome": outcomes.loc[i, "most_likely_outcome"],
            "p_home": round(float(outcomes.loc[i, "p_home"]), 4),
            "p_draw": round(float(outcomes.loc[i, "p_draw"]), 4),
            "p_away": round(float(outcomes.loc[i, "p_away"]), 4),
        }
        top = score_predictor.recommend(team_a, team_b, k=TOP_K)
        for rank in range(TOP_K):
            if rank < len(top):
                rec = top.iloc[rank]
                row[f"top{rank + 1}_score"] = f"{int(rec['score_a'])}-{int(rec['score_b'])}"
                row[f"top{rank + 1}_prob"] = round(float(rec["probability"]), 4)
            else:
                row[f"top{rank + 1}_score"] = ""
                row[f"top{rank + 1}_prob"] = 0.0
        rows.append(row)

    sheet = pd.DataFrame(rows)
    save_csv(sheet, "outputs/predictions/scoreline_predictions.csv")
    logger.info("Wrote scoreline_predictions.csv (%d matches)", len(sheet))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
