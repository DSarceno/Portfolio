"""Run a Monte-Carlo simulation of the entire tournament."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.ensemble.blender import BlendWeights, ProbabilityBlender
from src.models.base_model import BaseOutcomeModel
from src.models.calibration import ProbabilityCalibrator
from src.models.poisson_model import PoissonScoreModel
from src.prediction.feature_builder import (
    build_inference_feature_matrix,
    build_pairwise_feature_matrix,
)
from src.prediction.predictor import MatchPredictor
from src.simulation.tournament_simulator import TournamentSimulator
from src.utils.config import load_config
from src.utils.io import load_pickle, save_csv
from src.utils.logging_config import get_logger, setup_logging


def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(description="Monte-Carlo simulate the tournament")
    parser.add_argument(
        "--n-runs",
        type=int,
        default=None,
        help="Number of full tournament runs (default: simulation.n_runs from config).",
    )
    parser.add_argument(
        "--competition",
        type=str,
        default="WC",
        help="Competition code that defines the tournament (default: WC).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable the pair-probability cache (slower; only useful for debugging).",
    )
    parser.add_argument(
        "--no-vectorized",
        action="store_true",
        help="Disable the vectorized group-stage engine (slower; debugging only).",
    )
    return parser.parse_args()


def _select_fixtures(
    feature_matrix: pd.DataFrame,
    tournament_year: int,
    competition: str,
) -> pd.DataFrame:
    """Filter the feature matrix to upcoming tournament group fixtures."""
    df = feature_matrix.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    mask_year = df["date"].dt.year == tournament_year
    mask_comp = df["competition"].astype(str).str.upper() == competition.upper()
    if "outcome" in df.columns:
        mask_unplayed = df["outcome"].isna()
    else:
        mask_unplayed = df["score_a"].isna()

    stage_upper = df["stage"].astype(str).str.upper()
    mask_group_stage = stage_upper.str.contains("GROUP", na=False) | (
        df.get("group", pd.Series([""] * len(df))).astype(str).str.strip() != ""
    )

    fixtures = df[mask_year & mask_comp & mask_unplayed & mask_group_stage].copy()
    return fixtures


def _normalize_group_column(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Ensure each fixture has a non-empty ``group`` identifier."""
    out = fixtures.copy()
    if "group" not in out.columns:
        out["group"] = ""

    g = (
        out["group"]
        .astype(str)
        .str.upper()
        .str.replace("GROUP_", "", regex=False)
        .str.replace("GROUP ", "", regex=False)
        .str.strip()
    )
    needs_fallback = g.eq("") | g.eq("NAN") | g.eq("NONE")
    if needs_fallback.any():
        stage_fallback = (
            out.loc[needs_fallback, "stage"]
            .astype(str)
            .str.upper()
            .str.replace("GROUP_", "", regex=False)
            .str.replace("GROUP ", "", regex=False)
            .str.strip()
        )
        g.loc[needs_fallback] = stage_fallback
    out["group"] = g.where(g != "", "X")
    return out


def main() -> int:
    """Simulate the tournament and persist aggregated probabilities."""
    setup_logging(log_file="logs/prediction/simulate_tournament.log")
    logger = get_logger(__name__)
    config = load_config()
    args = parse_args()

    n_runs = args.n_runs or int(config.get("simulation.n_runs", 2000))
    tournament_year = int(config.get("tournament.year", 2026))

    feature_matrix = build_inference_feature_matrix(tournament_year=tournament_year)
    if feature_matrix.empty:
        logger.error("Feature matrix is empty; aborting")
        return 1

    fixtures = _select_fixtures(feature_matrix, tournament_year, args.competition)
    if fixtures.empty:
        logger.error(
            "No upcoming %s %d group-stage fixtures found in the canonical table. "
            "Run: python scripts/bootstrap_historical_data.py --competitions %s "
            "--start-year %d --end-year %d --no-kaggle",
            args.competition,
            tournament_year,
            args.competition,
            tournament_year,
            tournament_year,
        )
        return 1

    fixtures = _normalize_group_column(fixtures)
    n_groups = fixtures["group"].nunique()
    logger.info(
        "Selected %d group-stage fixtures across %d groups (%s %d)",
        len(fixtures),
        n_groups,
        args.competition,
        tournament_year,
    )
    logger.info("Group distribution: %s", fixtures["group"].value_counts().to_dict())

    if n_groups < 2:
        logger.error(
            "Only %d distinct group(s) detected; cannot run a meaningful simulation. "
            "Ensure the football-data.org payload included the 'group' field, or "
            "provide a manual fixture CSV with groups A..L populated.",
            n_groups,
        )
        return 1

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
    elo = None
    try:
        ensemble = load_pickle(models_dir / "rating_ensemble.pkl")
        elo = ensemble.elo
        logger.info("Loaded Elo ratings from rating_ensemble.pkl (%d teams)", len(elo.ratings))
    except FileNotFoundError:
        logger.warning("rating_ensemble.pkl not found; simulator will run on Poisson only")

    blend_weights = BlendWeights.from_config(config)
    logger.info(
        "Blend weights: ratings=%.2f multinomial=%.2f xgboost=%.2f poisson=%.2f",
        blend_weights.ratings,
        blend_weights.multinomial,
        blend_weights.xgboost,
        blend_weights.poisson,
    )
    predictor = MatchPredictor(
        outcome_models={"multinomial": mn, "xgboost": xgb},
        poisson_model=poisson,
        elo=elo,
        calibrator=calibrator,
        blender=ProbabilityBlender(weights=blend_weights),
    )

    # Score every possible knockout pairing with the FULL feature-based model
    # (squad value + engineered features), not just ratings + Poisson. We build a
    # feature row per ordered pair once and look it up in the hot loop.
    teams = sorted({str(t) for t in pd.concat([fixtures["team_a"], fixtures["team_b"]]).dropna()})
    pair_proba: dict[tuple[str, str], np.ndarray] = {}
    pair_features = build_pairwise_feature_matrix(teams, tournament_year=tournament_year)
    if not pair_features.empty:
        proba = predictor.predict_proba(pair_features)
        for (a, b), row in zip(
            zip(pair_features["team_a"].astype(str), pair_features["team_b"].astype(str)),
            proba,
        ):
            pair_proba[(a, b)] = np.asarray(row, dtype=float)
        logger.info("Pre-scored %d ordered pairs with the full model", len(pair_proba))
    else:
        logger.warning("Pairwise feature matrix empty; falling back to ratings + Poisson")

    def _predict_pair(a: str, b: str) -> np.ndarray:
        cached = pair_proba.get((str(a), str(b)))
        if cached is not None:
            return cached
        pred = predictor.predict_single(a, b)
        return np.array([pred.p_home, pred.p_draw, pred.p_away])

    simulator = TournamentSimulator(
        fixtures=fixtures,
        predict_fn=_predict_pair,
        cache_predictions=not args.no_cache,
        vectorized_group_stage=not args.no_vectorized,
    )
    logger.info("Running %d tournament simulations", n_runs)
    summary = simulator.run(n_runs=n_runs)

    save_csv(summary.qualification_probs, "outputs/simulations/tournament_probabilities.csv")
    save_csv(summary.bracket_paths, "outputs/simulations/bracket_paths.csv")
    save_csv(summary.championship_probs, "outputs/simulations/championship_probabilities.csv")
    save_csv(summary.round_reached_probs, "outputs/simulations/round_reached_probabilities.csv")
    logger.info("Simulation finished (n_runs=%d)", n_runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
