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

from src.data.data_loader import DataLoader
from src.ensemble.blender import BlendWeights, ProbabilityBlender
from src.models.base_model import BaseOutcomeModel
from src.models.calibration import ProbabilityCalibrator
from src.models.poisson_model import PoissonScoreModel
from src.prediction.feature_builder import (
    build_inference_feature_matrix,
    build_pairwise_feature_matrix,
)
from src.prediction.predictor import MatchPredictor
from src.simulation.bracket_generator import BracketPair, load_known_round_of_32
from src.simulation.tournament_simulator import TournamentSimulator
from src.utils.config import load_config
from src.utils.io import load_pickle, save_csv
from src.utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


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
    parser.add_argument(
        "--bracket-csv",
        type=str,
        default="data/raw/manual/wc2026_knockout_bracket.csv",
        help=(
            "Path to the manual knockout-bracket CSV. Used in knockout-only mode "
            "(once the group stage is over)."
        ),
    )
    parser.add_argument(
        "--knockout-only",
        action="store_true",
        help=(
            "Force knockout-only mode: seed the real round-of-32 from --bracket-csv "
            "and simulate only the knockout (no group re-simulation). Auto-enabled "
            "when there are no unplayed group-stage fixtures left."
        ),
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
    # A fixture is group-stage if its stage says so, or it carries a real group
    # label. Guard the group-label clause against NaN: ``str(NaN) == "nan"``,
    # which is non-empty, so a naive ``!= ""`` would wrongly tag scoreless
    # knockout rows (group is blank) as group fixtures.
    group_label = (
        df.get("group", pd.Series([""] * len(df), index=df.index))
        .astype(str)
        .str.strip()
        .str.upper()
    )
    has_group = ~group_label.isin(["", "NAN", "NONE"])
    mask_group_stage = stage_upper.str.contains("GROUP", na=False) | has_group

    fixtures = df[mask_year & mask_comp & mask_unplayed & mask_group_stage].copy()
    return fixtures


def _bracket_advancers(bracket_csv: str | Path) -> dict[frozenset, str]:
    """Infer each tie's advancer from the next round it feeds into.

    The bracket CSV carries the tournament tree (``feeds_winner_into`` links each
    tie to the next-round match it feeds). Once the user fills that next-round
    match with the team that advanced, the winner of the feeding tie is simply
    whichever of its two teams appears there. This recovers the winner of a
    **penalty-decided (drawn) tie**, whose score (e.g. 1-1) does not reveal it.

    Returns a map ``frozenset({team_a, team_b}) -> advancer`` for every tie whose
    advancer can be read off the (filled) next-round match.
    """
    path = Path(bracket_csv)
    if not path.exists():
        return {}
    bracket = pd.read_csv(path)
    required = {"match_id", "feeds_winner_into", "team_a", "team_b"}
    if not required.issubset(bracket.columns):
        return {}

    def _val(x: object) -> str | None:
        return None if pd.isna(x) or str(x).strip() == "" else str(x).strip()

    teams_by_id: dict[str, set[str]] = {}
    for r in bracket.itertuples():
        mid = _val(r.match_id)
        if mid is not None:
            teams_by_id[mid] = {t for t in (_val(r.team_a), _val(r.team_b)) if t is not None}

    advancers: dict[frozenset, str] = {}
    for r in bracket.itertuples():
        a, b, target = _val(r.team_a), _val(r.team_b), _val(r.feeds_winner_into)
        if a is None or b is None or target is None:
            continue
        advanced = {a, b} & teams_by_id.get(target, set())
        if len(advanced) == 1:  # exactly one of the two appears in the next round
            advancers[frozenset({a, b})] = next(iter(advanced))
    return advancers


def _played_knockout_winners(
    tournament_year: int, bracket_csv: str | Path | None = None
) -> dict[frozenset, str]:
    """Map already-played knockout ties to their winner.

    Lets the knockout-only simulation condition on real results: a decided tie
    always advances its actual winner instead of being re-simulated. A **drawn**
    tie's winner is not in the score (penalty shootout), so it is recovered from
    the bracket's next-round fill via :func:`_bracket_advancers` when
    *bracket_csv* is given; ties whose next round is not yet filled are left to
    the simulation.
    """
    df = DataLoader().load_matches()
    if df.empty or "score_a" not in df.columns:
        return {}
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    stage = df["stage"].astype(str).str.upper()
    mask_ko = stage.str.contains("ROUND|QUARTER|SEMI|FINAL|R16|R32|KNOCKOUT|THIRD", na=False)
    played = df[(df["date"].dt.year == tournament_year) & mask_ko].copy()

    winners: dict[frozenset, str] = {}
    drawn: list[frozenset] = []
    for r in played.itertuples():
        sa = pd.to_numeric(r.score_a, errors="coerce")
        sb = pd.to_numeric(r.score_b, errors="coerce")
        if pd.isna(sa) or pd.isna(sb):
            continue  # unplayed or malformed
        key = frozenset({str(r.team_a), str(r.team_b)})
        if sa == sb:
            drawn.append(key)  # penalty shootout — winner not in the score
        else:
            winners[key] = str(r.team_a) if sa > sb else str(r.team_b)

    if drawn and bracket_csv is not None:
        advancers = _bracket_advancers(bracket_csv)
        for key in drawn:
            if key not in winners and key in advancers:
                winners[key] = advancers[key]
                logger.info(
                    "Drawn knockout tie %s decided on penalties; advancer %s inferred "
                    "from the bracket's next round",
                    set(key),
                    advancers[key],
                )
    return winners


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

    group_fixtures = _select_fixtures(feature_matrix, tournament_year, args.competition)

    # Decide the mode. While groups still have unplayed matches we simulate the
    # whole tournament from scratch (the pre-/early-tournament use case). Once
    # every group is decided there are no unplayed group fixtures, so we seed the
    # *real* round-of-32 from the manual bracket and simulate only the knockout —
    # otherwise the group re-simulation silently drops the already-decided teams.
    knockout_only = args.knockout_only or group_fixtures.empty
    bracket: list[BracketPair] = []

    if knockout_only:
        try:
            bracket = load_known_round_of_32(args.bracket_csv)
        except (FileNotFoundError, ValueError) as exc:
            logger.error("Knockout-only mode requested but the bracket is not ready: %s", exc)
            return 1
        teams = sorted({t for p in bracket for t in (p.team_a, p.team_b)})
        known_winners = _played_knockout_winners(tournament_year, bracket_csv=args.bracket_csv)
        logger.info(
            "Knockout-only mode: seeded the real round-of-32 (%d ties, %d teams) from %s; "
            "%d already-played tie(s) fixed to their actual winner",
            len(bracket),
            len(teams),
            args.bracket_csv,
            len(known_winners),
        )
    else:
        group_fixtures = _normalize_group_column(group_fixtures)
        n_groups = group_fixtures["group"].nunique()
        logger.info(
            "Selected %d group-stage fixtures across %d groups (%s %d)",
            len(group_fixtures),
            n_groups,
            args.competition,
            tournament_year,
        )
        logger.info(
            "Group distribution: %s", group_fixtures["group"].value_counts().to_dict()
        )

        if n_groups < 2:
            logger.error(
                "Only %d distinct group(s) detected; cannot run a meaningful simulation. "
                "Ensure the football-data.org payload included the 'group' field, or "
                "provide a manual fixture CSV with groups A..L populated.",
                n_groups,
            )
            return 1
        teams = sorted(
            {str(t) for t in pd.concat([group_fixtures["team_a"], group_fixtures["team_b"]]).dropna()}
        )

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

    if knockout_only:
        sim_fixtures = pd.DataFrame(
            {
                "team_a": [p.team_a for p in bracket],
                "team_b": [p.team_b for p in bracket],
            }
        )
        simulator = TournamentSimulator(
            fixtures=sim_fixtures,
            predict_fn=_predict_pair,
            cache_predictions=not args.no_cache,
            vectorized_group_stage=False,
        )
        logger.info("Running %d knockout-only simulations", n_runs)
        summary = simulator.run_from_known_bracket(bracket, n_runs=n_runs, known_winners=known_winners)
    else:
        simulator = TournamentSimulator(
            fixtures=group_fixtures,
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
