"""Rolling-origin and cross-tournament backtesters."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from src.ensemble.blender import BlendWeights, ProbabilityBlender
from src.features.build_features import build_match_feature_matrix, select_feature_columns
from src.models.multinomial_model import MultinomialOutcomeModel
from src.models.xgboost_model import XGBoostOutcomeModel
from src.prediction.predictor import MatchPredictor
from src.ratings.rating_ensemble import RatingEnsemble
from src.ratings.shrinkage import RatingShrinker
from src.training.cross_validation import rolling_origin_splits
from src.training.evaluator import Evaluator
from src.training.trainer import Trainer
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class BacktestResult:
    """Container with per-fold metrics."""

    per_fold: pd.DataFrame
    aggregate: pd.Series


class Backtester:
    """Rolling-origin backtester for outcome models."""

    def __init__(self, n_folds: int = 5, min_train_size: int = 200) -> None:
        """Initialize the backtester.

        Args:
            n_folds: Number of rolling-origin folds.
            min_train_size: Minimum training rows per fold.
        """
        self.n_folds = n_folds
        self.min_train_size = min_train_size

    def run(
        self,
        features: pd.DataFrame,
        feature_columns: Optional[list[str]] = None,
        label_column: str = "outcome",
        date_column: str = "date",
    ) -> BacktestResult:
        """Run the backtest.

        Args:
            features: Feature DataFrame with labels.
            feature_columns: Optional explicit feature column list.
            label_column: Outcome label column.
            date_column: Date column used to order rows.

        Returns:
            :class:`BacktestResult`.

        Raises:
            ValueError: If *features* is empty.
        """
        if features.empty:
            raise ValueError("No features supplied to backtester")

        features = features.copy()
        features[date_column] = pd.to_datetime(features[date_column], errors="coerce")
        features = features.dropna(subset=[date_column, label_column]).sort_values(date_column)

        cols = feature_columns or select_feature_columns(features)
        evaluator = Evaluator()
        records: list[dict[str, float]] = []
        for fold_idx, (train_idx, test_idx) in enumerate(
            rolling_origin_splits(
                features, n_splits=self.n_folds, min_train_size=self.min_train_size
            )
        ):
            train = features.iloc[train_idx]
            test = features.iloc[test_idx]
            if train.empty or test.empty:
                continue

            model = XGBoostOutcomeModel(feature_columns=cols)
            try:
                model.fit(train[cols], train[label_column].astype(str))
            except (ValueError, RuntimeError) as exc:
                logger.warning(
                    "XGBoost failed on fold %d (%s); falling back to multinomial", fold_idx, exc
                )
                model = MultinomialOutcomeModel(feature_columns=cols)
                model.fit(train[cols], train[label_column].astype(str))

            proba = model.predict_proba(test[cols])
            report = evaluator.evaluate(
                y_true=test[label_column].astype(str).tolist(),
                proba=proba,
            )
            records.append({"fold": fold_idx, **report.to_dict()})

        per_fold = pd.DataFrame(records)
        if per_fold.empty:
            aggregate = pd.Series(dtype=float)
        else:
            aggregate = per_fold.drop(columns=["fold"]).mean()
        logger.info("Backtest aggregate: %s", aggregate.to_dict())
        return BacktestResult(per_fold=per_fold, aggregate=aggregate)


def _ranked_probability_score(y_true: Sequence[str], proba: np.ndarray) -> float:
    """Mean RPS over the ordered outcome scale ``[H, D, A]``.

    The draw sits between a home and an away win, so an ordinal RPS rewards
    predictions that miss by one category less than those that miss by two
    (Constantinou & Fenton 2012).
    """
    order = {"H": 0, "D": 1, "A": 2}
    cum_pred = np.cumsum(np.asarray(proba, dtype=float), axis=1)
    total = 0.0
    n = 0
    for i, label in enumerate(y_true):
        if label not in order:
            continue
        actual = np.zeros(3)
        actual[order[label]] = 1.0
        cum_actual = np.cumsum(actual)
        total += float(np.sum((cum_pred[i] - cum_actual) ** 2)) / 2.0
        n += 1
    return total / n if n else float("nan")


def _score_holdout(
    df: pd.DataFrame,
    year: int,
    competition: str,
    weights: BlendWeights,
    shrinker_factory: Optional[Callable[[], RatingShrinker]],
    hyperparameters: Optional[dict],
    lasso_select: bool,
    lasso_C: float,
    squad_values: Optional[pd.DataFrame],
    min_train_matches: int,
    played: pd.Series,
) -> Optional[tuple[pd.DataFrame, np.ndarray, int]]:
    """Train the full stack on ``< year`` and score the ``== year`` tournament.

    Returns ``(test_feats, proba, n_train)`` or ``None`` when the fold is skipped
    (insufficient history or no test matches). Leakage-safe: ratings and
    features for the test fixtures are derived only from pre-year data.
    """
    train_matches = df[df["year"] < year]
    comp_upper = df["competition"].astype(str).str.upper()
    test_matches = df[(df["year"] == year) & (comp_upper == competition.upper()) & played]
    if len(train_matches) < min_train_matches or test_matches.empty:
        return None

    shrinker = shrinker_factory() if shrinker_factory is not None else None
    ensemble = RatingEnsemble(shrinker=shrinker).fit(train_matches)
    composite = ensemble.composite_table()

    combined = pd.concat([train_matches, test_matches], ignore_index=True)
    feats = build_match_feature_matrix(
        combined, pd.DataFrame(), composite, squad_values=squad_values
    )
    feats["date"] = pd.to_datetime(feats["date"], errors="coerce")
    feats["year"] = feats["date"].dt.year
    feats_comp = feats["competition"].astype(str).str.upper()
    train_feats = feats[(feats["year"] < year) & feats["outcome"].notna()].copy()
    test_feats = feats[
        (feats["year"] == year) & (feats_comp == competition.upper()) & feats["outcome"].notna()
    ].copy()
    if train_feats.empty or test_feats.empty:
        return None

    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = Trainer(
            models_dir=tmp_dir,
            hyperparameters=hyperparameters,
            shrinker=shrinker_factory() if shrinker_factory is not None else None,
            lasso_select=lasso_select,
            lasso_C=lasso_C,
        )
        outputs = trainer.fit(train_feats, val_features=None)

    predictor = MatchPredictor(
        outcome_models={"multinomial": outputs.multinomial, "xgboost": outputs.xgboost},
        poisson_model=outputs.poisson,
        elo=ensemble.elo,
        calibrator=None,
        blender=ProbabilityBlender(weights=weights),
    )
    proba = predictor.predict_proba(test_feats)
    return test_feats, proba, len(train_feats)


def run_tournament_backtest(
    matches: pd.DataFrame,
    target_years: Sequence[int],
    competition: str = "WC",
    shrinker_factory: Optional[Callable[[], RatingShrinker]] = None,
    blend_weights: Optional[BlendWeights] = None,
    hyperparameters: Optional[dict] = None,
    lasso_select: bool = True,
    lasso_C: float = 0.1,
    min_train_matches: int = 500,
    squad_values: Optional[pd.DataFrame] = None,
) -> BacktestResult:
    """Leakage-free cross-tournament backtest of the full prediction stack.

    For each ``year`` in *target_years*, the full stack (rating ensemble +
    shrinkage + multinomial + XGBoost + Poisson, blended with *blend_weights*)
    is trained on every match strictly **before** that year and evaluated on the
    target tournament's matches. Ratings and features for the test fixtures are
    derived solely from pre-year data, so no future information leaks back.

    This is the harness that quantifies whether modelling changes (mixture
    prior, blend weights, K factors, ...) actually improve held-out tournament
    log-loss / Brier / RPS, following the Groll et al. protocol.

    Args:
        matches: Canonical match table (needs ``date``, ``team_a``, ``team_b``,
            ``competition``, ``outcome`` and scoreline columns).
        target_years: Tournament years to hold out and predict, e.g.
            ``[2018, 2022]``.
        competition: Competition code identifying the target tournament.
        shrinker_factory: Zero-arg callable returning a *fresh*
            :class:`RatingShrinker` per fold (shrinkage mutates state). ``None``
            disables shrinkage.
        blend_weights: Ensemble weights. Defaults to :class:`BlendWeights`
            defaults when ``None``.
        hyperparameters: Optional model hyperparameter overrides.
        lasso_select: Whether to run LASSO feature selection per fold.
        lasso_C: LASSO inverse-regularization strength.
        min_train_matches: Minimum training matches required to score a fold.
        squad_values: Optional squad-value snapshots (A.2). When provided, the
            squad-value features are added via an as-of join (date-safe, so the
            backtest stays leakage-free). ``None`` runs without them — the A/B
            baseline.

    Returns:
        :class:`BacktestResult` with one row per scored tournament year.

    Raises:
        ValueError: If *matches* is empty.
    """
    if matches.empty:
        raise ValueError("No matches supplied to the tournament backtester")

    weights = blend_weights or BlendWeights()
    df = matches.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["year"] = df["date"].dt.year
    # The canonical table stores scorelines, not a derived outcome; a match is
    # "played" when both scores are present.
    played = df["score_a"].notna() & df["score_b"].notna()

    evaluator = Evaluator()
    records: list[dict[str, float]] = []
    for year in sorted(target_years):
        scored = _score_holdout(
            df,
            year,
            competition,
            weights,
            shrinker_factory,
            hyperparameters,
            lasso_select,
            lasso_C,
            squad_values,
            min_train_matches,
            played,
        )
        if scored is None:
            logger.warning("Skipping %d: insufficient history or no test matches", year)
            continue
        test_feats, proba, n_train = scored
        y_true = test_feats["outcome"].astype(str).tolist()
        report = evaluator.evaluate(y_true=y_true, proba=proba)
        record = {
            "year": int(year),
            "n_test": int(len(test_feats)),
            "n_train": int(n_train),
            "rps": _ranked_probability_score(y_true, proba),
            **{k: v for k, v in report.to_dict().items() if not k.startswith("quiniela_")},
        }
        logger.info(
            "Backtest %d: log_loss=%.4f brier=%.4f rps=%.4f acc=%.3f (n=%d)",
            year,
            record["log_loss"],
            record["brier"],
            record["rps"],
            record["accuracy"],
            record["n_test"],
        )
        records.append(record)

    per_fold = pd.DataFrame(records)
    if per_fold.empty:
        aggregate = pd.Series(dtype=float)
    else:
        numeric = per_fold.drop(columns=["year"]).select_dtypes("number")
        aggregate = numeric.mean()
    logger.info("Tournament backtest aggregate: %s", aggregate.to_dict())
    return BacktestResult(per_fold=per_fold, aggregate=aggregate)


def collect_holdout_predictions(
    matches: pd.DataFrame,
    year: int,
    competition: str = "WC",
    shrinker_factory: Optional[Callable[[], RatingShrinker]] = None,
    blend_weights: Optional[BlendWeights] = None,
    hyperparameters: Optional[dict] = None,
    lasso_select: bool = True,
    lasso_C: float = 0.1,
    squad_values: Optional[pd.DataFrame] = None,
    min_train_matches: int = 500,
) -> tuple[np.ndarray, list[str]]:
    """Out-of-sample ``(proba, y_true)`` for one held-out tournament.

    Trains the full stack on every match before *year* and predicts that
    tournament's played matches. Used by the calibration and backtest notebooks.

    Returns:
        ``(proba, y_true)`` with ``proba`` shape ``(n, 3)`` summing to 1 row-wise
        and ``y_true`` a list of ``"H"/"D"/"A"``. Empty arrays if the fold is
        skipped (no history or no test matches).

    Raises:
        ValueError: If *matches* is empty.
    """
    if matches.empty:
        raise ValueError("No matches supplied to collect_holdout_predictions")
    weights = blend_weights or BlendWeights()
    df = matches.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["year"] = df["date"].dt.year
    played = df["score_a"].notna() & df["score_b"].notna()
    scored = _score_holdout(
        df,
        year,
        competition,
        weights,
        shrinker_factory,
        hyperparameters,
        lasso_select,
        lasso_C,
        squad_values,
        min_train_matches,
        played,
    )
    if scored is None:
        return np.empty((0, 3)), []
    test_feats, proba, _ = scored
    return proba, test_feats["outcome"].astype(str).tolist()
