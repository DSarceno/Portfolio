"""Tests for the model layer."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.build_features import build_match_feature_matrix, select_feature_columns
from src.models.calibration import CalibrationStrategy, ProbabilityCalibrator
from src.models.multinomial_model import MultinomialOutcomeModel
from src.models.poisson_model import PoissonScoreModel
from src.models.xgboost_model import XGBoostOutcomeModel
from src.ratings.rating_ensemble import RatingEnsemble


def _prepare_features(matches: pd.DataFrame) -> pd.DataFrame:
    composite = RatingEnsemble().fit(matches).composite_table()
    matrix = build_match_feature_matrix(matches, pd.DataFrame(), composite)
    return matrix.dropna(subset=["outcome"]).copy()


def test_multinomial_probabilities_sum_to_one(synthetic_matches: pd.DataFrame) -> None:
    matrix = _prepare_features(synthetic_matches)
    cols = select_feature_columns(matrix)
    model = MultinomialOutcomeModel(feature_columns=cols).fit(matrix[cols], matrix["outcome"])
    proba = model.predict_proba(matrix[cols])
    assert proba.shape[1] == 3
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_multinomial_drops_near_constant_training_feature() -> None:
    """A feature constant in training must not explode at inference.

    Regression for the live knockout-stage failure: ``stage_knockout`` is
    identically 0 across the (historical) training data but becomes 1 for every
    2026 knockout fixture. StandardScaler then maps that 1 to a ~100-SD value
    that dominates the linear logit and saturates the softmax to a single class
    for *every* knockout match. A standardized linear model cannot learn a
    feature with no training variance, so it must be dropped at fit time.
    """
    rng = np.random.default_rng(0)
    signal = rng.normal(size=400)
    outcome = np.where(signal > 0.4, "H", np.where(signal < -0.4, "A", "D"))
    train = pd.DataFrame({"signal": signal, "stage_knockout": 0.0})

    cols = ["signal", "stage_knockout"]
    model = MultinomialOutcomeModel(feature_columns=cols).fit(train, pd.Series(outcome))

    # The constant feature carries no learnable signal and must be excluded.
    assert "stage_knockout" not in model.feature_columns
    assert "signal" in model.feature_columns

    # At inference every fixture is a knockout (stage_knockout=1); predictions
    # must still track the real signal instead of collapsing to one class.
    test = pd.DataFrame({"signal": np.linspace(-2.0, 2.0, 21), "stage_knockout": 1.0})
    proba = model.predict_proba(test)
    predicted = proba.argmax(axis=1)
    assert len(set(predicted.tolist())) >= 2, "predictions saturated to a single class"
    assert proba.max(axis=1).mean() < 0.95, "predictions pathologically saturated"


def test_xgboost_probabilities_sum_to_one(synthetic_matches: pd.DataFrame) -> None:
    matrix = _prepare_features(synthetic_matches)
    cols = select_feature_columns(matrix)
    model = XGBoostOutcomeModel(feature_columns=cols).fit(matrix[cols], matrix["outcome"])
    proba = model.predict_proba(matrix[cols])
    assert proba.shape[1] == 3
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_poisson_model_outputs(synthetic_matches: pd.DataFrame) -> None:
    poisson = PoissonScoreModel().fit(synthetic_matches, synthetic_matches.get("outcome", pd.Series([], dtype=str)))
    proba = poisson.outcome_probabilities("Alpha", "Beta")
    assert abs(proba.sum() - 1.0) < 1e-6
    matrix = poisson.score_matrix("Alpha", "Beta")
    assert matrix.shape == (poisson.max_goals + 1, poisson.max_goals + 1)
    assert matrix.sum() > 0


def test_calibrator_transform(synthetic_matches: pd.DataFrame) -> None:
    matrix = _prepare_features(synthetic_matches)
    cols = select_feature_columns(matrix)
    model = MultinomialOutcomeModel(feature_columns=cols).fit(matrix[cols], matrix["outcome"])
    proba = model.predict_proba(matrix[cols])
    cal = ProbabilityCalibrator(strategy=CalibrationStrategy.ISOTONIC).fit(proba, matrix["outcome"])
    calibrated = cal.transform(proba)
    np.testing.assert_allclose(calibrated.sum(axis=1), 1.0, atol=1e-6)
