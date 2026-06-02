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
