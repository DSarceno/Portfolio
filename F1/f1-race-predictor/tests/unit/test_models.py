"""Unit tests for prediction models."""

import numpy as np
import pytest

from src.models.baseline import BaselineModel
from src.models.model_factory import ModelFactory
from src.models.xgboos_model import XGBoostModel


def _make_xy(
    n_samples: int = 100, n_features: int = 10
) -> tuple:
    """Create synthetic feature matrix and target labels.

    Args:
        n_samples: Number of samples.
        n_features: Number of features.

    Returns:
        Tuple of (X, y) numpy arrays.
    """
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n_samples, n_features))
    y = rng.integers(1, 21, size=n_samples)
    return X, y


class TestBaselineModel:
    """Tests for BaselineModel."""

    def test_predict_returns_valid_positions(
        self, mock_config: dict
    ) -> None:
        """predict should return positions in [1, 20].

        Args:
            mock_config: Configuration fixture.
        """
        model = BaselineModel(mock_config)
        X, y = _make_xy()
        model.train(X, y)
        preds = model.predict(X)
        assert preds.min() >= 1
        assert preds.max() <= 20

    def test_predict_positions_integer_type(
        self, mock_config: dict
    ) -> None:
        """predict should return integer array.

        Args:
            mock_config: Configuration fixture.
        """
        model = BaselineModel(mock_config)
        X, y = _make_xy()
        model.train(X, y)
        preds = model.predict(X)
        assert preds.dtype in [np.int32, np.int64, int]

    def test_untrained_model_raises_error(
        self, mock_config: dict
    ) -> None:
        """predict before train should raise RuntimeError.

        Args:
            mock_config: Configuration fixture.
        """
        model = BaselineModel(mock_config)
        X, _ = _make_xy()
        with pytest.raises(RuntimeError):
            model.predict(X)

    def test_get_feature_importance_returns_none(
        self, mock_config: dict
    ) -> None:
        """get_feature_importance should return None for baseline.

        Args:
            mock_config: Configuration fixture.
        """
        model = BaselineModel(mock_config)
        assert model.get_feature_importance() is None


class TestXGBoostModel:
    """Tests for XGBoostModel."""

    def test_train_with_valid_data(self, mock_config: dict) -> None:
        """train should fit without error on valid data.

        Args:
            mock_config: Configuration fixture.
        """
        model = XGBoostModel(mock_config)
        X, y = _make_xy(50, 8)
        model.train(X, y)
        assert model.is_trained

    def test_predict_returns_integer_positions(
        self, mock_config: dict
    ) -> None:
        """predict should return integer array in [1, 20].

        Args:
            mock_config: Configuration fixture.
        """
        model = XGBoostModel(mock_config)
        X, y = _make_xy(50, 8)
        model.train(X, y)
        preds = model.predict(X)
        assert preds.min() >= 1
        assert preds.max() <= 20

    def test_predict_proba_sums_to_one(self, mock_config: dict) -> None:
        """predict_proba rows should sum to ~1.0.

        Args:
            mock_config: Configuration fixture.
        """
        model = XGBoostModel(mock_config)
        X, y = _make_xy(50, 8)
        model.train(X, y)
        proba = model.predict_proba(X)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_feature_importance_not_none_after_training(
        self, mock_config: dict
    ) -> None:
        """get_feature_importance should return dict after training.

        Args:
            mock_config: Configuration fixture.
        """
        model = XGBoostModel(mock_config)
        X, y = _make_xy(50, 8)
        model.train(X, y)
        importance = model.get_feature_importance()
        assert importance is not None
        assert len(importance) > 0


class TestModelFactory:
    """Tests for ModelFactory."""

    def test_create_xgboost_model(self, mock_config: dict) -> None:
        """create should return XGBoostModel for 'xgboost'.

        Args:
            mock_config: Configuration fixture.
        """
        model = ModelFactory.create("xgboost", mock_config)
        assert model.name == "xgboost"

    def test_create_baseline_model(self, mock_config: dict) -> None:
        """create should return BaselineModel for 'baseline'.

        Args:
            mock_config: Configuration fixture.
        """
        model = ModelFactory.create("baseline", mock_config)
        assert model.name == "baseline"

    def test_create_neural_net_model(self, mock_config: dict) -> None:
        """create should return NeuralNetModel for 'neural_net'.

        Args:
            mock_config: Configuration fixture.
        """
        model = ModelFactory.create("neural_net", mock_config)
        assert model.name == "neural_net"

    def test_invalid_model_type_raises_valueerror(
        self, mock_config: dict
    ) -> None:
        """create should raise ValueError for unknown model type.

        Args:
            mock_config: Configuration fixture.
        """
        with pytest.raises(ValueError, match="Unknown model type"):
            ModelFactory.create("unknown_model", mock_config)

    def test_get_available_models_returns_list(self) -> None:
        """get_available_models should return a non-empty list."""
        models = ModelFactory.get_available_models()
        assert isinstance(models, list)
        assert len(models) > 0
        assert "xgboost" in models
