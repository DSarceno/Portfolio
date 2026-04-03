"""XGBoost model for F1 race position prediction."""

import logging
from typing import Dict, List, Optional

import numpy as np
from xgboost import XGBClassifier

from src.models.base_model import BaseModel

logger = logging.getLogger(__name__)


class XGBoostModel(BaseModel):
    """XGBoost multi-class classifier for predicting F1 race positions.

    Uses XGBClassifier with softmax output for positions 1-20.
    Supports early stopping when validation data is provided.
    """

    def __init__(
        self, config: Dict, params: Optional[Dict] = None
    ) -> None:
        """Initialize XGBoostModel.

        Args:
            config: Full model configuration dictionary.
            params: Optional hyperparameter overrides.
        """
        super().__init__("xgboost", config)
        self._feature_names: List[str] = []
        self._built_params = self._build_params(params)
        logger.info("XGBoostModel initialized with params: %s", self._built_params)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        """Train XGBoost classifier.

        Args:
            X_train: Training feature matrix.
            y_train: Training labels (1-indexed positions; converted to 0-indexed).
            X_val: Optional validation features for early stopping.
            y_val: Optional validation labels.
        """
        # Convert 1-indexed positions to 0-indexed classes
        y_train_0 = np.clip(y_train.astype(int) - 1, 0, 19)

        training_cfg = (
            self._config.get("models", {})
            .get("xgboost", {})
            .get("training", {})
        )
        early_stopping = int(training_cfg.get("early_stopping_rounds", 50))

        self._model = XGBClassifier(**self._built_params)

        fit_kwargs: Dict = {}
        if X_val is not None and y_val is not None:
            y_val_0 = np.clip(y_val.astype(int) - 1, 0, 19)
            fit_kwargs["eval_set"] = [(X_val, y_val_0)]
            fit_kwargs["early_stopping_rounds"] = early_stopping
            fit_kwargs["verbose"] = int(training_cfg.get("verbose", 100))

        try:
            self._model.fit(X_train, y_train_0, **fit_kwargs)
            self._is_trained = True
            logger.info(
                "XGBoostModel trained on %d samples", len(y_train)
            )
        except Exception as e:
            logger.error("XGBoost training failed: %s", e)
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict race positions (1-indexed).

        Args:
            X: Feature matrix.

        Returns:
            Integer array of positions 1-20.

        Raises:
            RuntimeError: If model not trained.
        """
        if not self._is_trained or self._model is None:
            raise RuntimeError("XGBoostModel must be trained before predict()")
        proba = self._model.predict_proba(X)
        classes_0 = np.argmax(proba, axis=1)
        return (classes_0 + 1).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability matrix.

        Args:
            X: Feature matrix.

        Returns:
            Probability matrix of shape (n_samples, 20).
        """
        if not self._is_trained or self._model is None:
            raise RuntimeError("XGBoostModel must be trained before predict_proba()")
        return self._model.predict_proba(X)

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Return XGBoost feature importances.

        Returns:
            Dict of feature_name -> importance, or None if not trained.
        """
        if not self._is_trained or self._model is None:
            return None
        importances = self._model.feature_importances_
        if self._feature_names and len(self._feature_names) == len(importances):
            return dict(zip(self._feature_names, importances.tolist()))
        return {f"f{i}": float(v) for i, v in enumerate(importances)}

    def set_feature_names(self, names: List[str]) -> None:
        """Store feature names for importance reporting.

        Args:
            names: List of feature column names.
        """
        self._feature_names = names

    def _build_params(self, overrides: Optional[Dict]) -> Dict:
        """Merge config hyperparameters with optional overrides.

        Args:
            overrides: Optional dict of param overrides.

        Returns:
            Final hyperparameter dictionary.
        """
        defaults = (
            self._config.get("models", {})
            .get("xgboost", {})
            .get("hyperparameters", {})
        )
        params = {
            "objective": "multi:softprob",
            "num_class": 20,
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 200,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
            "eval_metric": "mlogloss",
            "use_label_encoder": False,
        }
        params.update(defaults)
        if overrides:
            params.update(overrides)
        # Remove non-XGBClassifier keys
        params.pop("batch_size", None)
        return params
