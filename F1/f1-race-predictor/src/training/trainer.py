"""Training orchestrator for F1 race prediction models."""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.models.base_model import BaseModel
from src.models.model_factory import ModelFactory

logger = logging.getLogger(__name__)


class Trainer:
    """Orchestrates training and saving of F1 prediction models.

    Handles single model training, multi-model comparison,
    and model persistence.
    """

    def __init__(
        self,
        config: Dict,
        model_factory: Optional[ModelFactory] = None,
    ) -> None:
        """Initialize Trainer.

        Args:
            config: Full project configuration dictionary.
            model_factory: ModelFactory instance. Uses default if None.
        """
        self.config = config
        self.factory = model_factory or ModelFactory()
        paths_cfg = config.get("paths", {})
        self._models_dir = Path(paths_cfg.get("models_dir", "models"))
        self._models_dir.mkdir(parents=True, exist_ok=True)
        self._trained_models: Dict[str, BaseModel] = {}
        logger.info("Trainer initialized, models dir: %s", self._models_dir)

    def train_single_model(
        self,
        model: BaseModel,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, float]:
        """Train one model and evaluate on validation set.

        Args:
            model: Untrained BaseModel instance.
            X_train: Training features.
            y_train: Training labels.
            X_val: Validation features.
            y_val: Validation labels.

        Returns:
            Dictionary of evaluation metrics on validation set.
        """
        logger.info("Training model: %s", model.name)
        try:
            model.train(X_train, y_train, X_val, y_val)
            metrics = model.evaluate(X_val, y_val)
            self._log_training_results(model.name, metrics)
            self._trained_models[model.name] = model
            return metrics
        except Exception as e:
            logger.error("Training failed for %s: %s", model.name, e)
            raise

    def train_all_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        """Train all enabled models from config.

        Args:
            X_train: Training features.
            y_train: Training labels.
            X_val: Validation features.
            y_val: Validation labels.

        Returns:
            Dict of model_name -> metrics dict.
        """
        models = ModelFactory.create_all(self.config)
        results: Dict[str, Dict[str, float]] = {}
        for name, model in models.items():
            try:
                metrics = self.train_single_model(
                    model, X_train, y_train, X_val, y_val
                )
                results[name] = metrics
            except Exception as e:
                logger.warning("Skipping model %s due to error: %s", name, e)
                results[name] = {}
        logger.info("Trained %d models", len(results))
        return results

    def save_model(self, model: BaseModel, model_name: str) -> Path:
        """Save a trained model to disk.

        Args:
            model: Trained model instance.
            model_name: Name used for the file.

        Returns:
            Path where the model was saved.
        """
        save_path = self._models_dir / f"{model_name}.pkl"
        model.save(str(save_path))
        logger.info("Saved model '%s' to %s", model_name, save_path)
        return save_path

    def load_model(self, model_name: str, model_type: str) -> BaseModel:
        """Load a previously saved model.

        Args:
            model_name: File stem of the saved model.
            model_type: Model type string for factory creation.

        Returns:
            Loaded BaseModel instance.

        Raises:
            FileNotFoundError: If model file does not exist.
        """
        load_path = self._models_dir / f"{model_name}.pkl"
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")
        model = BaseModel.load(str(load_path))
        logger.info("Loaded model '%s' from %s", model_name, load_path)
        return model

    def get_best_model(
        self, results: Dict[str, Dict[str, float]]
    ) -> Tuple[str, BaseModel]:
        """Select the best model by top_3_accuracy.

        Args:
            results: Dict of model_name -> metrics dict.

        Returns:
            Tuple of (best_model_name, best_model_instance).

        Raises:
            ValueError: If no trained models are available.
        """
        if not self._trained_models:
            raise ValueError("No trained models available")

        best_name = max(
            results,
            key=lambda n: results[n].get("top_3_accuracy", 0.0),
        )
        best_model = self._trained_models[best_name]
        logger.info(
            "Best model: %s (top3=%.3f)",
            best_name,
            results[best_name].get("top_3_accuracy", 0.0),
        )
        return best_name, best_model

    def _log_training_results(
        self, model_name: str, metrics: Dict[str, float]
    ) -> None:
        """Log evaluation metrics for a trained model.

        Args:
            model_name: Name of the model.
            metrics: Dictionary of metric name -> value.
        """
        logger.info(
            "Results for '%s': MAE=%.3f | RMSE=%.3f | "
            "Top3=%.3f | Top5=%.3f | Exact=%.3f",
            model_name,
            metrics.get("mae", float("nan")),
            metrics.get("rmse", float("nan")),
            metrics.get("top_3_accuracy", float("nan")),
            metrics.get("top_5_accuracy", float("nan")),
            metrics.get("exact_accuracy", float("nan")),
        )
