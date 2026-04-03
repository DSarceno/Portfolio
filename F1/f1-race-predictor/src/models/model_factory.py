"""Factory pattern for creating F1 prediction model instances."""

import logging
from typing import Dict, List, Type

from src.models.base_model import BaseModel
from src.models.baseline import BaselineModel
from src.models.neural_net import NeuralNetModel
from src.models.xgboos_model import XGBoostModel

logger = logging.getLogger(__name__)


class ModelFactory:
    """Creates and manages F1 model instances.

    Supports XGBoost, Neural Network, and Baseline model types.
    Uses a registry pattern for extensibility.
    """

    MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {
        "xgboost": XGBoostModel,
        "neural_net": NeuralNetModel,
        "baseline": BaselineModel,
    }

    @classmethod
    def create(cls, model_type: str, config: Dict) -> BaseModel:
        """Instantiate a model by type name.

        Args:
            model_type: One of 'xgboost', 'neural_net', 'baseline'.
            config: Full model configuration dictionary.

        Returns:
            Instantiated BaseModel subclass.

        Raises:
            ValueError: If model_type is not registered.
        """
        if model_type not in cls.MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model type '{model_type}'. "
                f"Available: {list(cls.MODEL_REGISTRY.keys())}"
            )
        model_cls = cls.MODEL_REGISTRY[model_type]
        model = model_cls(config)
        logger.info("Created model: %s", model_type)
        return model

    @classmethod
    def create_all(cls, config: Dict) -> Dict[str, BaseModel]:
        """Create all enabled models defined in config.

        Args:
            config: Full model configuration dictionary.

        Returns:
            Dict mapping model_type -> BaseModel instance.
        """
        models: Dict[str, BaseModel] = {}
        models_cfg = config.get("models", {})
        for model_type in cls.MODEL_REGISTRY:
            type_cfg = models_cfg.get(model_type, {})
            if type_cfg.get("enabled", True):
                models[model_type] = cls.create(model_type, config)
        logger.info("Created %d models: %s", len(models), list(models.keys()))
        return models

    @classmethod
    def get_available_models(cls) -> List[str]:
        """Return list of registered model type names.

        Returns:
            List of model type strings.
        """
        return list(cls.MODEL_REGISTRY.keys())
