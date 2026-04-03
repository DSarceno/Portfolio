"""Configuration manager for F1 Race Predictor."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class Config:
    """Singleton configuration manager.

    Loads and merges settings from YAML config files and .env variables.

    Example:
        config = Config()
        data_cfg = config.data_config
        start = config.get("data.start_season", default=2020)
    """

    _instance: Optional["Config"] = None
    _initialized: bool = False

    def __new__(cls, config_path: Optional[str] = None) -> "Config":
        """Create or return singleton instance.

        Args:
            config_path: Optional path to main config.yaml.

        Returns:
            Config singleton instance.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config_path: Optional[str] = None) -> None:
        """Initialize configuration from YAML and .env files.

        Args:
            config_path: Path to main config.yaml. Defaults to
                CONFIG_PATH env var or 'config/config.yaml'.
        """
        if self._initialized:
            return

        load_dotenv()

        self._config_path = Path(
            config_path
            or os.getenv("CONFIG_PATH", "config/config.yaml")
        )
        self._features_path = Path(
            os.getenv("FEATURES_CONFIG_PATH", "config/features.yaml")
        )
        self._model_params_path = Path(
            os.getenv("MODEL_PARAMS_PATH", "config/model_params.yaml")
        )

        self._config: Dict[str, Any] = {}
        self._features: Dict[str, Any] = {}
        self._model_params: Dict[str, Any] = {}

        self._load_all()
        Config._initialized = True
        logger.info("Config initialized from %s", self._config_path)

    def _load_all(self) -> None:
        """Load all configuration files."""
        self._config = self._load_yaml(self._config_path)
        self._features = self._load_yaml(self._features_path)
        self._model_params = self._load_yaml(self._model_params_path)

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """Load a YAML file into a dictionary.

        Args:
            path: Path to YAML file.

        Returns:
            Dictionary with YAML contents, or empty dict on error.
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            logger.debug("Loaded config from %s", path)
            return data
        except FileNotFoundError:
            logger.warning("Config file not found: %s", path)
            return {}
        except yaml.YAMLError as e:
            logger.error("Failed to parse YAML %s: %s", path, e)
            return {}

    @property
    def data_config(self) -> Dict[str, Any]:
        """Data section of main config.

        Returns:
            Dictionary with data configuration.
        """
        return self._config.get("data", {})

    @property
    def training_config(self) -> Dict[str, Any]:
        """Training section of main config.

        Returns:
            Dictionary with training configuration.
        """
        return self._config.get("training", {})

    @property
    def model_config(self) -> Dict[str, Any]:
        """Model parameters config.

        Returns:
            Dictionary with model parameters.
        """
        return self._model_params

    @property
    def features_config(self) -> Dict[str, Any]:
        """Features engineering config.

        Returns:
            Dictionary with feature configuration.
        """
        return self._features

    @property
    def logging_config(self) -> Dict[str, Any]:
        """Logging section of main config.

        Returns:
            Dictionary with logging configuration.
        """
        return self._config.get("logging", {})

    @property
    def paths_config(self) -> Dict[str, Any]:
        """Paths section of main config.

        Returns:
            Dictionary with path configuration.
        """
        return self._config.get("paths", {})

    def get(self, key: str, default: Any = None) -> Any:
        """Get a config value using dot-notation key.

        Args:
            key: Dot-separated key, e.g. 'data.start_season'.
            default: Value to return if key not found.

        Returns:
            Config value or default.
        """
        parts = key.split(".")
        value: Any = self._config
        try:
            for part in parts:
                value = value[part]
            return value
        except (KeyError, TypeError):
            return default

    @classmethod
    def reset(cls) -> None:
        """Reset singleton for testing purposes."""
        cls._instance = None
        cls._initialized = False
