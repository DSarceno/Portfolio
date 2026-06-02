"""Project configuration loader."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class Config:
    """Container that exposes the merged YAML configurations as attribute trees."""

    main: dict[str, Any] = field(default_factory=dict)
    features: dict[str, Any] = field(default_factory=dict)
    model_params: dict[str, Any] = field(default_factory=dict)
    strategy: dict[str, Any] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)

    def get(self, dotted: str, default: Any = None) -> Any:
        """Read a dotted-path value from the main configuration.

        Args:
            dotted: Dot-separated key path, e.g. ``"tournament.year"``.
            default: Value returned when any key in the path is missing.

        Returns:
            The resolved value or ``default``.
        """
        node: Any = self.main
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        logger.warning("Configuration file %s does not exist, returning empty dict", path)
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_config(
    main_path: Optional[str] = None,
    features_path: Optional[str] = None,
    model_params_path: Optional[str] = None,
    strategy_path: Optional[str] = None,
    env_path: Optional[str] = None,
) -> Config:
    """Load YAML configurations and ``.env`` variables into a :class:`Config`.

    Args:
        main_path: Override path to ``config/config.yaml``.
        features_path: Override path to ``config/features.yaml``.
        model_params_path: Override path to ``config/model_params.yaml``.
        strategy_path: Override path to ``config/strategy.yaml``.
        env_path: Optional path to a ``.env`` file.

    Returns:
        A populated :class:`Config` object.
    """
    if env_path is None:
        env_path = ".env"
    if Path(env_path).exists():
        load_dotenv(env_path)

    main_path = main_path or os.getenv("CONFIG_PATH", "config/config.yaml")
    features_path = features_path or os.getenv("FEATURES_CONFIG_PATH", "config/features.yaml")
    model_params_path = model_params_path or os.getenv(
        "MODEL_PARAMS_PATH", "config/model_params.yaml"
    )
    strategy_path = strategy_path or os.getenv("STRATEGY_CONFIG_PATH", "config/strategy.yaml")

    config = Config(
        main=_read_yaml(Path(main_path)),
        features=_read_yaml(Path(features_path)),
        model_params=_read_yaml(Path(model_params_path)),
        strategy=_read_yaml(Path(strategy_path)),
        env={k: v for k, v in os.environ.items()},
    )
    logger.info("Loaded configuration from %s", main_path)
    return config
