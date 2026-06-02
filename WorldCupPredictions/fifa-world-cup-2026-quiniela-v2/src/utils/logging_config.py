"""Centralized logging configuration."""

from __future__ import annotations

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional

_DEFAULT_FORMAT = "%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s"


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    fmt: str = _DEFAULT_FORMAT,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> None:
    """Configure root logging with a console handler and an optional rotating file handler.

    Args:
        level: Logging level name (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional path to a log file. Parent directories are created if missing.
        fmt: Log line format string.
        max_bytes: Rotation threshold in bytes for the file handler.
        backup_count: Number of rotated log files to retain.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(numeric_level)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(fmt)

    console = logging.StreamHandler()
    console.setLevel(numeric_level)
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger, ensuring root logging is configured.

    Args:
        name: Logger name, usually ``__name__``.

    Returns:
        Configured :class:`logging.Logger`.
    """
    if not logging.getLogger().handlers:
        setup_logging(
            level=os.getenv("LOG_LEVEL", "INFO"),
            log_file=os.getenv("LOG_FILE", "logs/world_cup_quiniela_v2.log"),
        )
    return logging.getLogger(name)
