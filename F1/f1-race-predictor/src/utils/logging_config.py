"""Logging configuration module for F1 Race Predictor."""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: Optional[str] = None,
) -> None:
    """Configure project-wide logging with file and console handlers.

    Args:
        level: Logging level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Log filename. If None, only console logging is set up.
        log_dir: Directory for log files. Defaults to 'logs/'.

    Raises:
        ValueError: If level string is not a valid logging level.
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")

    fmt = "%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=date_fmt)

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        _add_file_handler(root_logger, log_file, log_dir, numeric_level, formatter)

    logging.getLogger(__name__).info(
        "Logging configured: level=%s, file=%s", level, log_file
    )


def _add_file_handler(
    logger: logging.Logger,
    log_file: str,
    log_dir: Optional[str],
    level: int,
    formatter: logging.Formatter,
) -> None:
    """Add a rotating file handler to the logger.

    Args:
        logger: Logger instance to add the handler to.
        log_file: Log filename.
        log_dir: Directory for log files.
        level: Numeric logging level.
        formatter: Formatter instance.
    """
    base_dir = Path(log_dir) if log_dir else Path("logs")
    base_dir.mkdir(parents=True, exist_ok=True)
    log_path = base_dir / log_file

    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_path,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Get a named logger instance.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        Configured Logger instance.
    """
    return logging.getLogger(name)
