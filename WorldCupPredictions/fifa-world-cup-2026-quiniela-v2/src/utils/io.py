"""Filesystem helpers for raw, interim and processed artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def ensure_dir(path: str | Path) -> Path:
    """Create *path* (recursively) if missing and return it as a :class:`Path`.

    Args:
        path: Directory path to create.

    Returns:
        The created or existing :class:`Path`.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def timestamp() -> str:
    """Return a UTC timestamp suitable for filenames (``YYYYMMDDTHHMMSSZ``)."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def save_csv(df: pd.DataFrame, path: str | Path, **kwargs: Any) -> Path:
    """Save *df* to CSV, creating parent directories if needed.

    Args:
        df: DataFrame to serialize.
        path: Output path.
        **kwargs: Forwarded to :meth:`pandas.DataFrame.to_csv`.

    Returns:
        The output :class:`Path`.
    """
    p = Path(path)
    ensure_dir(p.parent)
    df.to_csv(p, index=kwargs.pop("index", False), **kwargs)
    logger.debug("Wrote CSV %s (%d rows)", p, len(df))
    return p


def load_csv(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Load *path* into a :class:`pandas.DataFrame`.

    Args:
        path: Input CSV path.
        **kwargs: Forwarded to :func:`pandas.read_csv`.

    Returns:
        Parsed DataFrame.

    Raises:
        FileNotFoundError: When the path does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p}")
    df = pd.read_csv(p, **kwargs)
    logger.debug("Read CSV %s (%d rows)", p, len(df))
    return df


def save_json(payload: dict[str, Any], path: str | Path, indent: int = 2) -> Path:
    """Serialize *payload* as JSON.

    Args:
        payload: Mapping to serialize.
        path: Output path.
        indent: JSON indentation.

    Returns:
        The output :class:`Path`.
    """
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=indent, default=str)
    return p


def load_json(path: str | Path) -> dict[str, Any]:
    """Deserialize a JSON file.

    Args:
        path: Path to a JSON file.

    Returns:
        Parsed payload.

    Raises:
        FileNotFoundError: When the path does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_pickle(obj: Any, path: str | Path) -> Path:
    """Persist *obj* to disk via :mod:`joblib`.

    Args:
        obj: Serializable Python object.
        path: Output path.

    Returns:
        The output :class:`Path`.
    """
    p = Path(path)
    ensure_dir(p.parent)
    joblib.dump(obj, p)
    logger.debug("Saved pickle %s", p)
    return p


def load_pickle(path: str | Path) -> Any:
    """Load a previously :func:`save_pickle`-d object.

    Args:
        path: Pickle path.

    Returns:
        The deserialized object.

    Raises:
        FileNotFoundError: When the path does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Pickle not found: {p}")
    return joblib.load(p)


def snapshot_path(base_dir: str | Path, source: str, ext: str = "csv") -> Path:
    """Build a timestamped raw-data snapshot path under ``<base_dir>/<source>/``.

    Args:
        base_dir: Top-level raw directory.
        source: Source identifier (e.g. ``"football_data"``).
        ext: File extension without leading dot.

    Returns:
        Path under ``<base_dir>/<source>/<source>_<timestamp>.<ext>``.
    """
    base = ensure_dir(Path(base_dir) / source)
    return base / f"{source}_{timestamp()}.{ext}"
