"""Filesystem and serialisation helpers used across the package."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from simgen.utilities.logging_config import get_logger

logger = get_logger(__name__)

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(text: str, *, max_length: int = 60) -> str:
    """Convert arbitrary text into a filesystem- and URL-safe slug.

    Parameters
    ----------
    text:
        Input text (e.g. a phenomenon name).
    max_length:
        Maximum length of the resulting slug.

    Returns
    -------
    str
        A lowercase, hyphen-separated slug. Returns ``"phenomenon"`` if the
        input contains no alphanumeric characters.
    """
    lowered = text.strip().lower()
    slug = _SLUG_RE.sub("-", lowered).strip("-")
    slug = slug[:max_length].strip("-")
    return slug or "phenomenon"


def ensure_directory(path: str | Path) -> Path:
    """Create ``path`` (and parents) if needed and return it as a :class:`Path`.

    Parameters
    ----------
    path:
        Directory path to create.

    Returns
    -------
    pathlib.Path
        The created (or pre-existing) directory.
    """
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_text(path: str | Path, content: str) -> Path:
    """Write UTF-8 text to ``path``, creating parent directories as needed."""
    target = Path(path)
    ensure_directory(target.parent)
    target.write_text(content, encoding="utf-8")
    logger.debug("Wrote %d characters to %s", len(content), target)
    return target


def write_json(path: str | Path, data: Any, *, indent: int = 2) -> Path:
    """Serialise ``data`` to JSON at ``path``.

    NumPy scalars and arrays are converted to native Python types so that the
    output is portable.
    """
    target = Path(path)
    ensure_directory(target.parent)
    target.write_text(json.dumps(data, indent=indent, default=_json_default), encoding="utf-8")
    return target


def read_json(path: str | Path) -> Any:
    """Load JSON from ``path``.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    json.JSONDecodeError
        If the file is not valid JSON.
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_array(path: str | Path, array: np.ndarray) -> Path:
    """Save a NumPy array to ``path`` in ``.npy`` format."""
    target = Path(path)
    ensure_directory(target.parent)
    np.save(target, array)
    return target


def _json_default(obj: Any) -> Any:
    """JSON serialisation fallback for NumPy types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")
