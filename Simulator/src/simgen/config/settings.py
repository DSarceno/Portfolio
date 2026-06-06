"""Centralised configuration for :mod:`simgen`.

Settings are resolved from (in order of precedence):

1. Explicit keyword arguments passed to :class:`Settings`.
2. Environment variables (optionally loaded from a ``.env`` file).
3. Built-in defaults.

The module performs a minimal, dependency-free ``.env`` parse so that the
package does not require ``python-dotenv`` for basic operation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from simgen.utilities.logging_config import get_logger

logger = get_logger(__name__)


def _load_dotenv(path: Path) -> dict[str, str]:
    """Parse a simple ``KEY=VALUE`` ``.env`` file.

    Lines that are blank or start with ``#`` are ignored. Surrounding quotes
    around values are stripped. The function never raises on malformed lines;
    it logs and skips them instead.

    Parameters
    ----------
    path:
        Path to the ``.env`` file.

    Returns
    -------
    dict[str, str]
        Mapping of parsed environment variables (empty if the file is absent).
    """
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                values[key] = value
    except OSError as exc:  # pragma: no cover - filesystem dependent
        logger.warning("Could not read .env file at %s: %s", path, exc)
    return values


def _project_root() -> Path:
    """Return the repository root (three levels above this file)."""
    return Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Settings:
    """Resolved runtime configuration.

    Attributes
    ----------
    anthropic_api_key:
        API key for the Claude API. Empty string when not configured.
    model:
        Claude model identifier used for generation.
    max_tokens:
        Maximum number of tokens requested from the model.
    output_dir:
        Directory into which generated projects are written.
    project_root:
        Absolute path to the repository root.
    """

    anthropic_api_key: str = ""
    model: str = "claude-opus-4-8"
    max_tokens: int = 8000
    output_dir: Path = field(default_factory=lambda: Path("generated"))
    project_root: Path = field(default_factory=_project_root)

    @property
    def has_api_key(self) -> bool:
        """Whether an Anthropic API key is configured."""
        return bool(self.anthropic_api_key)

    @classmethod
    def from_environment(cls, **overrides: object) -> Settings:
        """Build :class:`Settings` from the environment and a ``.env`` file.

        Parameters
        ----------
        **overrides:
            Explicit values that take precedence over the environment.

        Returns
        -------
        Settings
            A fully resolved, immutable settings object.
        """
        root = _project_root()
        dotenv = _load_dotenv(root / ".env")

        def resolve(key: str, default: str) -> str:
            return os.environ.get(key, dotenv.get(key, default))

        api_key = resolve("ANTHROPIC_API_KEY", "")
        model = resolve("SIMGEN_MODEL", "claude-opus-4-8")
        max_tokens_raw = resolve("SIMGEN_MAX_TOKENS", "8000")
        output_dir_raw = resolve("SIMGEN_OUTPUT_DIR", "generated")

        try:
            max_tokens = int(max_tokens_raw)
        except ValueError:
            logger.warning("Invalid SIMGEN_MAX_TOKENS=%r; using 8000", max_tokens_raw)
            max_tokens = 8000

        resolved: dict[str, object] = {
            "anthropic_api_key": api_key,
            "model": model,
            "max_tokens": max_tokens,
            "output_dir": Path(output_dir_raw),
            "project_root": root,
        }
        resolved.update(overrides)
        return cls(**resolved)  # type: ignore[arg-type]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached, process-wide :class:`Settings` instance."""
    return Settings.from_environment()
