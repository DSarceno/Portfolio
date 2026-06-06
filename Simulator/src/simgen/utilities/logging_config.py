"""Project-wide logging configuration.

All modules obtain their logger via :func:`get_logger`. Production code never
uses ``print``; user-facing CLI output is handled separately in
:mod:`simgen.cli`. The default handler writes to ``stderr`` so that it does not
interfere with machine-readable ``stdout`` output.
"""

from __future__ import annotations

import logging
import os
import sys

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_CONFIGURED = False


def configure_logging(level: int | str | None = None, *, force: bool = False) -> None:
    """Configure the root logger for the package.

    Parameters
    ----------
    level:
        Logging level as an ``int`` or name (e.g. ``"INFO"``). If ``None``,
        the ``SIMGEN_LOG_LEVEL`` environment variable is consulted, defaulting
        to ``WARNING``.
    force:
        When ``True``, reconfigure even if logging was already configured.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    if level is None:
        level = os.environ.get("SIMGEN_LOG_LEVEL", "WARNING")
    if isinstance(level, str):
        level = logging.getLevelName(level.upper())
        if not isinstance(level, int):
            level = logging.WARNING

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))

    root = logging.getLogger("simgen")
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger, configuring logging on first use.

    Parameters
    ----------
    name:
        Usually ``__name__`` of the calling module.

    Returns
    -------
    logging.Logger
        A logger under the ``simgen`` namespace.
    """
    if not _CONFIGURED:
        configure_logging()
    # Normalise to the package namespace so the package-level level applies.
    if not name.startswith("simgen"):
        name = f"simgen.{name}"
    return logging.getLogger(name)
