"""Helpers for assembling LaTeX reports from simulation output.

The actual document layout lives in the Jinja2 template
``codegen/templates/report.tex.j2``; this module provides the small, testable
pieces used to populate it: LaTeX escaping and a numerical-summary table.
"""

from __future__ import annotations

import numpy as np

from simgen.simulations.base import SimulationResult

#: Characters that must be escaped in LaTeX text mode.
_LATEX_SPECIAL = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "<": r"\textless{}",
    ">": r"\textgreater{}",
}


def escape_latex(text: str) -> str:
    """Escape characters that are special in LaTeX text mode.

    Parameters
    ----------
    text:
        Arbitrary text destined for a LaTeX document body.

    Returns
    -------
    str
        The escaped text, safe to embed.
    """
    # Process the backslash first to avoid double-escaping the replacements.
    out = text.replace("\\", _LATEX_SPECIAL["\\"])
    for char, replacement in _LATEX_SPECIAL.items():
        if char == "\\":
            continue
        out = out.replace(char, replacement)
    return out


def summarize_result(result: SimulationResult) -> list[dict[str, float | str]]:
    """Compute per-variable summary statistics for a LaTeX table.

    Parameters
    ----------
    result:
        A completed simulation result.

    Returns
    -------
    list[dict]
        One row per state variable with ``symbol``, ``min``, ``max``,
        ``mean`` and ``final`` values.
    """
    rows: list[dict[str, float | str]] = []
    for i, symbol in enumerate(result.symbols):
        column = result.states[:, i]
        rows.append(
            {
                "symbol": symbol,
                "min": float(np.min(column)),
                "max": float(np.max(column)),
                "mean": float(np.mean(column)),
                "final": float(column[-1]),
            }
        )
    return rows
