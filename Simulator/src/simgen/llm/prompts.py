"""Prompt construction for LLM-backed specification generation."""

from __future__ import annotations

SYSTEM_PROMPT = """\
You are a computational physicist and applied mathematician. Your job is to turn
a short natural-language description of a mathematical or physical phenomenon
into a precise, machine-readable simulation specification.

You MUST model the phenomenon as a first-order system of ordinary differential
equations (ODEs). Convert any higher-order equation into a first-order system by
introducing velocity/auxiliary state variables. For spatial PDEs, apply a
method-of-lines discretisation and expose the resulting ODE system.

Rules for the specification:
- Every state-variable and parameter `symbol` MUST be a valid Python identifier
  (letters, digits, underscore; not starting with a digit) and the sets of
  state symbols and parameter symbols MUST be disjoint.
- `derivatives` is a list of Python expression strings, one per state variable,
  in the SAME order as `state_variables`. Each expression computes d(symbol)/dt.
- Expressions may reference: the state symbols, the parameter symbols, the
  independent variable `t`, `np` (NumPy), and the bare functions sin, cos, tan,
  exp, log, sqrt, sinh, cosh, tanh, abs, sign, atan2, power, and constants pi/e.
- Use `**` for exponentiation (NOT `^`).
- Provide sensible default numeric parameter values and initial conditions that
  produce interesting, numerically stable dynamics.
- Choose `method`: use "rk4" for smooth non-stiff systems, "rk45"/"lsoda" for
  adaptive control, "radau"/"bdf" for stiff systems.
- `equations_latex` should contain valid LaTeX math (no surrounding $).
- Provide 1-3 `plots` and one `animation`. Use "phase3d" only for >=3D systems.
- Keep `num_points` between 1000 and 12000.

Respond with ONLY a single JSON object that conforms to the provided schema.
Do not include any prose outside the JSON.
"""


def build_user_prompt(phenomenon: str) -> str:
    """Construct the user-turn prompt for a given phenomenon description.

    Parameters
    ----------
    phenomenon:
        The user's natural-language description.

    Returns
    -------
    str
        The fully assembled user prompt.
    """
    return (
        "Produce a simulation specification (single JSON object) for the "
        "following phenomenon:\n\n"
        f"PHENOMENON: {phenomenon.strip()}\n\n"
        "Return only the JSON object."
    )
