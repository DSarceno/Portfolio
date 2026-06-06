"""JSON schema and parsing for LLM-produced phenomenon specifications.

The schema mirrors :class:`~simgen.simulations.spec.PhenomenonSpec`. It is sent
to the model (as part of the system prompt and as a tool input schema) and used
to validate the returned JSON before it is converted into a spec.
"""

from __future__ import annotations

import json
import re
from typing import Any

from simgen.simulations.spec import PhenomenonSpec, SpecValidationError
from simgen.utilities.logging_config import get_logger

logger = get_logger(__name__)

#: JSON Schema for a phenomenon specification (Anthropic tool-use compatible).
SPEC_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "name",
        "summary",
        "state_variables",
        "parameters",
        "derivatives",
        "initial_conditions",
    ],
    "properties": {
        "name": {"type": "string", "description": "Human-readable phenomenon name."},
        "summary": {"type": "string", "description": "One-paragraph plain-language summary."},
        "domain": {"type": "string", "enum": ["physics", "mathematics"]},
        "state_variables": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["symbol"],
                "properties": {
                    "symbol": {"type": "string", "description": "Valid Python identifier."},
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "unit": {"type": "string"},
                },
            },
        },
        "parameters": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["symbol", "value"],
                "properties": {
                    "symbol": {"type": "string"},
                    "value": {"type": "number"},
                    "description": {"type": "string"},
                    "unit": {"type": "string"},
                },
            },
        },
        "derivatives": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Python expressions for d(state_i)/dt, one per state variable, in "
                "the same order. May use state symbols, parameter symbols, t, np, "
                "and functions sin/cos/exp/sqrt/log etc."
            ),
        },
        "initial_conditions": {"type": "array", "items": {"type": "number"}},
        "t_start": {"type": "number"},
        "t_end": {"type": "number"},
        "num_points": {"type": "integer"},
        "method": {
            "type": "string",
            "enum": ["euler", "rk4", "rk45", "dopri5", "radau", "bdf", "lsoda"],
        },
        "equations_latex": {"type": "array", "items": {"type": "string"}},
        "plots": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["kind"],
                "properties": {
                    "kind": {"type": "string", "enum": ["time_series", "phase2d", "phase3d"]},
                    "title": {"type": "string"},
                    "x": {"type": "string"},
                    "y": {"type": ["string", "null"]},
                    "z": {"type": ["string", "null"]},
                },
            },
        },
        "animation": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["time_series", "phase2d", "phase3d"]},
                "title": {"type": "string"},
                "description": {"type": "string"},
                "x": {"type": "string"},
                "y": {"type": ["string", "null"]},
                "z": {"type": ["string", "null"]},
                "trail": {"type": "boolean"},
            },
        },
        "theory": {"type": "string"},
        "references": {"type": "array", "items": {"type": "string"}},
        "energy_expression": {"type": ["string", "null"]},
    },
}

_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _extract_json(text: str) -> str:
    """Extract a JSON object from raw model text.

    Handles fenced code blocks and leading/trailing prose by locating the first
    balanced ``{ ... }`` span.

    Raises
    ------
    SpecValidationError
        If no JSON object can be located.
    """
    match = _JSON_FENCE.search(text)
    if match:
        return match.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    raise SpecValidationError("No JSON object found in model output")


def spec_from_json(payload: str | dict[str, Any]) -> PhenomenonSpec:
    """Parse model output (text or dict) into a validated :class:`PhenomenonSpec`.

    Parameters
    ----------
    payload:
        Either a JSON string (possibly fenced/with prose) or an already-parsed
        dictionary (e.g. from a tool-use response).

    Returns
    -------
    PhenomenonSpec

    Raises
    ------
    SpecValidationError
        If the payload cannot be parsed or fails spec validation.
    """
    if isinstance(payload, dict):
        data = payload
    else:
        raw = _extract_json(payload)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SpecValidationError(f"Invalid JSON from model: {exc}") from exc

    logger.debug("Parsing spec from model output: name=%r", data.get("name"))
    return PhenomenonSpec.from_dict(data)
