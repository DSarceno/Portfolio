"""LLM-backed generation engine (Claude API).

The engine is optional: importing this subpackage never requires the
``anthropic`` package. Generation falls back to the offline reference library
when no API key or client is available.
"""

from simgen.llm.generator import SimulationGenerator, generate_from_prompt
from simgen.llm.schema import SPEC_JSON_SCHEMA, spec_from_json

__all__ = [
    "SimulationGenerator",
    "generate_from_prompt",
    "SPEC_JSON_SCHEMA",
    "spec_from_json",
]
