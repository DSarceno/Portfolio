"""Thin wrapper around the Anthropic Claude API.

The ``anthropic`` package is an optional dependency. :class:`ClaudeClient`
imports it lazily so that the rest of the package works without it; attempting
to actually call the API without the dependency or an API key raises a clear
:class:`LLMUnavailableError`.
"""

from __future__ import annotations

from simgen.config.settings import Settings, get_settings
from simgen.llm.prompts import SYSTEM_PROMPT, build_user_prompt
from simgen.llm.schema import SPEC_JSON_SCHEMA
from simgen.utilities.logging_config import get_logger

logger = get_logger(__name__)


class LLMUnavailableError(RuntimeError):
    """Raised when the Claude API cannot be used (missing dep, key, or error)."""


class ClaudeClient:
    """Minimal client that returns a phenomenon-spec dict from a prompt.

    Parameters
    ----------
    settings:
        Resolved settings; defaults to :func:`get_settings`.
    """

    _TOOL_NAME = "emit_specification"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client = None  # lazily constructed

    @property
    def available(self) -> bool:
        """Whether the client can plausibly make a call (dep + key present)."""
        if not self.settings.has_api_key:
            return False
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False
        return True

    def _ensure_client(self) -> object:
        """Construct (once) and return the underlying Anthropic client.

        Raises
        ------
        LLMUnavailableError
            If the dependency or API key is missing.
        """
        if self._client is not None:
            return self._client
        if not self.settings.has_api_key:
            raise LLMUnavailableError(
                "ANTHROPIC_API_KEY is not set. Configure it in your environment "
                "or .env file, or use the offline library."
            )
        try:
            import anthropic
        except ImportError as exc:
            raise LLMUnavailableError(
                "The 'anthropic' package is not installed. Install with "
                "'pip install \"simgen[llm]\"'."
            ) from exc
        self._client = anthropic.Anthropic(api_key=self.settings.anthropic_api_key)
        return self._client

    def generate_spec_dict(self, phenomenon: str) -> dict:
        """Request a phenomenon-spec dictionary from Claude.

        Uses tool-use so the model returns a structured object matching
        :data:`SPEC_JSON_SCHEMA`.

        Parameters
        ----------
        phenomenon:
            Natural-language description of the phenomenon.

        Returns
        -------
        dict
            The raw specification dictionary (unvalidated).

        Raises
        ------
        LLMUnavailableError
            If the API is unavailable or returns no usable tool output.
        """
        client = self._ensure_client()
        logger.info("Requesting spec from Claude model %s", self.settings.model)
        try:
            response = client.messages.create(  # type: ignore[attr-defined]
                model=self.settings.model,
                max_tokens=self.settings.max_tokens,
                system=SYSTEM_PROMPT,
                tools=[
                    {
                        "name": self._TOOL_NAME,
                        "description": "Emit the simulation specification as structured data.",
                        "input_schema": SPEC_JSON_SCHEMA,
                    }
                ],
                tool_choice={"type": "tool", "name": self._TOOL_NAME},
                messages=[{"role": "user", "content": build_user_prompt(phenomenon)}],
            )
        except Exception as exc:  # noqa: BLE001 - normalise SDK/network errors
            raise LLMUnavailableError(f"Claude API call failed: {exc}") from exc

        for block in getattr(response, "content", []):
            if getattr(block, "type", None) == "tool_use" and getattr(block, "name", "") == self._TOOL_NAME:
                return dict(block.input)  # type: ignore[arg-type]
        raise LLMUnavailableError("Claude response did not contain tool output")
