"""High-level orchestration: prompt -> spec -> generated project.

:class:`SimulationGenerator` ties together the Claude client, the schema parser,
the offline reference library (fallback) and the code generator.
"""

from __future__ import annotations

from pathlib import Path

from simgen.codegen.renderer import GeneratedProject, generate_project
from simgen.config.settings import Settings, get_settings
from simgen.llm.client import ClaudeClient, LLMUnavailableError
from simgen.llm.schema import spec_from_json
from simgen.simulations import library
from simgen.simulations.spec import PhenomenonSpec
from simgen.utilities.logging_config import get_logger

logger = get_logger(__name__)


def _match_library(prompt: str) -> PhenomenonSpec | None:
    """Heuristically match a prompt to a reference spec by keyword overlap."""
    text = prompt.lower()
    keywords = {
        "lorenz-system": ("lorenz", "attractor", "convection", "butterfly"),
        "double-pendulum": ("double pendulum", "pendula", "coupled pendulum"),
        "simple-harmonic-oscillator": ("harmonic", "spring", "mass on a spring", "sho"),
        "duffing-oscillator": ("duffing", "cubic", "double well", "forced oscillator"),
        "van-der-pol-oscillator": ("van der pol", "vanderpol", "limit cycle", "relaxation"),
    }
    best_slug: str | None = None
    best_score = 0
    for slug, words in keywords.items():
        score = sum(1 for w in words if w in text)
        if score > best_score:
            best_score, best_slug = score, slug
    if best_slug is None:
        return None
    logger.info("Matched prompt to library spec %r (score=%d)", best_slug, best_score)
    return library.get_spec(best_slug)


class SimulationGenerator:
    """Generate phenomenon specs and projects from natural-language prompts.

    Parameters
    ----------
    settings:
        Resolved settings; defaults to :func:`get_settings`.
    client:
        Optional pre-built :class:`ClaudeClient` (useful for testing).
    allow_fallback:
        When ``True`` (default), fall back to the offline reference library if
        the LLM is unavailable. When ``False``, raise instead.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        client: ClaudeClient | None = None,
        *,
        allow_fallback: bool = True,
    ) -> None:
        self.settings = settings or get_settings()
        self.client = client or ClaudeClient(self.settings)
        self.allow_fallback = allow_fallback

    def spec_from_prompt(self, prompt: str) -> PhenomenonSpec:
        """Produce a validated :class:`PhenomenonSpec` for ``prompt``.

        Tries the Claude API first; on any failure falls back (if allowed) to
        the closest reference library spec, then to the Lorenz system.

        Parameters
        ----------
        prompt:
            Natural-language phenomenon description.

        Returns
        -------
        PhenomenonSpec

        Raises
        ------
        LLMUnavailableError
            If the LLM fails and ``allow_fallback`` is ``False``.
        """
        if self.client.available:
            try:
                raw = self.client.generate_spec_dict(prompt)
                spec = spec_from_json(raw)
                logger.info("Generated spec %r via Claude", spec.slug)
                return spec
            except (LLMUnavailableError, Exception) as exc:  # noqa: BLE001
                logger.warning("LLM generation failed (%s); considering fallback", exc)
                if not self.allow_fallback:
                    raise
        elif not self.allow_fallback:
            raise LLMUnavailableError(
                "LLM unavailable and fallback disabled. Set ANTHROPIC_API_KEY "
                "and install 'simgen[llm]', or enable fallback."
            )

        matched = _match_library(prompt)
        if matched is not None:
            return matched
        logger.info("No library match for prompt; defaulting to Lorenz system")
        return library.get_spec("lorenz-system")

    def generate(
        self, spec: PhenomenonSpec, output_dir: str | Path | None = None
    ) -> GeneratedProject:
        """Render ``spec`` to a project directory."""
        out = output_dir if output_dir is not None else self.settings.output_dir
        return generate_project(spec, out)

    def generate_from_prompt(
        self, prompt: str, output_dir: str | Path | None = None
    ) -> tuple[PhenomenonSpec, GeneratedProject]:
        """One-shot: prompt -> spec -> generated project.

        Returns
        -------
        tuple[PhenomenonSpec, GeneratedProject]
        """
        spec = self.spec_from_prompt(prompt)
        project = self.generate(spec, output_dir)
        return spec, project


def generate_from_prompt(
    prompt: str, output_dir: str | Path | None = None
) -> tuple[PhenomenonSpec, GeneratedProject]:
    """Module-level convenience wrapper around :class:`SimulationGenerator`."""
    return SimulationGenerator().generate_from_prompt(prompt, output_dir)
