"""Source registry mapping logical names to client classes and metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class SourceSpec:
    """Static description of an external data source."""

    name: str
    requires_api_key: bool
    fallback_to_manual: bool
    description: str


SOURCES: dict[str, SourceSpec] = {
    "fifa_rankings": SourceSpec(
        name="fifa_rankings",
        requires_api_key=False,
        fallback_to_manual=True,
        description="Official FIFA Men's World Ranking snapshots.",
    ),
    "football_data": SourceSpec(
        name="football_data",
        requires_api_key=False,
        fallback_to_manual=True,
        description="football-data.org fixtures, results and standings.",
    ),
    "statsbomb_open": SourceSpec(
        name="statsbomb_open",
        requires_api_key=False,
        fallback_to_manual=False,
        description="StatsBomb open-data event-level historical matches.",
    ),
    "manual": SourceSpec(
        name="manual",
        requires_api_key=False,
        fallback_to_manual=False,
        description="Manual CSV overrides located in data/raw/manual.",
    ),
    "kaggle_international": SourceSpec(
        name="kaggle_international",
        requires_api_key=False,
        fallback_to_manual=False,
        description="Kaggle 'International football results 1872-2024' dataset.",
    ),
}


def get_source(name: str) -> SourceSpec:
    """Return the :class:`SourceSpec` for *name*.

    Args:
        name: Logical source name.

    Raises:
        KeyError: If *name* is not registered.
    """
    return SOURCES[name]


def list_sources() -> list[str]:
    """List all registered source names."""
    return sorted(SOURCES.keys())


def register_source(spec: SourceSpec) -> None:
    """Register a new source spec, overwriting any existing entry with the same name.

    Args:
        spec: Source specification.
    """
    SOURCES[spec.name] = spec


def build_loader(name: str, factory: Callable[[], object]) -> object:
    """Return ``factory()`` after validating that *name* is registered.

    Args:
        name: Logical source name.
        factory: Zero-argument callable that returns the source client.

    Returns:
        Instance produced by *factory*.

    Raises:
        KeyError: If *name* is not registered.
    """
    get_source(name)
    return factory()
