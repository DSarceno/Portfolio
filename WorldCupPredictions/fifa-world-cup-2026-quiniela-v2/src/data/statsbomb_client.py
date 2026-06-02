"""Loader for StatsBomb open data (local-only)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.utils.io import ensure_dir
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

MATCH_COLUMNS = [
    "match_id",
    "match_date",
    "competition",
    "season",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "stage",
]


class StatsBombClient:
    """Reads StatsBomb open-data competitions and matches from local files."""

    def __init__(self, base_dir: str | Path = "data/raw/statsbomb") -> None:
        """Initialize the loader.

        Args:
            base_dir: Directory containing the open-data dump (or extracted CSVs).
        """
        self.base_dir = ensure_dir(base_dir)

    def load_competitions(self) -> pd.DataFrame:
        """Load the competitions index.

        Returns:
            DataFrame with the original StatsBomb competition fields, or an
            empty frame with placeholder columns.
        """
        candidate = self.base_dir / "competitions.json"
        if not candidate.exists():
            logger.warning("StatsBomb competitions.json missing in %s", self.base_dir)
            return pd.DataFrame(columns=["competition_id", "season_id", "competition_name"])
        with candidate.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return pd.DataFrame(data)

    def load_matches(self) -> pd.DataFrame:
        """Load every match JSON file found under ``base_dir/matches``.

        Returns:
            DataFrame with :data:`MATCH_COLUMNS`.
        """
        matches_dir = self.base_dir / "matches"
        if not matches_dir.exists():
            logger.warning("StatsBomb matches directory missing in %s", self.base_dir)
            return pd.DataFrame(columns=MATCH_COLUMNS)

        rows: list[dict] = []
        for path in matches_dir.rglob("*.json"):
            try:
                with path.open("r", encoding="utf-8") as fh:
                    entries = json.load(fh)
                for entry in entries:
                    rows.append(
                        {
                            "match_id": entry.get("match_id"),
                            "match_date": entry.get("match_date"),
                            "competition": entry.get("competition", {}).get("competition_name"),
                            "season": entry.get("season", {}).get("season_name"),
                            "home_team": entry.get("home_team", {}).get("home_team_name"),
                            "away_team": entry.get("away_team", {}).get("away_team_name"),
                            "home_score": entry.get("home_score"),
                            "away_score": entry.get("away_score"),
                            "stage": entry.get("competition_stage", {}).get("name"),
                        }
                    )
            except (json.JSONDecodeError, OSError) as exc:
                logger.error("Failed to read %s: %s", path, exc)

        if not rows:
            logger.info("StatsBomb open data not present; returning empty table")
            return pd.DataFrame(columns=MATCH_COLUMNS)
        return pd.DataFrame(rows, columns=MATCH_COLUMNS)
