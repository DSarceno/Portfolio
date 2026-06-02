"""Unifies historical and live match data from multiple sources."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from src.data.fifa_rankings_client import FifaRankingsClient
from src.data.football_data_client import FootballDataClient
from src.data.statsbomb_client import StatsBombClient
from src.utils.io import ensure_dir, save_csv
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

CANONICAL_COLUMNS = [
    "match_id",
    "date",
    "competition",
    "season",
    "stage",
    "team_a",
    "team_b",
    "score_a",
    "score_b",
    "neutral_venue",
    "host_country",
    "source",
]


class ResultsCollector:
    """High-level facade that builds a single match table from all clients."""

    def __init__(
        self,
        football_client: Optional[FootballDataClient] = None,
        statsbomb_client: Optional[StatsBombClient] = None,
        rankings_client: Optional[FifaRankingsClient] = None,
        interim_dir: str | Path = "data/interim",
    ) -> None:
        """Initialize the collector.

        Args:
            football_client: Optional pre-configured football-data client.
            statsbomb_client: Optional pre-configured StatsBomb client.
            rankings_client: Optional pre-configured FIFA rankings client.
            interim_dir: Directory used to persist the unified match table.
        """
        self.football_client = football_client or FootballDataClient()
        self.statsbomb_client = statsbomb_client or StatsBombClient()
        self.rankings_client = rankings_client or FifaRankingsClient()
        self.interim_dir = ensure_dir(interim_dir)

    def collect_all(self, competitions: list[str]) -> pd.DataFrame:
        """Collect matches for every competition code in *competitions*.

        Args:
            competitions: Competition codes to query.

        Returns:
            Unified DataFrame with :data:`CANONICAL_COLUMNS`.
        """
        frames: list[pd.DataFrame] = []
        for competition in competitions:
            df_football = self.football_client.fetch_matches(competition)
            if not df_football.empty:
                frames.append(self._normalize_football(df_football, competition))

        df_statsbomb = self.statsbomb_client.load_matches()
        if not df_statsbomb.empty:
            frames.append(self._normalize_statsbomb(df_statsbomb))

        if not frames:
            logger.warning("No match data collected; returning empty canonical table")
            return pd.DataFrame(columns=CANONICAL_COLUMNS)

        unified = pd.concat(frames, ignore_index=True)
        unified = unified.drop_duplicates(subset=["date", "team_a", "team_b"])
        unified = unified.sort_values("date").reset_index(drop=True)
        save_csv(unified, self.interim_dir / "matches_unified.csv")
        return unified

    @staticmethod
    def _normalize_football(df: pd.DataFrame, competition: str) -> pd.DataFrame:
        """Map a football-data table into the canonical schema.

        Args:
            df: DataFrame from :class:`FootballDataClient`.
            competition: Competition code.

        Returns:
            DataFrame with :data:`CANONICAL_COLUMNS`.
        """
        out = pd.DataFrame()
        out["match_id"] = df.get("match_id")
        out["date"] = df.get("date")
        out["competition"] = competition
        out["season"] = df.get("season")
        out["stage"] = df.get("stage", "")
        out["team_a"] = df.get("home_team")
        out["team_b"] = df.get("away_team")
        out["score_a"] = df.get("home_score")
        out["score_b"] = df.get("away_score")
        out["neutral_venue"] = df.get("neutral_venue", False)
        out["host_country"] = df.get("venue", "")
        out["source"] = "football_data"
        return out

    @staticmethod
    def _normalize_statsbomb(df: pd.DataFrame) -> pd.DataFrame:
        """Map a StatsBomb match table into the canonical schema.

        Args:
            df: DataFrame from :class:`StatsBombClient`.

        Returns:
            DataFrame with :data:`CANONICAL_COLUMNS`.
        """
        out = pd.DataFrame()
        out["match_id"] = df.get("match_id")
        out["date"] = df.get("match_date")
        out["competition"] = df.get("competition")
        out["season"] = df.get("season")
        out["stage"] = df.get("stage", "")
        out["team_a"] = df.get("home_team")
        out["team_b"] = df.get("away_team")
        out["score_a"] = df.get("home_score")
        out["score_b"] = df.get("away_score")
        out["neutral_venue"] = False
        out["host_country"] = ""
        out["source"] = "statsbomb"
        return out
