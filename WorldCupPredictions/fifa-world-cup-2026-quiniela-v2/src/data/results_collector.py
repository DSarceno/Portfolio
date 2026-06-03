"""Unifies historical and live match data from multiple sources."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from src.data.fifa_rankings_client import FifaRankingsClient
from src.data.football_data_client import FootballDataClient
from src.data.kaggle_results_client import KaggleResultsClient
from src.data.statsbomb_client import StatsBombClient
from src.utils.io import ensure_dir, load_csv, save_csv
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

CANONICAL_COLUMNS = [
    "match_id",
    "date",
    "competition",
    "season",
    "stage",
    "group",
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
        kaggle_client: Optional[KaggleResultsClient] = None,
        interim_dir: str | Path = "data/interim",
    ) -> None:
        """Initialize the collector.

        Args:
            football_client: Optional pre-configured football-data client.
            statsbomb_client: Optional pre-configured StatsBomb client.
            rankings_client: Optional pre-configured FIFA rankings client.
            kaggle_client: Optional pre-configured Kaggle dataset client.
            interim_dir: Directory used to persist the unified match table.
        """
        self.football_client = football_client or FootballDataClient()
        self.statsbomb_client = statsbomb_client or StatsBombClient()
        self.rankings_client = rankings_client or FifaRankingsClient()
        self.kaggle_client = kaggle_client or KaggleResultsClient()
        self.interim_dir = ensure_dir(interim_dir)

    def collect_all(
        self,
        competitions: list[str],
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        include_kaggle: bool = True,
        include_friendlies: bool = True,
        include_qualifications: bool = True,
        merge_with_existing: bool = True,
        persist: bool = True,
    ) -> pd.DataFrame:
        """Collect matches from every configured source and persist the union.

        Args:
            competitions: Football-data.org competition codes to query.
            start_year: Optional minimum year (inclusive) applied to all
                sources.
            end_year: Optional maximum year (inclusive) applied to all sources.
            include_kaggle: When ``True`` the Kaggle dataset is included
                whenever the local file is present.
            include_friendlies: Forward to the Kaggle filter.
            include_qualifications: Forward to the Kaggle filter.
            merge_with_existing: When ``True`` the result is merged with the
                already-persisted canonical table (deduplicated on date +
                team pair) instead of replacing it.
            persist: When ``True`` the result is written to
                ``data/interim/matches_unified.csv``.

        Returns:
            Unified DataFrame with :data:`CANONICAL_COLUMNS`.
        """
        frames: list[pd.DataFrame] = []

        if include_kaggle and self.kaggle_client.is_available():
            kaggle_df = self.kaggle_client.load(
                start_year=start_year,
                end_year=end_year,
                include_friendlies=include_friendlies,
                include_qualifications=include_qualifications,
            )
            if not kaggle_df.empty:
                logger.info("Kaggle source contributed %d matches", len(kaggle_df))
                frames.append(kaggle_df)
        elif include_kaggle:
            logger.info(
                "Kaggle dataset not found at %s; skipping (download instructions in README)",
                self.kaggle_client.csv_path,
            )

        for competition in competitions:
            df_football = self.football_client.fetch_matches(competition)
            if not df_football.empty:
                normalized = self._normalize_football(df_football, competition)
                if start_year is not None or end_year is not None:
                    normalized = self._apply_year_filter(normalized, start_year, end_year)
                if not normalized.empty:
                    logger.info(
                        "football-data source contributed %d matches for %s",
                        len(normalized),
                        competition,
                    )
                    frames.append(normalized)

        df_statsbomb = self.statsbomb_client.load_matches()
        if not df_statsbomb.empty:
            normalized = self._normalize_statsbomb(df_statsbomb)
            if start_year is not None or end_year is not None:
                normalized = self._apply_year_filter(normalized, start_year, end_year)
            if not normalized.empty:
                logger.info("StatsBomb source contributed %d matches", len(normalized))
                frames.append(normalized)

        unified = self._merge_frames(frames)

        if merge_with_existing:
            unified = self._merge_with_existing(unified)

        if persist:
            save_csv(unified, self.interim_dir / "matches_unified.csv")
        logger.info("Canonical table built: %d matches", len(unified))
        return unified

    def collect_kaggle_only(
        self,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        include_friendlies: bool = True,
        include_qualifications: bool = True,
        merge_with_existing: bool = True,
        persist: bool = True,
    ) -> pd.DataFrame:
        """Collect from the Kaggle dataset only (useful for fast bootstraps).

        Args:
            start_year: Optional minimum year.
            end_year: Optional maximum year.
            include_friendlies: When ``False`` drop friendlies.
            include_qualifications: When ``False`` drop qualification matches.
            merge_with_existing: When ``True`` merge with the existing table.
            persist: When ``True`` persist the canonical table.

        Returns:
            Unified DataFrame.
        """
        kaggle_df = self.kaggle_client.load(
            start_year=start_year,
            end_year=end_year,
            include_friendlies=include_friendlies,
            include_qualifications=include_qualifications,
        )
        unified = self._merge_frames([kaggle_df] if not kaggle_df.empty else [])
        if merge_with_existing:
            unified = self._merge_with_existing(unified)
        if persist:
            save_csv(unified, self.interim_dir / "matches_unified.csv")
        logger.info("Canonical table built (Kaggle-only): %d matches", len(unified))
        return unified

    def _merge_frames(self, frames: list[pd.DataFrame]) -> pd.DataFrame:
        """Concatenate, deduplicate, and sort multiple source frames."""
        if not frames:
            logger.warning("No match data collected; returning empty canonical table")
            return pd.DataFrame(columns=CANONICAL_COLUMNS)
        unified = pd.concat(frames, ignore_index=True)
        unified = unified.dropna(subset=["team_a", "team_b", "date"])
        unified = unified.drop_duplicates(subset=["date", "team_a", "team_b"], keep="last")
        unified = unified.sort_values("date").reset_index(drop=True)
        return unified[CANONICAL_COLUMNS]

    def _merge_with_existing(self, new_df: pd.DataFrame) -> pd.DataFrame:
        """Merge *new_df* with the persisted canonical table, if any."""
        path = self.interim_dir / "matches_unified.csv"
        if not path.exists():
            return new_df
        try:
            existing = load_csv(path)
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("Could not read existing canonical table: %s", exc)
            return new_df
        if existing.empty:
            return new_df
        for col in CANONICAL_COLUMNS:
            if col not in existing.columns:
                existing[col] = pd.NA
        existing = existing[CANONICAL_COLUMNS]
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
        combined = combined.dropna(subset=["team_a", "team_b", "date"])
        combined = combined.drop_duplicates(subset=["date", "team_a", "team_b"], keep="last")
        combined = combined.sort_values("date").reset_index(drop=True)
        combined["date"] = combined["date"].dt.strftime("%Y-%m-%d")
        return combined

    @staticmethod
    def _apply_year_filter(
        df: pd.DataFrame, start_year: Optional[int], end_year: Optional[int]
    ) -> pd.DataFrame:
        """Filter *df* by the inclusive ``[start_year, end_year]`` range."""
        if df.empty:
            return df
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        if start_year is not None:
            d = d[d["date"].dt.year >= int(start_year)]
        if end_year is not None:
            d = d[d["date"].dt.year <= int(end_year)]
        d["date"] = d["date"].dt.strftime("%Y-%m-%d")
        return d

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
        out["group"] = df.get("group", "")
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
        out["group"] = ""
        out["team_a"] = df.get("home_team")
        out["team_b"] = df.get("away_team")
        out["score_a"] = df.get("home_score")
        out["score_b"] = df.get("away_score")
        out["neutral_venue"] = False
        out["host_country"] = ""
        out["source"] = "statsbomb"
        return out
