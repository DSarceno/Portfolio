"""FastF1 API client for fetching F1 session data."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import fastf1
import pandas as pd

logger = logging.getLogger(__name__)


class FastF1Client:
    """Client for the FastF1 library to fetch F1 session data.

    Manages cache configuration and provides clean interfaces for
    fetching race, qualifying, and telemetry data.
    """

    def __init__(
        self, cache_dir: str = "data/raw/fastf1_cache", enabled: bool = True
    ) -> None:
        """Initialize FastF1 client with cache settings.

        Args:
            cache_dir: Directory path for FastF1 cache.
            enabled: Whether to enable local caching.
        """
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled

        if enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            fastf1.Cache.enable_cache(str(self.cache_dir))
            logger.info("FastF1 cache enabled at %s", self.cache_dir)
        else:
            logger.info("FastF1 cache disabled")

    def get_session(
        self, year: int, round_num: int, session_type: str = "R"
    ) -> fastf1.core.Session:
        """Load a FastF1 session.

        Args:
            year: Season year.
            round_num: Round number within the season.
            session_type: Session type ('R', 'Q', 'FP1', 'FP2', 'FP3').

        Returns:
            Loaded FastF1 Session object.

        Raises:
            ValueError: If year or round_num are invalid.
            RuntimeError: If session loading fails.
        """
        if year < 2018 or year > 2030:
            raise ValueError(f"Invalid year: {year}")
        if round_num < 1:
            raise ValueError(f"Invalid round: {round_num}")

        try:
            session = fastf1.get_session(year, round_num, session_type)
            session.load()
            logger.info("Loaded session: %d R%d %s", year, round_num, session_type)
            return session
        except Exception as e:
            logger.error(
                "Failed to load session %d R%d %s: %s", year, round_num, session_type, e
            )
            raise RuntimeError(f"Session load failed: {e}") from e

    def get_race_data(self, year: int, round_num: int) -> pd.DataFrame:
        """Fetch race results as a DataFrame.

        Args:
            year: Season year.
            round_num: Round number.

        Returns:
            DataFrame with race results columns.
        """
        try:
            session = self.get_session(year, round_num, "R")
            results = session.results
            if results is None or results.empty:
                logger.warning("No race results for %d R%d", year, round_num)
                return pd.DataFrame()
            results["Year"] = year
            results["Round"] = round_num
            logger.info("Fetched race data: %d rows", len(results))
            return results.reset_index(drop=True)
        except RuntimeError:
            return pd.DataFrame()

    def get_qualifying_data(self, year: int, round_num: int) -> pd.DataFrame:
        """Fetch qualifying results as a DataFrame.

        Args:
            year: Season year.
            round_num: Round number.

        Returns:
            DataFrame with qualifying results.
        """
        try:
            session = self.get_session(year, round_num, "Q")
            results = session.results
            if results is None or results.empty:
                return pd.DataFrame()
            results["Year"] = year
            results["Round"] = round_num
            return results.reset_index(drop=True)
        except RuntimeError:
            return pd.DataFrame()

    def get_lap_data(self, session: fastf1.core.Session) -> pd.DataFrame:
        """Extract lap data from a loaded session.

        Args:
            session: Loaded FastF1 Session object.

        Returns:
            DataFrame with per-lap telemetry data.
        """
        try:
            laps = session.laps
            if laps is None or laps.empty:
                return pd.DataFrame()
            logger.info("Fetched %d laps", len(laps))
            return laps.reset_index(drop=True)
        except Exception as e:
            logger.error("Failed to get lap data: %s", e)
            return pd.DataFrame()

    def get_weather_data(self, session: fastf1.core.Session) -> pd.DataFrame:
        """Extract weather data from a loaded session.

        Args:
            session: Loaded FastF1 Session object.

        Returns:
            DataFrame with weather measurements.
        """
        try:
            weather = session.weather_data
            if weather is None or weather.empty:
                return pd.DataFrame()
            return weather.reset_index(drop=True)
        except Exception as e:
            logger.error("Failed to get weather data: %s", e)
            return pd.DataFrame()

    def get_all_sessions_for_season(self, year: int) -> List[Dict[str, Any]]:
        """Get metadata for all rounds in a season.

        Args:
            year: Season year.

        Returns:
            List of dicts with round metadata.
        """
        try:
            schedule = fastf1.get_event_schedule(year)
            sessions = []
            for _, row in schedule.iterrows():
                sessions.append(
                    {
                        "year": year,
                        "round": int(row.get("RoundNumber", 0)),
                        "name": str(row.get("EventName", "")),
                        "circuit": str(row.get("Location", "")),
                    }
                )
            logger.info("Found %d sessions for %d", len(sessions), year)
            return sessions
        except Exception as e:
            logger.error("Failed to get season schedule for %d: %s", year, e)
            return []

    def get_available_seasons(
        self, start_year: int = 2020, end_year: int = 2025
    ) -> List[int]:
        """Return list of available seasons in range.

        Args:
            start_year: First season year (inclusive).
            end_year: Last season year (inclusive).

        Returns:
            List of year integers.
        """
        return list(range(start_year, end_year + 1))
