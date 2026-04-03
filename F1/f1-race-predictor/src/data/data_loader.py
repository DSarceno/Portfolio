"""Data loading orchestrator for F1 race data."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.data.fastf1_client import FastF1Client

logger = logging.getLogger(__name__)


class DataLoader:
    """Orchestrates loading of multi-season F1 race data.

    Combines race results, qualifying, lap, and weather data into
    unified DataFrames ready for feature engineering.
    """

    def __init__(
        self,
        config: Dict,
        fastf1_client: Optional[FastF1Client] = None,
    ) -> None:
        """Initialize DataLoader.

        Args:
            config: Data configuration dictionary.
            fastf1_client: FastF1Client instance. Created from config if None.
        """
        self.config = config
        self.client = fastf1_client or FastF1Client(
            cache_dir=config.get("cache_dir", "data/raw/fastf1_cache"),
            enabled=config.get("cache_enabled", True),
        )
        logger.info("DataLoader initialized")

    def load_race(self, year: int, round_num: int) -> Dict[str, pd.DataFrame]:
        """Load all session data for a single race weekend.

        Args:
            year: Season year.
            round_num: Round number.

        Returns:
            Dictionary with keys: 'race', 'qualifying', 'laps', 'weather'.
        """
        logger.info("Loading race data: %d R%d", year, round_num)
        race_df = self.client.get_race_data(year, round_num)
        qual_df = self.client.get_qualifying_data(year, round_num)

        laps_df = pd.DataFrame()
        weather_df = pd.DataFrame()
        try:
            session = self.client.get_session(year, round_num, "R")
            laps_df = self.client.get_lap_data(session)
            weather_df = self.client.get_weather_data(session)
        except RuntimeError as e:
            logger.warning("Could not load lap/weather data: %s", e)

        return {
            "race": race_df,
            "qualifying": qual_df,
            "laps": laps_df,
            "weather": weather_df,
        }

    def load_season(self, year: int) -> pd.DataFrame:
        """Load all races for a full season.

        Args:
            year: Season year.

        Returns:
            Combined DataFrame of all race results for the season.

        Raises:
            ValueError: If year is outside valid range.
        """
        if year < 2018 or year > 2030:
            raise ValueError(f"Year out of range: {year}")

        logger.info("Loading season %d", year)
        sessions = self.client.get_all_sessions_for_season(year)
        season_frames: List[pd.DataFrame] = []

        for meta in sessions:
            round_num = meta["round"]
            if round_num < 1:
                continue
            try:
                data = self.load_race(year, round_num)
                combined = self._combine_session_data(
                    data["race"],
                    data["qualifying"],
                    data["laps"],
                    data["weather"],
                )
                if not combined.empty:
                    season_frames.append(combined)
            except Exception as e:
                logger.warning("Skipping R%d due to error: %s", round_num, e)

        if not season_frames:
            logger.warning("No data loaded for season %d", year)
            return pd.DataFrame()

        result = pd.concat(season_frames, ignore_index=True)
        logger.info("Season %d loaded: %d rows", year, len(result))
        return result

    def load_multiple_seasons(
        self, start_year: int, end_year: int
    ) -> pd.DataFrame:
        """Load data for multiple seasons.

        Args:
            start_year: First season year (inclusive).
            end_year: Last season year (inclusive).

        Returns:
            Combined DataFrame for all seasons.
        """
        logger.info("Loading seasons %d-%d", start_year, end_year)
        frames: List[pd.DataFrame] = []

        for year in range(start_year, end_year + 1):
            try:
                season_df = self.load_season(year)
                if not season_df.empty:
                    frames.append(season_df)
            except Exception as e:
                logger.error("Failed to load season %d: %s", year, e)

        if not frames:
            return pd.DataFrame()

        result = pd.concat(frames, ignore_index=True)
        logger.info("Loaded %d total rows for %d-%d", len(result), start_year, end_year)
        return result

    def _combine_session_data(
        self,
        race_df: pd.DataFrame,
        qual_df: pd.DataFrame,
        lap_df: pd.DataFrame,
        weather_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge race, qualifying, lap and weather data into one DataFrame.

        Args:
            race_df: Race results DataFrame.
            qual_df: Qualifying results DataFrame.
            lap_df: Lap times DataFrame.
            weather_df: Weather data DataFrame.

        Returns:
            Merged DataFrame, or empty DataFrame if race_df is empty.
        """
        if race_df.empty:
            return pd.DataFrame()

        combined = race_df.copy()

        if not qual_df.empty:
            qual_cols = [c for c in qual_df.columns if c not in combined.columns]
            qual_cols += ["Abbreviation"]
            qual_merge = qual_df[
                [c for c in qual_cols if c in qual_df.columns]
            ].drop_duplicates("Abbreviation")
            combined = combined.merge(qual_merge, on="Abbreviation", how="left")

        if not weather_df.empty:
            weather_summary = weather_df.mean(numeric_only=True).to_dict()
            for col, val in weather_summary.items():
                combined[f"weather_{col}"] = val

        logger.debug("Combined session data: %d rows", len(combined))
        return combined

    def save_raw_data(self, data: pd.DataFrame, filepath: str) -> None:
        """Save DataFrame to parquet file.

        Args:
            data: DataFrame to save.
            filepath: Destination file path.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        data.to_parquet(path, index=False)
        logger.info("Saved %d rows to %s", len(data), path)

    def load_raw_data(self, filepath: str) -> pd.DataFrame:
        """Load DataFrame from parquet file.

        Args:
            filepath: Source file path.

        Returns:
            Loaded DataFrame.

        Raises:
            FileNotFoundError: If file does not exist.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        data = pd.read_parquet(path)
        logger.info("Loaded %d rows from %s", len(data), path)
        return data
