"""Unit tests for DataLoader."""

import pandas as pd
import pytest

from src.data.data_loader import DataLoader


class TestDataLoader:
    """Tests for the DataLoader class."""

    def test_load_season_invalid_year_raises_error(
        self, mock_config: dict
    ) -> None:
        """load_season should raise ValueError for out-of-range year.

        Args:
            mock_config: Configuration fixture.
        """
        loader = DataLoader(config=mock_config["data"])
        with pytest.raises(ValueError, match="Year out of range"):
            loader.load_season(1990)

    def test_save_and_load_raw_data(
        self, tmp_path: pytest.TempPathFactory, sample_race_data: pd.DataFrame
    ) -> None:
        """save_raw_data and load_raw_data should round-trip data.

        Args:
            tmp_path: Pytest temporary directory fixture.
            sample_race_data: Sample race DataFrame fixture.
        """
        loader = DataLoader(config={})
        filepath = str(tmp_path / "test_data.parquet")
        loader.save_raw_data(sample_race_data, filepath)
        loaded = loader.load_raw_data(filepath)
        assert len(loaded) == len(sample_race_data)

    def test_load_raw_data_missing_file_raises(self, mock_config: dict) -> None:
        """load_raw_data should raise FileNotFoundError for missing file.

        Args:
            mock_config: Configuration fixture.
        """
        loader = DataLoader(config=mock_config["data"])
        with pytest.raises(FileNotFoundError):
            loader.load_raw_data("/nonexistent/path/data.parquet")

    def test_combine_session_data_empty_race_returns_empty(
        self, mock_config: dict
    ) -> None:
        """_combine_session_data should return empty if race_df is empty.

        Args:
            mock_config: Configuration fixture.
        """
        loader = DataLoader(config=mock_config["data"])
        result = loader._combine_session_data(
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )
        assert result.empty

    def test_combine_session_data_with_valid_race(
        self, mock_config: dict, sample_race_data: pd.DataFrame
    ) -> None:
        """_combine_session_data should merge qualifying columns.

        Args:
            mock_config: Configuration fixture.
            sample_race_data: Sample race DataFrame fixture.
        """
        loader = DataLoader(config=mock_config["data"])
        race_subset = sample_race_data.head(5).copy()
        result = loader._combine_session_data(
            race_subset,
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )
        assert len(result) == 5
