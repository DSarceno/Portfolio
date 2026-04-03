"""Unit tests for feature extractors."""

import numpy as np
import pandas as pd
import pytest

from src.features.constructor_features import ConstructorFeaturesExtractor
from src.features.driver_features import DriverFeaturesExtractor
from src.features.lap_features import LapFeaturesExtractor
from src.features.weather_features import WeatherFeaturesExtractor


class TestLapFeatures:
    """Tests for LapFeaturesExtractor."""

    def test_extract_returns_expected_columns(
        self, sample_lap_data: pd.DataFrame, mock_config: dict
    ) -> None:
        """extract should return all 15 feature columns.

        Args:
            sample_lap_data: Synthetic lap data fixture.
            mock_config: Configuration fixture.
        """
        extractor = LapFeaturesExtractor(mock_config)
        result = extractor.extract(sample_lap_data)
        for feature in LapFeaturesExtractor.FEATURES:
            assert feature in result.columns, f"Missing column: {feature}"

    def test_avg_lap_time_calculation(
        self, sample_lap_data: pd.DataFrame, mock_config: dict
    ) -> None:
        """avg_lap_time_last_5_laps should be a positive number.

        Args:
            sample_lap_data: Synthetic lap data fixture.
            mock_config: Configuration fixture.
        """
        extractor = LapFeaturesExtractor(mock_config)
        result = extractor.extract(sample_lap_data)
        avg_times = result["avg_lap_time_last_5_laps"].dropna()
        assert (avg_times > 0).all()

    def test_handles_missing_telemetry(self, mock_config: dict) -> None:
        """extract should handle empty DataFrame gracefully.

        Args:
            mock_config: Configuration fixture.
        """
        extractor = LapFeaturesExtractor(mock_config)
        result = extractor.extract(pd.DataFrame())
        assert result.empty or len(result) == 0

    def test_tire_degradation_non_negative(
        self, sample_lap_data: pd.DataFrame, mock_config: dict
    ) -> None:
        """tire_degradation should be non-negative when computable.

        Args:
            sample_lap_data: Synthetic lap data fixture.
            mock_config: Configuration fixture.
        """
        extractor = LapFeaturesExtractor(mock_config)
        result = extractor.extract(sample_lap_data)
        if "tire_degradation" in result.columns:
            valid = result["tire_degradation"].dropna()
            assert (valid >= 0).all()


class TestDriverFeatures:
    """Tests for DriverFeaturesExtractor."""

    def test_extract_returns_expected_columns(
        self, sample_race_data: pd.DataFrame, mock_config: dict
    ) -> None:
        """extract should return all 20 driver feature columns.

        Args:
            sample_race_data: Sample race data fixture.
            mock_config: Configuration fixture.
        """
        extractor = DriverFeaturesExtractor(mock_config)
        result = extractor.extract(
            sample_race_data, sample_race_data
        )
        for feature in DriverFeaturesExtractor.FEATURES:
            assert feature in result.columns, f"Missing column: {feature}"

    def test_win_rate_between_0_and_1(
        self, sample_race_data: pd.DataFrame, mock_config: dict
    ) -> None:
        """win_rate should be in [0, 1].

        Args:
            sample_race_data: Sample race data fixture.
            mock_config: Configuration fixture.
        """
        extractor = DriverFeaturesExtractor(mock_config)
        result = extractor.extract(sample_race_data, sample_race_data)
        valid = result["win_rate"].dropna()
        assert (valid >= 0).all() and (valid <= 1).all()

    def test_dnf_rate_for_dnf_driver(
        self, sample_race_data: pd.DataFrame, mock_config: dict
    ) -> None:
        """dnf_rate should be > 0 for a driver with DNF records.

        Args:
            sample_race_data: Sample race data fixture.
            mock_config: Configuration fixture.
        """
        extractor = DriverFeaturesExtractor(mock_config)
        result = extractor.extract(sample_race_data, sample_race_data)
        sar_row = result[result["Abbreviation"] == "SAR"]
        if not sar_row.empty:
            assert float(sar_row["dnf_rate"].iloc[0]) > 0

    def test_rookie_detection(
        self, sample_race_data: pd.DataFrame, mock_config: dict
    ) -> None:
        """is_rookie should be 0 or 1 for all drivers.

        Args:
            sample_race_data: Sample race data fixture.
            mock_config: Configuration fixture.
        """
        extractor = DriverFeaturesExtractor(mock_config)
        result = extractor.extract(sample_race_data, sample_race_data)
        assert result["is_rookie"].isin([0.0, 1.0]).all()


class TestWeatherFeatures:
    """Tests for WeatherFeaturesExtractor."""

    def test_extract_returns_weather_columns(
        self, sample_weather_data: pd.DataFrame, mock_config: dict
    ) -> None:
        """extract should return all 8 weather feature columns.

        Args:
            sample_weather_data: Synthetic weather data fixture.
            mock_config: Configuration fixture.
        """
        extractor = WeatherFeaturesExtractor(mock_config)
        result = extractor.extract(sample_weather_data, pd.DataFrame())
        for feature in WeatherFeaturesExtractor.FEATURES:
            assert feature in result.columns, f"Missing: {feature}"

    def test_aggressiveness_between_0_and_1(
        self, sample_weather_data: pd.DataFrame, mock_config: dict
    ) -> None:
        """weather_aggressiveness should be in [0, 1].

        Args:
            sample_weather_data: Synthetic weather data fixture.
            mock_config: Configuration fixture.
        """
        extractor = WeatherFeaturesExtractor(mock_config)
        result = extractor.extract(sample_weather_data, pd.DataFrame())
        val = float(result["weather_aggressiveness"].iloc[0])
        assert 0.0 <= val <= 1.0

    def test_handles_missing_weather_with_defaults(
        self, mock_config: dict
    ) -> None:
        """extract should return default values for empty weather data.

        Args:
            mock_config: Configuration fixture.
        """
        extractor = WeatherFeaturesExtractor(mock_config)
        result = extractor.extract(pd.DataFrame(), pd.DataFrame())
        assert not result.empty
        assert float(result["is_raining"].iloc[0]) == 0.0
