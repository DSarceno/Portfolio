"""Lap-level feature extractor for F1 race data."""

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class LapFeaturesExtractor:
    """Extracts 15 lap-level features from FastF1 telemetry data.

    Features cover timing, speed, throttle/brake usage, and tire health.
    Missing telemetry is handled gracefully with NaN fill.
    """

    FEATURES = [
        "avg_lap_time_last_5_laps",
        "median_lap_time",
        "best_lap_time",
        "lap_time_std",
        "pace_trend",
        "consistency_score",
        "max_speed",
        "avg_speed",
        "throttle_usage",
        "brake_usage",
        "cornering_speed",
        "straight_speed",
        "fuel_consumption_rate",
        "tire_degradation",
        "drs_usage_count",
    ]

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize LapFeaturesExtractor.

        Args:
            config: Features configuration dictionary.
        """
        self.config = config
        logger.info("LapFeaturesExtractor initialized")

    def extract(self, lap_data: pd.DataFrame) -> pd.DataFrame:
        """Extract lap features grouped by driver.

        Args:
            lap_data: FastF1 laps DataFrame with per-lap rows.

        Returns:
            DataFrame with one row per driver and 15 feature columns.
        """
        if lap_data.empty:
            logger.warning("Empty lap data, returning empty features")
            return pd.DataFrame(columns=["Driver"] + self.FEATURES)
        try:
            result = (
                lap_data.groupby("Driver")
                .apply(self._extract_driver_features)
                .reset_index()
            )
            logger.info("Extracted lap features for %d drivers", len(result))
            return result
        except Exception as e:
            logger.error("Lap feature extraction failed: %s", e)
            raise

    def _extract_driver_features(
        self, driver_laps: pd.DataFrame
    ) -> pd.Series:
        """Compute all 15 features for a single driver.

        Args:
            driver_laps: Subset of lap_data for one driver.

        Returns:
            pd.Series with feature values indexed by feature name.
        """
        lap_times = self._get_lap_times(driver_laps)
        features = {
            "avg_lap_time_last_5_laps": self._avg_last_n(lap_times, 5),
            "median_lap_time": (
                float(np.median(lap_times)) if len(lap_times) else np.nan
            ),
            "best_lap_time": (
                float(np.min(lap_times)) if len(lap_times) else np.nan
            ),
            "lap_time_std": (
                float(np.std(lap_times)) if len(lap_times) > 1 else np.nan
            ),
            "pace_trend": self._pace_trend(lap_times),
            "consistency_score": self._consistency_score(lap_times),
            "max_speed": self._get_speed_stat(driver_laps, "SpeedST", "max"),
            "avg_speed": self._get_speed_stat(driver_laps, "SpeedST", "mean"),
            "throttle_usage": self._throttle_usage(driver_laps),
            "brake_usage": self._brake_usage(driver_laps),
            "cornering_speed": self._get_speed_stat(
                driver_laps, "SpeedFL", "mean"
            ),
            "straight_speed": self._get_speed_stat(
                driver_laps, "SpeedI1", "mean"
            ),
            "fuel_consumption_rate": self._fuel_consumption_rate(lap_times),
            "tire_degradation": self._tire_degradation(lap_times),
            "drs_usage_count": self._drs_usage_count(driver_laps),
        }
        return pd.Series(features)

    def _get_lap_times(self, driver_laps: pd.DataFrame) -> np.ndarray:
        """Extract numeric lap times in seconds."""
        if "LapTime" not in driver_laps.columns:
            return np.array([])
        times = (
            pd.to_timedelta(driver_laps["LapTime"], errors="coerce")
            .dt.total_seconds()
        )
        return times.dropna().values

    def _avg_last_n(self, lap_times: np.ndarray, n: int) -> float:
        """Average of last n lap times."""
        if len(lap_times) == 0:
            return np.nan
        return float(np.mean(lap_times[-n:]))

    def _pace_trend(self, lap_times: np.ndarray) -> float:
        """Linear slope of lap times (positive = degrading)."""
        if len(lap_times) < 3:
            return np.nan
        x = np.arange(len(lap_times))
        return float(np.polyfit(x, lap_times, 1)[0])

    def _consistency_score(self, lap_times: np.ndarray) -> float:
        """Inverse of lap time standard deviation."""
        if len(lap_times) < 2:
            return np.nan
        std = float(np.std(lap_times))
        return 1.0 / std if std > 0 else np.nan

    def _get_speed_stat(
        self, driver_laps: pd.DataFrame, col: str, stat: str
    ) -> float:
        """Get a speed statistic from a column."""
        if col not in driver_laps.columns:
            return np.nan
        vals = pd.to_numeric(driver_laps[col], errors="coerce").dropna()
        if vals.empty:
            return np.nan
        return float(vals.max() if stat == "max" else vals.mean())

    def _throttle_usage(self, driver_laps: pd.DataFrame) -> float:
        """Estimate throttle usage fraction."""
        if "Throttle" not in driver_laps.columns:
            return np.nan
        vals = pd.to_numeric(
            driver_laps["Throttle"], errors="coerce"
        ).dropna()
        return float((vals > 0).mean()) if not vals.empty else np.nan

    def _brake_usage(self, driver_laps: pd.DataFrame) -> float:
        """Estimate brake usage fraction."""
        if "Brake" not in driver_laps.columns:
            return np.nan
        vals = pd.to_numeric(driver_laps["Brake"], errors="coerce").dropna()
        return float(vals.mean()) if not vals.empty else np.nan

    def _fuel_consumption_rate(self, lap_times: np.ndarray) -> float:
        """Estimate fuel consumption proxy from pace trend."""
        trend = self._pace_trend(lap_times)
        return float(abs(trend)) if not np.isnan(trend) else np.nan

    def _tire_degradation(self, lap_times: np.ndarray) -> float:
        """Compute lap time increase rate as tire degradation proxy."""
        if len(lap_times) < 3:
            return np.nan
        diffs = np.diff(lap_times)
        pos_diffs = diffs[diffs > 0]
        return float(np.mean(pos_diffs)) if len(pos_diffs) else np.nan

    def _drs_usage_count(self, driver_laps: pd.DataFrame) -> float:
        """Count DRS activations."""
        for col in ["DRS", "IsDrsOpen"]:
            if col in driver_laps.columns:
                vals = pd.to_numeric(
                    driver_laps[col], errors="coerce"
                ).fillna(0)
                return float((vals > 0).sum())
        return np.nan
