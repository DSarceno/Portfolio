"""Weather feature extractor for F1 race sessions."""

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class WeatherFeaturesExtractor:
    """Extracts 8 weather condition features from FastF1 weather data.

    Weather is treated as a critical feature group — always included
    regardless of feature selection thresholds.
    """

    FEATURES = [
        "track_temp_celsius",
        "air_temp_celsius",
        "humidity_percent",
        "wind_speed_kmh",
        "is_raining",
        "track_status",
        "temp_diff",
        "weather_aggressiveness",
    ]

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize WeatherFeaturesExtractor.

        Args:
            config: Features configuration dictionary.
        """
        self.config = config
        logger.info("WeatherFeaturesExtractor initialized")

    def extract(
        self,
        weather_data: pd.DataFrame,
        session_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Extract weather features summarized for a session.

        Args:
            weather_data: FastF1 weather DataFrame with time-series rows.
            session_data: Session results DataFrame for merging context.

        Returns:
            Single-row DataFrame with 8 weather feature columns.
        """
        if weather_data.empty:
            logger.warning("Empty weather data, using defaults")
            return self._default_features()

        try:
            features = self._compute_features(weather_data)
            result = pd.DataFrame([features])
            logger.info("Extracted weather features")
            return result
        except Exception as e:
            logger.error("Weather feature extraction failed: %s", e)
            return self._default_features()

    def _compute_features(
        self, weather_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Compute all 8 weather features from time-series data.

        Args:
            weather_data: Weather DataFrame from FastF1.

        Returns:
            Dictionary with feature values.
        """
        filled = weather_data.fillna(method="ffill").fillna(method="bfill")

        track_temp = self._get_mean(filled, "TrackTemp")
        air_temp = self._get_mean(filled, "AirTemp")
        humidity = self._get_mean(filled, "Humidity")
        wind_speed = self._get_mean(filled, "WindSpeed")
        rainfall = self._get_max(filled, "Rainfall")

        is_raining = float(rainfall > 0) if not np.isnan(rainfall) else 0.0
        track_status = self._track_status(is_raining, rainfall)
        temp_diff = (
            float(track_temp - air_temp)
            if not (np.isnan(track_temp) or np.isnan(air_temp))
            else np.nan
        )

        features = {
            "track_temp_celsius": track_temp,
            "air_temp_celsius": air_temp,
            "humidity_percent": humidity,
            "wind_speed_kmh": wind_speed,
            "is_raining": is_raining,
            "track_status": track_status,
            "temp_diff": temp_diff,
            "weather_aggressiveness": self._calculate_aggressiveness(
                {
                    "is_raining": is_raining,
                    "wind_speed_kmh": wind_speed,
                    "humidity_percent": humidity,
                    "track_status": track_status,
                }
            ),
        }
        return features

    def _get_mean(self, df: pd.DataFrame, col: str) -> float:
        """Get mean of a numeric column.

        Args:
            df: DataFrame.
            col: Column name.

        Returns:
            Mean value or NaN.
        """
        if col not in df.columns:
            return np.nan
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        return float(vals.mean()) if not vals.empty else np.nan

    def _get_max(self, df: pd.DataFrame, col: str) -> float:
        """Get max of a numeric column.

        Args:
            df: DataFrame.
            col: Column name.

        Returns:
            Max value or 0.
        """
        if col not in df.columns:
            return 0.0
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        return float(vals.max()) if not vals.empty else 0.0

    def _track_status(self, is_raining: float, rainfall: float) -> float:
        """Map rainfall to track status code.

        Args:
            is_raining: Boolean flag as float.
            rainfall: Max rainfall value.

        Returns:
            Track status: 0=dry, 1=damp, 2=wet, 3=very wet.
        """
        if is_raining == 0:
            return 0.0
        if rainfall < 0.5:
            return 1.0
        if rainfall < 2.0:
            return 2.0
        return 3.0

    def _calculate_aggressiveness(self, row: Dict[str, float]) -> float:
        """Compute composite weather aggressiveness score [0, 1].

        Args:
            row: Dict with weather feature values.

        Returns:
            Aggressiveness score between 0 and 1.
        """
        score = 0.0
        # Rain contribution (0-0.5)
        rain_contrib = row.get("is_raining", 0) * 0.5
        # Wind contribution (0-0.3, normalised at 50 km/h)
        wind = row.get("wind_speed_kmh", 0)
        wind_contrib = min(float(wind) / 50.0, 1.0) * 0.3 if wind else 0.0
        # Humidity contribution (0-0.2)
        humidity = row.get("humidity_percent", 50)
        hum_contrib = (max(0.0, float(humidity) - 50) / 50.0) * 0.2

        score = rain_contrib + wind_contrib + hum_contrib
        return float(min(score, 1.0))

    def _default_features(self) -> pd.DataFrame:
        """Return a DataFrame with default weather feature values.

        Returns:
            Single-row DataFrame with neutral weather defaults.
        """
        defaults = {
            "track_temp_celsius": 35.0,
            "air_temp_celsius": 25.0,
            "humidity_percent": 50.0,
            "wind_speed_kmh": 10.0,
            "is_raining": 0.0,
            "track_status": 0.0,
            "temp_diff": 10.0,
            "weather_aggressiveness": 0.0,
        }
        return pd.DataFrame([defaults])
