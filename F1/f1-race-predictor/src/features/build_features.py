"""Feature orchestrator for F1 Race Predictor."""

import logging
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

from src.features.constructor_features import ConstructorFeaturesExtractor
from src.features.driver_features import DriverFeaturesExtractor
from src.features.lap_features import LapFeaturesExtractor
from src.features.weather_features import WeatherFeaturesExtractor

logger = logging.getLogger(__name__)


class FeatureBuilder:
    """Orchestrates feature extraction from all feature groups.

    Coordinates LapFeaturesExtractor, DriverFeaturesExtractor,
    ConstructorFeaturesExtractor, and WeatherFeaturesExtractor.
    Also handles feature selection and scaling.
    """

    def __init__(
        self,
        config: dict,
        lap_extractor: Optional[LapFeaturesExtractor] = None,
        driver_extractor: Optional[DriverFeaturesExtractor] = None,
        constructor_extractor: Optional[ConstructorFeaturesExtractor] = None,
        weather_extractor: Optional[WeatherFeaturesExtractor] = None,
    ) -> None:
        """Initialize FeatureBuilder.

        Args:
            config: Full project configuration dictionary.
            lap_extractor: Lap features extractor instance.
            driver_extractor: Driver features extractor instance.
            constructor_extractor: Constructor features extractor instance.
            weather_extractor: Weather features extractor instance.
        """
        features_cfg = config.get("features", config)
        self.config = features_cfg
        self.lap_ext = lap_extractor or LapFeaturesExtractor(features_cfg)
        self.driver_ext = driver_extractor or DriverFeaturesExtractor(
            features_cfg
        )
        self.constructor_ext = (
            constructor_extractor
            or ConstructorFeaturesExtractor(features_cfg)
        )
        self.weather_ext = weather_extractor or WeatherFeaturesExtractor(
            features_cfg
        )
        self._scaler: Optional[StandardScaler] = None
        self._selector: Optional[VarianceThreshold] = None
        self._feature_names: List[str] = []
        logger.info("FeatureBuilder initialized")

    def build_features(self, data: dict) -> pd.DataFrame:
        """Orchestrate feature extraction from all groups.

        Args:
            data: Dict with keys: 'race', 'qualifying', 'laps', 'weather',
                  'historical'. Each value is a pd.DataFrame.

        Returns:
            Combined feature DataFrame.
        """
        race_df = data.get("race", pd.DataFrame())
        laps_df = data.get("laps", pd.DataFrame())
        weather_df = data.get("weather", pd.DataFrame())
        historical_df = data.get("historical", pd.DataFrame())

        logger.info("Building features from %d race rows", len(race_df))

        lap_feats = self.lap_ext.extract(laps_df)
        driver_feats = self.driver_ext.extract(race_df, historical_df)
        constructor_feats = self.constructor_ext.extract(
            race_df, historical_df
        )
        weather_feats = self.weather_ext.extract(weather_df, race_df)

        combined = race_df.copy()
        if not lap_feats.empty and "Driver" in lap_feats.columns:
            combined = combined.merge(
                lap_feats, left_on="Abbreviation", right_on="Driver", how="left"
            )
        if not driver_feats.empty and "Abbreviation" in driver_feats.columns:
            combined = combined.merge(
                driver_feats, on="Abbreviation", how="left", suffixes=("", "_drv")
            )
        if not constructor_feats.empty and "TeamName" in constructor_feats.columns:
            combined = combined.merge(
                constructor_feats, on="TeamName", how="left", suffixes=("", "_cst")
            )
        if not weather_feats.empty:
            for col in weather_feats.columns:
                combined[col] = weather_feats[col].iloc[0]

        logger.info("Built feature matrix: %d rows x %d cols", *combined.shape)
        return combined

    def select_features(
        self, data: pd.DataFrame, n_features: int = 30
    ) -> pd.DataFrame:
        """Select top features using variance threshold and correlation.

        Args:
            data: Feature DataFrame (numeric columns only).
            n_features: Maximum number of features to keep.

        Returns:
            DataFrame with selected feature columns.
        """
        numeric = data.select_dtypes(include=[np.number])
        threshold = self.config.get("selection", {}).get(
            "variance_threshold", 0.01
        )
        self._selector = VarianceThreshold(threshold=threshold)
        selected = self._selector.fit_transform(numeric)
        selected_cols = numeric.columns[self._selector.get_support()].tolist()

        # Remove correlated features
        corr_threshold = self.config.get("selection", {}).get(
            "correlation_threshold", 0.95
        )
        selected_df = pd.DataFrame(selected, columns=selected_cols)
        selected_df = self._remove_correlated_features(
            selected_df, corr_threshold
        )

        # Keep top n by variance
        if len(selected_df.columns) > n_features:
            variances = selected_df.var().sort_values(ascending=False)
            selected_df = selected_df[variances.head(n_features).index.tolist()]

        self._feature_names = selected_df.columns.tolist()
        logger.info("Selected %d features", len(self._feature_names))
        return selected_df

    def scale_features(
        self, data: pd.DataFrame, fit: bool = True
    ) -> pd.DataFrame:
        """Apply StandardScaler to numeric features.

        Args:
            data: Feature DataFrame.
            fit: If True, fit the scaler on this data. If False, use fitted.

        Returns:
            Scaled DataFrame.
        """
        if fit:
            self._scaler = StandardScaler()
            scaled = self._scaler.fit_transform(data)
        else:
            if self._scaler is None:
                raise RuntimeError("Scaler not fitted. Call with fit=True first.")
            scaled = self._scaler.transform(data)

        result = pd.DataFrame(scaled, columns=data.columns, index=data.index)
        logger.info("Scaled %d features", len(data.columns))
        return result

    def create_lag_features(
        self,
        data: pd.DataFrame,
        columns: List[str],
        periods: List[int],
    ) -> pd.DataFrame:
        """Create lag features for rolling window analysis.

        Args:
            data: Feature DataFrame sorted by time.
            columns: Columns to create lags for.
            periods: List of lag periods (e.g. [1, 3, 5]).

        Returns:
            DataFrame with additional lag columns.
        """
        result = data.copy()
        for col in columns:
            if col not in result.columns:
                continue
            for p in periods:
                lag_col = f"{col}_lag{p}"
                result[lag_col] = result[col].shift(p)
        logger.info(
            "Created lag features for %d columns x %d periods",
            len(columns),
            len(periods),
        )
        return result

    def _remove_correlated_features(
        self, data: pd.DataFrame, threshold: float
    ) -> pd.DataFrame:
        """Remove features that are highly correlated with others.

        Args:
            data: Numeric feature DataFrame.
            threshold: Correlation threshold above which one feature is dropped.

        Returns:
            DataFrame with correlated features removed.
        """
        corr_matrix = data.corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        to_drop = [
            col for col in upper.columns if any(upper[col] > threshold)
        ]
        logger.debug("Dropping %d correlated features", len(to_drop))
        return data.drop(columns=to_drop)

    def save_feature_pipeline(self, path: str) -> None:
        """Serialize scaler and selector to disk.

        Args:
            path: File path for the pickle file.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "scaler": self._scaler,
                    "selector": self._selector,
                    "feature_names": self._feature_names,
                },
                f,
            )
        logger.info("Saved feature pipeline to %s", path)

    def load_feature_pipeline(self, path: str) -> None:
        """Load scaler and selector from disk.

        Args:
            path: File path for the pickle file.

        Raises:
            FileNotFoundError: If the pipeline file does not exist.
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"Pipeline not found: {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._scaler = data.get("scaler")
        self._selector = data.get("selector")
        self._feature_names = data.get("feature_names", [])
        logger.info("Loaded feature pipeline from %s", path)
