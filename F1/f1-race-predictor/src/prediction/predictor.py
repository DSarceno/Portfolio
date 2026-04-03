"""Race outcome predictor using trained ML models."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.models.base_model import BaseModel

logger = logging.getLogger(__name__)


class RacePredictor:
    """Generates race outcome predictions using a trained model.

    Handles data fetching, feature building, and result formatting.
    """

    def __init__(
        self,
        config: Dict,
        model: Optional[BaseModel] = None,
        feature_builder=None,
    ) -> None:
        """Initialize RacePredictor.

        Args:
            config: Full project configuration dictionary.
            model: Optional pre-loaded trained model.
            feature_builder: Optional FeatureBuilder instance.
        """
        self.config = config
        self._model = model
        self._feature_builder = feature_builder
        logger.info("RacePredictor initialized")

    def load_model(self, model_path: str) -> None:
        """Load a trained model from disk.

        Args:
            model_path: Path to the serialized model file.

        Raises:
            FileNotFoundError: If model file does not exist.
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        self._model = BaseModel.load(model_path)
        logger.info("Loaded model from %s", model_path)

    def predict_race(self, season: int, round_num: int) -> pd.DataFrame:
        """Predict race results for a given season and round.

        Args:
            season: F1 season year.
            round_num: Race round number.

        Returns:
            DataFrame with columns: driver, team, predicted_position,
            probability, confidence.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if self._model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        logger.info("Predicting race: %d R%d", season, round_num)
        try:
            from src.data.fastf1_client import FastF1Client
            from src.features.build_features import FeatureBuilder

            client = FastF1Client(
                cache_dir=self.config.get("data", {}).get(
                    "cache_dir", "data/raw/fastf1_cache"
                )
            )
            data = {
                "race": client.get_race_data(season, round_num),
                "qualifying": client.get_qualifying_data(season, round_num),
                "laps": pd.DataFrame(),
                "weather": pd.DataFrame(),
                "historical": pd.DataFrame(),
            }
            if self._feature_builder is None:
                self._feature_builder = FeatureBuilder(self.config)

            features_df = self._feature_builder.build_features(data)
            numeric_cols = features_df.select_dtypes(include=[np.number]).columns
            X = features_df[numeric_cols].fillna(0).values

            return self.predict_from_features(X, features_df)
        except Exception as e:
            logger.error("Race prediction failed: %s", e)
            raise

    def predict_from_features(
        self,
        X: np.ndarray,
        context_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Generate predictions from a pre-built feature matrix.

        Args:
            X: Feature matrix of shape (n_drivers, n_features).
            context_df: Optional DataFrame with driver/team context.

        Returns:
            Predictions DataFrame.
        """
        if self._model is None:
            raise RuntimeError("No model loaded.")

        proba = self._model.predict_proba(X)
        positions = self._rank_predictions(proba)

        drivers: List[str] = []
        teams: List[str] = []
        if context_df is not None:
            if "Abbreviation" in context_df.columns:
                drivers = context_df["Abbreviation"].tolist()
            if "TeamName" in context_df.columns:
                teams = context_df["TeamName"].tolist()

        if not drivers:
            drivers = [f"DRV_{i+1}" for i in range(len(X))]
        if not teams:
            teams = ["Unknown"] * len(X)

        return self._format_results(drivers, teams, positions, proba)

    def get_race_probabilities(
        self, season: int, round_num: int
    ) -> Dict[str, List[float]]:
        """Get per-position probability for each driver.

        Args:
            season: F1 season year.
            round_num: Round number.

        Returns:
            Dict of driver_abbr -> list of 20 position probabilities.
        """
        predictions_df = self.predict_race(season, round_num)
        result: Dict[str, List[float]] = {}
        for _, row in predictions_df.iterrows():
            result[str(row["driver"])] = [float(row["probability"])] * 20
        return result

    def _rank_predictions(self, probabilities: np.ndarray) -> np.ndarray:
        """Convert probability matrix to ranked position predictions.

        Assigns unique positions 1-N based on argmax probabilities,
        breaking ties by rank.

        Args:
            probabilities: Array of shape (n_drivers, n_classes).

        Returns:
            Array of 1-indexed position integers.
        """
        # Expected position = sum(position * probability)
        positions_range = np.arange(1, probabilities.shape[1] + 1)
        expected_pos = (probabilities * positions_range).sum(axis=1)
        # Rank by expected position (lower = better)
        ranks = expected_pos.argsort().argsort() + 1
        return ranks.astype(int)

    def _format_results(
        self,
        drivers: List[str],
        teams: List[str],
        positions: np.ndarray,
        probabilities: np.ndarray,
    ) -> pd.DataFrame:
        """Build output DataFrame with prediction results.

        Args:
            drivers: Driver abbreviation list.
            teams: Team name list.
            positions: Predicted position array.
            probabilities: Probability matrix.

        Returns:
            Formatted predictions DataFrame.
        """
        max_proba = probabilities.max(axis=1)
        confidence = (probabilities.max(axis=1) - 1.0 / probabilities.shape[1]) / (
            1.0 - 1.0 / probabilities.shape[1] + 1e-9
        )

        df = pd.DataFrame(
            {
                "driver": drivers,
                "team": teams,
                "predicted_position": positions,
                "probability": max_proba,
                "confidence": np.clip(confidence, 0, 1),
            }
        )
        df = df.sort_values("predicted_position").reset_index(drop=True)
        return df
