"""Constructor-level feature extractor for F1 race data."""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ConstructorFeaturesExtractor:
    """Extracts 10 constructor/team performance features."""

    FEATURES = [
        "constructor_avg_points",
        "constructor_dnf_rate",
        "constructor_avg_finish_pos",
        "constructor_points_5races",
        "constructor_win_count",
        "constructor_recent_form",
        "engine_max_speed_rank",
        "reliability_score",
        "new_regulations_season",
        "mid_season_upgrades",
    ]

    NEW_REGULATION_YEARS = {2014, 2017, 2021, 2022}

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize ConstructorFeaturesExtractor.

        Args:
            config: Features configuration dictionary.
        """
        self.config = config
        logger.info("ConstructorFeaturesExtractor initialized")

    def extract(
        self, race_data: pd.DataFrame, historical_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Extract constructor features for all teams in race_data.

        Args:
            race_data: Current race DataFrame with TeamName column.
            historical_data: Historical results for rolling calculations.

        Returns:
            DataFrame with constructor features, one row per team.
        """
        if race_data.empty or "TeamName" not in race_data.columns:
            return pd.DataFrame(columns=["TeamName"] + self.FEATURES)

        try:
            teams = race_data["TeamName"].unique()
            rows = []
            for team in teams:
                feats = self._extract_team_features(
                    str(team), race_data, historical_data
                )
                feats["TeamName"] = team
                rows.append(feats)
            result = pd.DataFrame(rows)
            logger.info(
                "Extracted constructor features for %d teams", len(result)
            )
            return result
        except Exception as e:
            logger.error("Constructor feature extraction failed: %s", e)
            raise

    def _extract_team_features(
        self,
        team: str,
        race_data: pd.DataFrame,
        historical_data: pd.DataFrame,
    ) -> Dict[str, float]:
        """Compute 10 features for one constructor.

        Args:
            team: Team name.
            race_data: Current race data.
            historical_data: Full historical dataset.

        Returns:
            Dictionary of feature name -> value.
        """
        hist = (
            historical_data[historical_data["TeamName"] == team].copy()
            if not historical_data.empty
            else pd.DataFrame()
        )
        last5 = hist.tail(5) if not hist.empty else pd.DataFrame()
        points = self._get_col(hist, "Points")
        pos = self._get_col(hist, "Position")
        last5_pts = self._get_col(last5, "Points")

        year = (
            int(race_data["Year"].iloc[0])
            if "Year" in race_data.columns and not race_data.empty
            else 2024
        )

        return {
            "constructor_avg_points": (
                float(points.mean()) if len(points) else np.nan
            ),
            "constructor_dnf_rate": self._dnf_rate(hist),
            "constructor_avg_finish_pos": (
                float(pos.mean()) if len(pos) else np.nan
            ),
            "constructor_points_5races": (
                float(last5_pts.mean()) if len(last5_pts) else np.nan
            ),
            "constructor_win_count": self._win_count(hist),
            "constructor_recent_form": self._recent_form(last5_pts),
            "engine_max_speed_rank": self._engine_speed_rank(
                team, historical_data
            ),
            "reliability_score": self._reliability_score(hist),
            "new_regulations_season": float(year in self.NEW_REGULATION_YEARS),
            "mid_season_upgrades": 0.0,  # Placeholder: no real-time upgrade data
        }

    def _get_col(self, df: pd.DataFrame, col: str) -> pd.Series:
        """Safely get a numeric column."""
        if df.empty or col not in df.columns:
            return pd.Series(dtype=float)
        return pd.to_numeric(df[col], errors="coerce").dropna()

    def _dnf_rate(self, hist: pd.DataFrame) -> float:
        """Compute DNF rate for team."""
        if hist.empty or "Status" not in hist.columns:
            return 0.0
        dnf_mask = hist["Status"].str.contains("DNF|Ret|DNS", na=False)
        return float(dnf_mask.mean())

    def _win_count(self, hist: pd.DataFrame) -> float:
        """Count race wins for team in current season."""
        if hist.empty:
            return 0.0
        pos = self._get_col(hist, "Position")
        return float((pos == 1).sum())

    def _recent_form(self, last5_pts: pd.Series) -> float:
        """Points trend over last 5 races (slope).

        Args:
            last5_pts: Points series for last 5 races.

        Returns:
            Trend slope, or NaN if insufficient data.
        """
        if len(last5_pts) < 3:
            return np.nan
        x = np.arange(len(last5_pts))
        return float(np.polyfit(x, last5_pts.values, 1)[0])

    def _engine_speed_rank(
        self, team: str, historical_data: pd.DataFrame
    ) -> float:
        """Rank engine by average max speed relative to other teams.

        Args:
            team: Team name.
            historical_data: Full historical dataset.

        Returns:
            Rank (1 = fastest), or NaN.
        """
        if historical_data.empty or "TeamName" not in historical_data.columns:
            return np.nan
        if "SpeedST" not in historical_data.columns:
            return np.nan
        team_speeds = (
            historical_data.groupby("TeamName")["SpeedST"]
            .mean()
            .rank(ascending=False)
        )
        return float(team_speeds.get(team, np.nan))

    def _reliability_score(self, hist: pd.DataFrame) -> float:
        """Fraction of races completed (not DNF).

        Args:
            hist: Historical team data.

        Returns:
            Reliability score in [0, 1].
        """
        if hist.empty or "Status" not in hist.columns:
            return 1.0
        dnf_mask = hist["Status"].str.contains("DNF|Ret|DNS", na=False)
        return float(1.0 - dnf_mask.mean())
