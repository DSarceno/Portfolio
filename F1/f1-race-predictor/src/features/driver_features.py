"""Driver-level feature extractor for F1 race data."""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DriverFeaturesExtractor:
    """Extracts 20 driver performance features from historical race data."""

    FEATURES = [
        "driver_experience_years",
        "total_races",
        "avg_finish_position_all_time",
        "win_rate",
        "podium_rate",
        "points_per_race",
        "dnf_rate",
        "avg_finish_position_5races",
        "points_per_race_5races",
        "avg_grid_to_finish_delta_5races",
        "team_car_mate_comparison",
        "team_avg_points",
        "best_circuit",
        "best_circuit_avg_points",
        "pole_position_count",
        "team_change_recent",
        "seasons_with_current_team",
        "is_rookie",
        "avg_qualify_position",
        "avg_race_position",
    ]

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize DriverFeaturesExtractor.

        Args:
            config: Features configuration dictionary.
        """
        self.config = config
        logger.info("DriverFeaturesExtractor initialized")

    def extract(
        self, race_data: pd.DataFrame, historical_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Extract driver features for all drivers in race_data.

        Args:
            race_data: Current race DataFrame with driver/team columns.
            historical_data: Historical results for rolling calculations.

        Returns:
            DataFrame with driver features, one row per driver.
        """
        if race_data.empty:
            return pd.DataFrame(columns=["Abbreviation"] + self.FEATURES)
        try:
            drivers = race_data["Abbreviation"].unique()
            rows = []
            for driver in drivers:
                feats = self._extract_single_driver(
                    driver, race_data, historical_data
                )
                feats["Abbreviation"] = driver
                rows.append(feats)
            result = pd.DataFrame(rows)
            logger.info(
                "Extracted driver features for %d drivers", len(result)
            )
            return result
        except Exception as e:
            logger.error("Driver feature extraction failed: %s", e)
            raise

    def _extract_single_driver(
        self,
        driver: str,
        race_data: pd.DataFrame,
        historical_data: pd.DataFrame,
    ) -> Dict[str, float]:
        """Compute all 20 features for one driver.

        Args:
            driver: Driver abbreviation.
            race_data: Current race data.
            historical_data: Full historical dataset.

        Returns:
            Dictionary of feature name -> value.
        """
        hist = (
            historical_data[historical_data["Abbreviation"] == driver].copy()
            if not historical_data.empty
            else pd.DataFrame()
        )
        current = race_data[race_data["Abbreviation"] == driver]
        team: Optional[str] = (
            str(current["TeamName"].iloc[0])
            if not current.empty and "TeamName" in current.columns
            else None
        )
        last5 = hist.tail(5) if not hist.empty else pd.DataFrame()
        positions = self._get_col(hist, "Position")
        points = self._get_col(hist, "Points")
        grid = self._get_col(hist, "GridPosition")
        years = hist["Year"].nunique() if "Year" in hist.columns else 0

        return {
            "driver_experience_years": float(years),
            "total_races": float(len(hist)),
            "avg_finish_position_all_time": (
                float(positions.mean()) if len(positions) else np.nan
            ),
            "win_rate": (
                float((positions == 1).mean()) if len(positions) else 0.0
            ),
            "podium_rate": (
                float((positions <= 3).mean()) if len(positions) else 0.0
            ),
            "points_per_race": (
                float(points.mean()) if len(points) else 0.0
            ),
            "dnf_rate": self._dnf_rate(hist),
            "avg_finish_position_5races": (
                float(self._get_col(last5, "Position").mean())
                if not last5.empty
                else np.nan
            ),
            "points_per_race_5races": (
                float(self._get_col(last5, "Points").mean())
                if not last5.empty
                else 0.0
            ),
            "avg_grid_to_finish_delta_5races": self._grid_to_finish_delta(
                last5
            ),
            "team_car_mate_comparison": self._teammate_comparison(
                driver, historical_data, team
            ),
            "team_avg_points": self._team_avg_points(historical_data, team),
            "best_circuit": self._best_circuit_encoded(hist),
            "best_circuit_avg_points": self._best_circuit_avg_points(hist),
            "pole_position_count": (
                float((grid == 1).sum()) if len(grid) else 0.0
            ),
            "team_change_recent": self._team_change_recent(hist),
            "seasons_with_current_team": self._seasons_with_team(hist, team),
            "is_rookie": float(years <= 1),
            "avg_qualify_position": (
                float(self._get_col(last5, "GridPosition").mean())
                if not last5.empty
                else np.nan
            ),
            "avg_race_position": (
                float(self._get_col(last5, "Position").mean())
                if not last5.empty
                else np.nan
            ),
        }

    def _get_col(self, df: pd.DataFrame, col: str) -> pd.Series:
        """Safely get a numeric column."""
        if df.empty or col not in df.columns:
            return pd.Series(dtype=float)
        return pd.to_numeric(df[col], errors="coerce").dropna()

    def _dnf_rate(self, hist: pd.DataFrame) -> float:
        """Compute DNF rate from historical data."""
        if hist.empty or "Status" not in hist.columns:
            return 0.0
        dnf_mask = hist["Status"].str.contains("DNF|Ret|DNS", na=False)
        return float(dnf_mask.mean())

    def _grid_to_finish_delta(self, last5: pd.DataFrame) -> float:
        """Avg positions gained/lost vs grid."""
        if last5.empty:
            return np.nan
        grid = self._get_col(last5, "GridPosition").values
        pos = self._get_col(last5, "Position").values
        n = min(len(grid), len(pos))
        if n == 0:
            return np.nan
        return float(np.mean(grid[:n] - pos[:n]))

    def _teammate_comparison(
        self,
        driver: str,
        historical_data: pd.DataFrame,
        team: Optional[str],
    ) -> float:
        """Compare driver avg points to teammate avg."""
        if historical_data.empty or team is None:
            return np.nan
        team_data = historical_data[historical_data["TeamName"] == team]
        teammates = team_data[team_data["Abbreviation"] != driver]
        driver_pts = self._get_col(
            historical_data[historical_data["Abbreviation"] == driver], "Points"
        ).mean()
        mate_pts = self._get_col(teammates, "Points").mean()
        if np.isnan(driver_pts) or np.isnan(mate_pts):
            return np.nan
        return float(driver_pts - mate_pts)

    def _team_avg_points(
        self, historical_data: pd.DataFrame, team: Optional[str]
    ) -> float:
        """Compute team average points per race."""
        if historical_data.empty or team is None:
            return np.nan
        team_data = historical_data[historical_data["TeamName"] == team]
        return float(self._get_col(team_data, "Points").mean())

    def _best_circuit_encoded(self, hist: pd.DataFrame) -> float:
        """Encode best circuit as numeric hash."""
        if hist.empty or "Location" not in hist.columns:
            return np.nan
        if "Points" not in hist.columns:
            return np.nan
        circuit_avg = hist.groupby("Location")["Points"].mean()
        if circuit_avg.empty:
            return np.nan
        return float(hash(str(circuit_avg.idxmax())) % 10000)

    def _best_circuit_avg_points(self, hist: pd.DataFrame) -> float:
        """Average points at driver best circuit."""
        if (
            hist.empty
            or "Location" not in hist.columns
            or "Points" not in hist.columns
        ):
            return np.nan
        circuit_avg = hist.groupby("Location")["Points"].mean()
        return float(circuit_avg.max()) if not circuit_avg.empty else np.nan

    def _team_change_recent(self, hist: pd.DataFrame) -> float:
        """Detect team change in last 2 seasons."""
        if (
            hist.empty
            or "TeamName" not in hist.columns
            or "Year" not in hist.columns
        ):
            return 0.0
        recent = hist[hist["Year"] >= hist["Year"].max() - 1]
        return float(recent["TeamName"].nunique() > 1)

    def _seasons_with_team(
        self, hist: pd.DataFrame, team: Optional[str]
    ) -> float:
        """Number of seasons with current team."""
        if (
            hist.empty
            or team is None
            or "TeamName" not in hist.columns
        ):
            return 0.0
        team_hist = hist[hist["TeamName"] == team]
        if team_hist.empty or "Year" not in team_hist.columns:
            return 0.0
        return float(team_hist["Year"].nunique())
