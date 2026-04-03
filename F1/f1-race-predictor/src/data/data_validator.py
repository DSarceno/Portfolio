"""Data validation for F1 race datasets."""

import logging
from typing import Any, Dict, List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates F1 race DataFrames against quality rules.

    Checks for required columns, null thresholds, minimum row counts,
    and basic data type constraints.
    """

    REQUIRED_COLUMNS: List[str] = [
        "Year",
        "Round",
        "Abbreviation",
        "Position",
        "GridPosition",
        "Points",
        "TeamName",
    ]

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize validator with configuration.

        Args:
            config: Validation config dict with keys:
                max_null_percentage, min_laps_per_race, min_drivers_per_race.
        """
        validation_cfg = config.get("validation", {})
        self.max_null_pct = float(
            validation_cfg.get("max_null_percentage", 0.3)
        )
        self.min_laps = int(validation_cfg.get("min_laps_per_race", 10))
        self.min_drivers = int(validation_cfg.get("min_drivers_per_race", 10))
        logger.info("DataValidator initialized")

    def validate(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Run all validation checks on the dataset.

        Args:
            data: DataFrame to validate.

        Returns:
            Tuple of (is_valid, list_of_error_messages).
        """
        errors: List[str] = []

        if data.empty:
            return False, ["DataFrame is empty"]

        checks = [
            self._check_required_columns,
            self._check_null_percentage,
            self._check_min_drivers,
            self._check_data_types,
        ]

        for check in checks:
            ok, msg = check(data)
            if not ok:
                errors.append(msg)

        is_valid = len(errors) == 0
        if is_valid:
            logger.info("Validation passed for %d rows", len(data))
        else:
            logger.warning("Validation failed: %s", errors)
        return is_valid, errors

    def _check_required_columns(
        self, data: pd.DataFrame
    ) -> Tuple[bool, str]:
        """Check that all required columns are present.

        Args:
            data: DataFrame to check.

        Returns:
            Tuple of (passed, error_message).
        """
        missing = [c for c in self.REQUIRED_COLUMNS if c not in data.columns]
        if missing:
            return False, f"Missing required columns: {missing}"
        return True, ""

    def _check_null_percentage(
        self, data: pd.DataFrame
    ) -> Tuple[bool, str]:
        """Check that null percentage is below threshold.

        Args:
            data: DataFrame to check.

        Returns:
            Tuple of (passed, error_message).
        """
        null_pct = data.isnull().mean().max()
        if null_pct > self.max_null_pct:
            return (
                False,
                f"Null percentage {null_pct:.2%} exceeds limit {self.max_null_pct:.2%}",
            )
        return True, ""

    def _check_min_drivers(self, data: pd.DataFrame) -> Tuple[bool, str]:
        """Check minimum drivers per race.

        Args:
            data: DataFrame to check.

        Returns:
            Tuple of (passed, error_message).
        """
        if "Abbreviation" not in data.columns or "Round" not in data.columns:
            return True, ""

        drivers_per_race = (
            data.groupby(["Year", "Round"])["Abbreviation"].nunique()
        )
        min_count = int(drivers_per_race.min())
        if min_count < self.min_drivers:
            return (
                False,
                f"Min drivers per race {min_count} < required {self.min_drivers}",
            )
        return True, ""

    def _check_data_types(self, data: pd.DataFrame) -> Tuple[bool, str]:
        """Check that key numeric columns have correct types.

        Args:
            data: DataFrame to check.

        Returns:
            Tuple of (passed, error_message).
        """
        numeric_cols = ["Position", "GridPosition", "Points"]
        for col in numeric_cols:
            if col in data.columns:
                try:
                    pd.to_numeric(data[col], errors="raise")
                except (ValueError, TypeError):
                    return False, f"Column {col} has non-numeric values"
        return True, ""

    def generate_validation_report(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate a detailed validation report.

        Args:
            data: DataFrame to analyze.

        Returns:
            Dictionary with stats: rows, columns, null_counts, dtypes.
        """
        is_valid, errors = self.validate(data)
        report: Dict[str, Any] = {
            "is_valid": is_valid,
            "errors": errors,
            "rows": len(data),
            "columns": list(data.columns),
            "null_counts": data.isnull().sum().to_dict(),
            "dtypes": {c: str(t) for c, t in data.dtypes.items()},
        }
        if "Year" in data.columns:
            report["seasons"] = sorted(data["Year"].unique().tolist())
        if "Round" in data.columns and "Year" in data.columns:
            report["races"] = int(data.groupby(["Year", "Round"]).ngroups)
        logger.info("Validation report generated: valid=%s", is_valid)
        return report
