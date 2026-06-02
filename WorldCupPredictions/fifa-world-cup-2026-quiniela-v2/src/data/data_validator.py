"""Schema and integrity validation for the canonical match table."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

REQUIRED_COLUMNS = [
    "date",
    "team_a",
    "team_b",
    "score_a",
    "score_b",
]


@dataclass
class ValidationReport:
    """Summary of a validation pass."""

    n_rows: int
    n_missing_required: int
    n_negative_scores: int
    n_duplicate_keys: int
    columns_present: list[str]
    columns_missing: list[str]

    @property
    def is_valid(self) -> bool:
        """``True`` when no schema or integrity problems were detected."""
        return (
            self.n_missing_required == 0
            and self.n_negative_scores == 0
            and not self.columns_missing
        )


def validate_match_dataframe(df: pd.DataFrame) -> ValidationReport:
    """Run schema + integrity checks against *df*.

    Args:
        df: Match DataFrame to validate.

    Returns:
        A :class:`ValidationReport` with the diagnostic counts.
    """
    columns_missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    columns_present = [c for c in df.columns]

    if columns_missing:
        logger.error("Match table missing required columns: %s", columns_missing)

    n_missing_required = 0
    n_negative_scores = 0
    n_duplicate_keys = 0
    if not columns_missing:
        n_missing_required = int(df[REQUIRED_COLUMNS].isna().any(axis=1).sum())
        scores = df[["score_a", "score_b"]].apply(pd.to_numeric, errors="coerce")
        n_negative_scores = int(((scores < 0)).any(axis=1).sum())
        n_duplicate_keys = int(df.duplicated(subset=["date", "team_a", "team_b"]).sum())

    report = ValidationReport(
        n_rows=int(len(df)),
        n_missing_required=n_missing_required,
        n_negative_scores=n_negative_scores,
        n_duplicate_keys=n_duplicate_keys,
        columns_present=columns_present,
        columns_missing=columns_missing,
    )
    logger.info(
        "Validation: rows=%d, missing=%d, neg=%d, dup=%d, missing_cols=%s",
        report.n_rows,
        report.n_missing_required,
        report.n_negative_scores,
        report.n_duplicate_keys,
        report.columns_missing,
    )
    return report
