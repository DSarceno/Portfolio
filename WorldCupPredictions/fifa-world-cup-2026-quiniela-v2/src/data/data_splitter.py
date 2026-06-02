"""Strict temporal data splitting helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class TemporalSplit:
    """Container holding a temporal split.

    Attributes:
        train: Training rows.
        validation: Validation rows.
        test: Test / backtest rows (may be empty).
    """

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def temporal_split(
    df: pd.DataFrame,
    date_column: str = "date",
    validation_year: int = 2025,
    test_start_date: Optional[str] = None,
) -> TemporalSplit:
    """Split *df* temporally into train / validation / test.

    Args:
        df: Input DataFrame, must contain a date column.
        date_column: Name of the date column.
        validation_year: Year used for validation rows.
        test_start_date: Optional ISO date marking the test set start.

    Returns:
        A :class:`TemporalSplit`.

    Raises:
        KeyError: If *date_column* is missing.
    """
    if date_column not in df.columns:
        raise KeyError(f"Missing date column '{date_column}'")
    work = df.copy()
    work[date_column] = pd.to_datetime(work[date_column], errors="coerce")
    work = work.dropna(subset=[date_column]).sort_values(date_column)

    if test_start_date is not None:
        test_mask = work[date_column] >= pd.Timestamp(test_start_date)
        train_val = work[~test_mask]
        test = work[test_mask]
    else:
        train_val = work
        test = pd.DataFrame(columns=work.columns)

    val_mask = train_val[date_column].dt.year == validation_year
    train = train_val[~val_mask]
    validation = train_val[val_mask]

    logger.info(
        "Temporal split: train=%d, validation=%d, test=%d", len(train), len(validation), len(test)
    )
    return TemporalSplit(train=train, validation=validation, test=test)
