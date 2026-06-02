"""Date helpers used across the data and feature layers."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Iterable

import pandas as pd


def parse_date(value: str | date | datetime) -> date:
    """Coerce *value* into a :class:`datetime.date`.

    Args:
        value: ISO-formatted string, :class:`date`, or :class:`datetime`.

    Returns:
        The corresponding :class:`date`.
    """
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.fromisoformat(str(value)[:10]).date()


def days_between(a: str | date | datetime, b: str | date | datetime) -> int:
    """Return ``abs(b - a)`` in whole days.

    Args:
        a: Start date.
        b: End date.

    Returns:
        Non-negative number of days separating *a* and *b*.
    """
    return abs((parse_date(b) - parse_date(a)).days)


def daterange(start: str | date, end: str | date, step: int = 1) -> Iterable[date]:
    """Yield dates in ``[start, end]`` with a configurable step.

    Args:
        start: First date.
        end: Last date (inclusive).
        step: Step in days. Must be positive.

    Yields:
        Each :class:`date` in the range.

    Raises:
        ValueError: When ``step`` is not positive.
    """
    if step <= 0:
        raise ValueError("step must be positive")
    current = parse_date(start)
    last = parse_date(end)
    while current <= last:
        yield current
        current += timedelta(days=step)


def today_utc() -> date:
    """Return today's date in UTC."""
    return datetime.utcnow().date()


def to_pandas_dt(series: pd.Series) -> pd.Series:
    """Convert a series of date-likes into :class:`pandas.Timestamp`.

    Args:
        series: Input series.

    Returns:
        Series with dtype ``datetime64[ns]``.
    """
    return pd.to_datetime(series, errors="coerce")
