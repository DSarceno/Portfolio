"""Cross-validation split utilities."""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def rolling_origin_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    min_train_size: int = 200,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield rolling-origin train/test index pairs.

    Args:
        df: DataFrame ordered by date.
        n_splits: Number of folds.
        min_train_size: Minimum training rows for a valid fold.

    Yields:
        Tuples ``(train_idx, test_idx)`` as integer numpy arrays.
    """
    n = len(df)
    if n <= min_train_size:
        return
    fold_size = max((n - min_train_size) // n_splits, 1)
    for fold in range(n_splits):
        train_end = min_train_size + fold * fold_size
        test_end = train_end + fold_size
        if test_end > n:
            test_end = n
        if train_end <= 0 or train_end >= n or train_end >= test_end:
            continue
        train_idx = np.arange(train_end)
        test_idx = np.arange(train_end, test_end)
        yield train_idx, test_idx
