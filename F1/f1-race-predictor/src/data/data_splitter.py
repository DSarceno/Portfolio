"""Temporal train/validation/test data splitter."""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class DataSplitter:
    """Splits F1 race data using temporal strategy.

    Ensures no future data leaks into training set by splitting
    based on season year rather than random sampling.
    """

    def __init__(self, config: Dict) -> None:
        """Initialize DataSplitter with training configuration.

        Args:
            config: Training config dict with keys:
                test_size, val_size, random_seed, split_strategy.
        """
        self.config = config
        training_cfg = config.get("training", config)
        self.test_size = float(training_cfg.get("test_size", 0.2))
        self.val_size = float(training_cfg.get("val_size", 0.15))
        self.random_seed = int(training_cfg.get("random_seed", 42))
        logger.info("DataSplitter initialized with test_size=%.2f", self.test_size)

    def temporal_split(
        self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data temporally into train, validation, and test sets.

        Uses year-based splitting: latest years become val/test.
        Default: 2020-2023 train, 2024 val, 2025 test.

        Args:
            data: Combined multi-season DataFrame with 'Year' column.

        Returns:
            Tuple of (train_df, val_df, test_df).

        Raises:
            ValueError: If 'Year' column is missing.
        """
        if "Year" not in data.columns:
            raise ValueError("Data must contain 'Year' column for temporal split")

        years = sorted(data["Year"].unique().tolist())
        train_years, val_years, test_years = self._assign_years(years)

        logger.info(
            "Temporal split - train: %s, val: %s, test: %s",
            train_years,
            val_years,
            test_years,
        )

        return self._split_by_year(data, train_years, val_years, test_years)

    def _assign_years(
        self, years: List[int]
    ) -> Tuple[List[int], List[int], List[int]]:
        """Assign years to train/val/test based on size ratios.

        Args:
            years: Sorted list of available years.

        Returns:
            Tuple of (train_years, val_years, test_years).
        """
        n = len(years)
        if n == 0:
            return [], [], []
        if n == 1:
            return years, [], []
        if n == 2:
            return [years[0]], [years[1]], []

        n_test = max(1, round(n * self.test_size))
        n_val = max(1, round(n * self.val_size))
        n_train = n - n_test - n_val

        if n_train < 1:
            n_train = 1
            n_val = max(0, n - n_test - n_train)

        train_years = years[:n_train]
        val_years = years[n_train: n_train + n_val]
        test_years = years[n_train + n_val:]
        return train_years, val_years, test_years

    def _split_by_year(
        self,
        data: pd.DataFrame,
        train_years: List[int],
        val_years: List[int],
        test_years: List[int],
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Filter data by year lists to create split DataFrames.

        Args:
            data: Source DataFrame.
            train_years: Years for training set.
            val_years: Years for validation set.
            test_years: Years for test set.

        Returns:
            Tuple of (train_df, val_df, test_df).
        """
        train = data[data["Year"].isin(train_years)].reset_index(drop=True)
        val = (
            data[data["Year"].isin(val_years)].reset_index(drop=True)
            if val_years
            else pd.DataFrame(columns=data.columns)
        )
        test = (
            data[data["Year"].isin(test_years)].reset_index(drop=True)
            if test_years
            else pd.DataFrame(columns=data.columns)
        )
        logger.info(
            "Split sizes — train: %d, val: %d, test: %d",
            len(train),
            len(val),
            len(test),
        )
        return train, val, test

    def get_split_info(self, data: pd.DataFrame) -> Dict:
        """Compute split sizes and date ranges without performing split.

        Args:
            data: DataFrame with 'Year' column.

        Returns:
            Dictionary with split metadata.
        """
        if "Year" not in data.columns:
            return {}

        years = sorted(data["Year"].unique().tolist())
        train_years, val_years, test_years = self._assign_years(years)
        return {
            "total_rows": len(data),
            "train_years": train_years,
            "val_years": val_years,
            "test_years": test_years,
            "train_rows": int(data["Year"].isin(train_years).sum()),
            "val_rows": int(data["Year"].isin(val_years).sum()),
            "test_rows": int(data["Year"].isin(test_years).sum()),
        }

    def save_splits(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
        output_dir: str,
    ) -> None:
        """Save train/val/test splits to parquet files.

        Args:
            train: Training DataFrame.
            val: Validation DataFrame.
            test: Test DataFrame.
            output_dir: Directory to save files.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        train.to_parquet(out / "train.parquet", index=False)
        val.to_parquet(out / "val.parquet", index=False)
        test.to_parquet(out / "test.parquet", index=False)
        logger.info("Saved splits to %s", output_dir)

    def load_splits(
        self, input_dir: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load train/val/test splits from parquet files.

        Args:
            input_dir: Directory containing split parquet files.

        Returns:
            Tuple of (train_df, val_df, test_df).

        Raises:
            FileNotFoundError: If split files are missing.
        """
        inp = Path(input_dir)
        for fname in ["train.parquet", "val.parquet", "test.parquet"]:
            if not (inp / fname).exists():
                raise FileNotFoundError(f"Split file not found: {inp / fname}")

        train = pd.read_parquet(inp / "train.parquet")
        val = pd.read_parquet(inp / "val.parquet")
        test = pd.read_parquet(inp / "test.parquet")
        logger.info("Loaded splits from %s", input_dir)
        return train, val, test
