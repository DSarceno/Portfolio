"""Canonical match table loader."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.io import ensure_dir, load_csv, save_csv
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

CANONICAL_FILE = "matches_unified.csv"


class DataLoader:
    """Loads (and caches) the canonical match table used by the rest of the pipeline."""

    def __init__(
        self,
        interim_dir: str | Path = "data/interim",
        processed_dir: str | Path = "data/processed",
    ) -> None:
        """Initialize the loader.

        Args:
            interim_dir: Directory containing the unified raw table.
            processed_dir: Directory used for processed outputs.
        """
        self.interim_dir = ensure_dir(interim_dir)
        self.processed_dir = ensure_dir(processed_dir)

    def load_matches(self) -> pd.DataFrame:
        """Load the unified match table, returning an empty schema-shaped table when absent.

        Returns:
            DataFrame with at least the canonical columns.
        """
        path = self.interim_dir / CANONICAL_FILE
        if not path.exists():
            logger.warning("Canonical match table missing at %s; returning empty frame", path)
            cols = [
                "match_id",
                "date",
                "competition",
                "season",
                "stage",
                "group",
                "team_a",
                "team_b",
                "score_a",
                "score_b",
                "neutral_venue",
                "host_country",
                "source",
            ]
            return pd.DataFrame(columns=cols)
        df = load_csv(path)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df

    def save_processed(self, df: pd.DataFrame, name: str) -> Path:
        """Persist *df* under ``processed_dir/<name>.csv``.

        Args:
            df: DataFrame to save.
            name: File stem (no extension).

        Returns:
            The output :class:`Path`.
        """
        return save_csv(df, self.processed_dir / f"{name}.csv")
