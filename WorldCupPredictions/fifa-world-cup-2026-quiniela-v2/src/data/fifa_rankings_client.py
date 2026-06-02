"""Client that reads FIFA ranking snapshots from local CSV files."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils.io import ensure_dir, save_csv
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

EXPECTED_COLUMNS = ["snapshot_date", "team", "rank", "points", "confederation"]


class FifaRankingsClient:
    """Loader for FIFA ranking snapshots.

    The client expects CSV files under ``base_dir`` with columns
    ``[snapshot_date, team, rank, points, confederation]``.
    """

    def __init__(self, base_dir: str | Path = "data/raw/fifa_rankings") -> None:
        """Initialize the client.

        Args:
            base_dir: Directory containing ranking snapshots.
        """
        self.base_dir = ensure_dir(base_dir)

    def load_all(self) -> pd.DataFrame:
        """Concatenate every ranking snapshot found in ``base_dir``.

        Returns:
            A DataFrame with the canonical columns. If no files are present a
            template DataFrame is returned (and persisted) so the pipeline can
            keep running.
        """
        files = sorted(self.base_dir.glob("*.csv"))
        if not files:
            logger.warning(
                "No FIFA ranking snapshots found in %s; generating template", self.base_dir
            )
            template = self._template()
            save_csv(template, self.base_dir / "fifa_ranking_template.csv")
            return template

        frames: list[pd.DataFrame] = []
        for path in files:
            try:
                df = pd.read_csv(path)
                missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
                if missing:
                    logger.warning("Snapshot %s missing columns %s", path, missing)
                    continue
                frames.append(df[EXPECTED_COLUMNS])
            except (ValueError, pd.errors.ParserError) as exc:
                logger.error("Failed to parse %s: %s", path, exc)
        if not frames:
            return self._template()
        return pd.concat(frames, ignore_index=True)

    def latest(self, on_date: Optional[str] = None) -> pd.DataFrame:
        """Return the most recent ranking snapshot at or before *on_date*.

        Args:
            on_date: ISO-formatted cut-off date. ``None`` means latest available.

        Returns:
            One row per team for the chosen snapshot date.
        """
        df = self.load_all()
        if df.empty:
            return df
        df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce")
        if on_date is not None:
            df = df[df["snapshot_date"] <= pd.Timestamp(on_date)]
        if df.empty:
            return df
        latest_date = df["snapshot_date"].max()
        return df[df["snapshot_date"] == latest_date].copy()

    @staticmethod
    def _template() -> pd.DataFrame:
        """Build an empty DataFrame with the expected schema."""
        return pd.DataFrame(columns=EXPECTED_COLUMNS)
