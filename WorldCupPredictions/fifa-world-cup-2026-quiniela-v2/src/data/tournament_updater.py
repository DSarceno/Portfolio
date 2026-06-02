"""Tournament-state updater: ingest results and refresh derived tables."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

from src.data.data_loader import DataLoader
from src.data.results_collector import CANONICAL_COLUMNS
from src.utils.io import ensure_dir, save_csv, timestamp
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class UpdateSummary:
    """Summary of a tournament update run."""

    matches_appended: int
    matches_total: int
    snapshot_path: str


class TournamentUpdater:
    """Appends newly played World Cup matches and persists raw snapshots."""

    def __init__(
        self,
        loader: Optional[DataLoader] = None,
        updates_dir: str | Path = "data/raw/tournament_updates",
        interim_dir: str | Path = "data/interim",
    ) -> None:
        """Initialize the updater.

        Args:
            loader: Optional pre-built :class:`DataLoader`.
            updates_dir: Directory storing raw daily snapshots.
            interim_dir: Directory storing the unified canonical table.
        """
        self.loader = loader or DataLoader(interim_dir=interim_dir)
        self.updates_dir = ensure_dir(updates_dir)
        self.interim_dir = ensure_dir(interim_dir)

    def append_results(
        self, new_matches: pd.DataFrame, source: str = "tournament_update"
    ) -> UpdateSummary:
        """Append *new_matches* to the canonical match table.

        Args:
            new_matches: DataFrame with the canonical columns plus required keys.
            source: Tag stored in the ``source`` column.

        Returns:
            An :class:`UpdateSummary` describing the operation.
        """
        if new_matches.empty:
            logger.warning("No new matches provided to append_results()")
            return UpdateSummary(0, 0, "")

        prepared = new_matches.copy()
        for col in CANONICAL_COLUMNS:
            if col not in prepared.columns:
                prepared[col] = pd.NA
        prepared["source"] = source

        snapshot_file = self.updates_dir / f"update_{timestamp()}.csv"
        save_csv(prepared[CANONICAL_COLUMNS], snapshot_file)

        existing = self.loader.load_matches()
        combined = pd.concat([existing, prepared[CANONICAL_COLUMNS]], ignore_index=True)
        combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
        combined = combined.drop_duplicates(subset=["date", "team_a", "team_b"], keep="last")
        combined = combined.sort_values("date").reset_index(drop=True)
        save_csv(combined, self.interim_dir / "matches_unified.csv")

        appended = int(len(combined) - len(existing))
        logger.info("Appended %d match(es); canonical table now has %d rows", appended, len(combined))
        return UpdateSummary(
            matches_appended=appended,
            matches_total=int(len(combined)),
            snapshot_path=str(snapshot_file),
        )

    def matches_played_on(self, on_date: date | str) -> pd.DataFrame:
        """Return canonical-table rows played on a given date.

        Args:
            on_date: ISO string or :class:`date`.

        Returns:
            DataFrame with the matches played that day.
        """
        matches = self.loader.load_matches()
        if matches.empty:
            return matches
        matches["date"] = pd.to_datetime(matches["date"], errors="coerce")
        target = pd.Timestamp(on_date).normalize()
        return matches[matches["date"].dt.normalize() == target].copy()

    def world_cup_state(
        self, tournament_year: int, today: Optional[date | str] = None
    ) -> pd.DataFrame:
        """Compute per-team tournament-state aggregates as of *today*.

        Args:
            tournament_year: Tournament year filter (e.g. ``2026``).
            today: Optional cut-off date; ``None`` means no cut-off.

        Returns:
            DataFrame with one row per team and aggregated columns.
        """
        matches = self.loader.load_matches()
        if matches.empty:
            return pd.DataFrame(
                columns=[
                    "team",
                    "world_cup_points_so_far",
                    "world_cup_goal_diff_so_far",
                    "world_cup_goals_for_so_far",
                    "world_cup_goals_against_so_far",
                    "world_cup_clean_sheets",
                ]
            )
        matches["date"] = pd.to_datetime(matches["date"], errors="coerce")
        wc = matches[matches["date"].dt.year == tournament_year].copy()
        if today is not None:
            wc = wc[wc["date"] <= pd.Timestamp(today)]
        wc = wc.dropna(subset=["score_a", "score_b"])

        records: dict[str, dict[str, float]] = {}
        for _, row in wc.iterrows():
            for team, gf, ga in (
                (row["team_a"], row["score_a"], row["score_b"]),
                (row["team_b"], row["score_b"], row["score_a"]),
            ):
                if pd.isna(team):
                    continue
                rec = records.setdefault(
                    team,
                    {
                        "team": team,
                        "world_cup_points_so_far": 0.0,
                        "world_cup_goal_diff_so_far": 0.0,
                        "world_cup_goals_for_so_far": 0.0,
                        "world_cup_goals_against_so_far": 0.0,
                        "world_cup_clean_sheets": 0.0,
                    },
                )
                rec["world_cup_goals_for_so_far"] += float(gf)
                rec["world_cup_goals_against_so_far"] += float(ga)
                rec["world_cup_goal_diff_so_far"] += float(gf) - float(ga)
                if ga == 0:
                    rec["world_cup_clean_sheets"] += 1
                if gf > ga:
                    rec["world_cup_points_so_far"] += 3
                elif gf == ga:
                    rec["world_cup_points_so_far"] += 1
        return pd.DataFrame(list(records.values()))
