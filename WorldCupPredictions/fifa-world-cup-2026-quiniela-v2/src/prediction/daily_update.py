"""Daily-update orchestration: ingest results, refresh ratings, regenerate picks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from src.data.data_loader import DataLoader
from src.data.tournament_updater import TournamentUpdater
from src.ratings.rating_ensemble import RatingEnsemble
from src.utils.io import ensure_dir, save_csv
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class DailyUpdateResult:
    """Summary of a single daily-update run."""

    matches_appended: int
    composite_table_path: str
    state_path: str


class DailyUpdater:
    """Orchestrates the tournament-update workflow."""

    def __init__(
        self,
        loader: Optional[DataLoader] = None,
        updater: Optional[TournamentUpdater] = None,
        outputs_dir: str | Path = "outputs",
        tournament_year: int = 2026,
    ) -> None:
        """Initialize the updater.

        Args:
            loader: Optional pre-built :class:`DataLoader`.
            updater: Optional pre-built :class:`TournamentUpdater`.
            outputs_dir: Directory used for output artifacts.
            tournament_year: Tournament year filter.
        """
        self.loader = loader or DataLoader()
        self.updater = updater or TournamentUpdater(loader=self.loader)
        self.outputs_dir = ensure_dir(outputs_dir)
        self.tournament_year = tournament_year

    def update(self, new_matches: Optional[pd.DataFrame] = None) -> DailyUpdateResult:
        """Append new matches (if any), refit ratings and persist a state snapshot.

        Args:
            new_matches: Optional DataFrame with matches to append.

        Returns:
            :class:`DailyUpdateResult`.
        """
        if new_matches is not None and not new_matches.empty:
            summary = self.updater.append_results(new_matches)
            matches_appended = summary.matches_appended
        else:
            matches_appended = 0

        matches = self.loader.load_matches()
        ensemble = RatingEnsemble().fit(matches)
        composite = ensemble.composite_table()
        composite_path = save_csv(composite, self.outputs_dir / "diagnostics" / "composite_ratings.csv")

        state = self.updater.world_cup_state(self.tournament_year)
        state_path = save_csv(state, self.outputs_dir / "diagnostics" / "tournament_state.csv")

        logger.info(
            "Daily update completed: appended=%d, composite=%s, state=%s",
            matches_appended,
            composite_path,
            state_path,
        )
        return DailyUpdateResult(
            matches_appended=matches_appended,
            composite_table_path=str(composite_path),
            state_path=str(state_path),
        )
