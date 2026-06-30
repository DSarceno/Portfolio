"""Tests for TournamentUpdater result ingestion (append vs replace modes)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.tournament_updater import TournamentUpdater


def _seed_canonical(interim_dir: Path) -> None:
    """Write a canonical table with an orphan manual row, a fixture and history."""
    existing = pd.DataFrame(
        [
            # Orphan: an earlier manual entry on the WRONG date (06-19). Not in the new CSV.
            {
                "date": "2026-06-19",
                "competition": "WC",
                "stage": "GROUP_STAGE",
                "group": "C",
                "team_a": "Brazil",
                "team_b": "Haiti",
                "score_a": 3.0,
                "score_b": 0.0,
                "neutral_venue": False,
                "source": "tournament_update",
            },
            # Scheduled fixture (unplayed) on the correct date.
            {
                "date": "2026-06-20",
                "competition": "WC",
                "stage": "GROUP_STAGE",
                "group": "C",
                "team_a": "Brazil",
                "team_b": "Haiti",
                "score_a": pd.NA,
                "score_b": pd.NA,
                "neutral_venue": False,
                "source": "football_data",
            },
            # Untouched historical row from a different source.
            {
                "date": "2016-06-08",
                "competition": "FRIENDLY",
                "stage": "historical",
                "group": "",
                "team_a": "Brazil",
                "team_b": "Haiti",
                "score_a": 7.0,
                "score_b": 1.0,
                "neutral_venue": True,
                "source": "kaggle_international",
            },
        ]
    )
    existing.to_csv(interim_dir / "matches_unified.csv", index=False)


def _new_csv() -> pd.DataFrame:
    """The corrected manual CSV: Brazil-Haiti on the right date (06-20)."""
    return pd.DataFrame(
        [
            {
                "date": "2026-06-20",
                "competition": "WC",
                "stage": "GROUP_STAGE",
                "group": "C",
                "team_a": "Brazil",
                "team_b": "Haiti",
                "score_a": 3,
                "score_b": 0,
                "neutral_venue": False,
            }
        ]
    )


def _make_updater(tmp_path: Path) -> TournamentUpdater:
    interim = tmp_path / "interim"
    updates = tmp_path / "updates"
    interim.mkdir()
    updates.mkdir()
    _seed_canonical(interim)
    return TournamentUpdater(updates_dir=updates, interim_dir=interim)


def test_replace_source_removes_orphans(tmp_path: Path) -> None:
    """replace_source=True makes the canonical table mirror the CSV exactly."""
    updater = _make_updater(tmp_path)

    updater.append_results(_new_csv(), replace_source=True)

    combined = updater.loader.load_matches()
    manual = combined[combined["source"] == "tournament_update"]

    # The 06-19 orphan must be gone.
    assert (combined["date"].dt.strftime("%Y-%m-%d") == "2026-06-19").sum() == 0
    # Exactly the CSV's manual rows survive as tournament_update.
    assert len(manual) == 1
    # The played result merged onto the 06-20 fixture.
    bh = combined[
        (combined["team_a"] == "Brazil")
        & (combined["team_b"] == "Haiti")
        & (combined["date"].dt.strftime("%Y-%m-%d") == "2026-06-20")
    ]
    assert len(bh) == 1
    assert float(bh.iloc[0]["score_a"]) == 3.0
    # Historical row from another source untouched.
    assert (combined["source"] == "kaggle_international").sum() == 1


def test_append_mode_keeps_orphans(tmp_path: Path) -> None:
    """Default (replace_source=False) keeps pre-existing rows (backward compatible)."""
    updater = _make_updater(tmp_path)

    updater.append_results(_new_csv())

    combined = updater.loader.load_matches()
    # The orphan 06-19 row survives in append mode.
    assert (combined["date"].dt.strftime("%Y-%m-%d") == "2026-06-19").sum() == 1
