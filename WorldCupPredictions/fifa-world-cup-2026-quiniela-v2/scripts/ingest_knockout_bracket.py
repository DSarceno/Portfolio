"""Ingest the manual knockout bracket into the canonical match table.

Once the group stage is over the round-of-32 matchups are known and are filled
into ``data/raw/manual/wc2026_knockout_bracket.csv``. This script maps the
*filled* ties (both teams present) into the canonical table as unplayed WC
fixtures so the predictors (``predict_scorelines.py`` / ``predict_knockout.py``)
and the bracket notebook pick them up. Scores stay empty; results are appended
later through the normal matchday flow.

It is ingested under ``source="knockout_bracket"`` with ``replace_source=True``,
i.e. the idempotent purge-then-reingest path: re-running after you correct a
date (e.g. once football-data.org confirms the real UTC kickoff) or a matchup
rewrites the whole partition with **zero** orphan/duplicate rows.

Team names are validated against the canonical table *before* anything is
written: a name the model has never seen (a typo, or a display alias such as
"DR Congo" vs the canonical "Congo DR") would silently score 33/33/33, so a
mismatch aborts the ingest loudly instead.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.data_loader import DataLoader
from src.data.tournament_updater import TournamentUpdater
from src.utils.config import load_config
from src.utils.logging_config import get_logger, setup_logging

SOURCE_TAG = "knockout_bracket"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Ingest the manual knockout bracket")
    parser.add_argument(
        "--bracket-csv",
        type=str,
        default="data/raw/manual/wc2026_knockout_bracket.csv",
        help="Path to the manual knockout-bracket CSV.",
    )
    parser.add_argument(
        "--append-only",
        action="store_true",
        help=(
            "Use a plain upsert instead of the default purge-then-reingest. "
            "Leaves stale rows behind if a previously ingested date/matchup was "
            "edited; only use for a partial top-up."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print what would be ingested without writing anything.",
    )
    return parser.parse_args()


def _filled(value: object) -> bool:
    """Return True if a CSV cell carries a real (non-empty) value."""
    return not (pd.isna(value) or str(value).strip() == "")


def build_canonical_rows(bracket: pd.DataFrame, tournament_year: int) -> pd.DataFrame:
    """Map filled bracket ties to canonical fixture rows (no scores)."""
    filled = bracket[
        bracket["team_a"].apply(_filled) & bracket["team_b"].apply(_filled)
    ].copy()
    rows: list[dict[str, object]] = []
    for r in filled.itertuples():
        rows.append(
            {
                "match_id": str(r.match_id),
                "date": str(r.date).strip() if _filled(r.date) else pd.NA,
                "competition": "WC",
                "season": tournament_year,
                "stage": str(r.stage).strip(),
                "group": "",
                "team_a": str(r.team_a).strip(),
                "team_b": str(r.team_b).strip(),
                "score_a": pd.NA,
                "score_b": pd.NA,
                "neutral_venue": bool(r.neutral_venue),
                "host_country": pd.NA,
            }
        )
    return pd.DataFrame(rows)


def _date_key(value: object) -> str:
    """Normalise a date to a ``YYYY-MM-DD`` string for key comparison."""
    ts = pd.to_datetime(value, errors="coerce")
    return "" if pd.isna(ts) else ts.strftime("%Y-%m-%d")


def drop_already_played(rows: pd.DataFrame, loader: DataLoader) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split *rows* into (still-unplayed, already-played) ties.

    A bracket fixture is "already played" when the canonical table holds a row
    with a non-null score on the **same (date, team_a, team_b) key** (either
    orientation). Re-ingesting a scoreless fixture for such a tie would blank
    out a result that was entered through the matchday flow, so those rows are
    held back. Matching on the full date key (not just the team pair) avoids
    mistaking a group-stage meeting between the same two teams for the tie.
    """
    canonical = loader.load_matches()
    if canonical.empty or "score_a" not in canonical.columns:
        return rows, rows.iloc[0:0]

    played = canonical[canonical["score_a"].notna()]
    played_keys: set[tuple[str, str, str]] = set()
    for r in played.itertuples():
        d, a, b = _date_key(r.date), str(r.team_a), str(r.team_b)
        played_keys.add((d, a, b))
        played_keys.add((d, b, a))

    def _is_played(r: pd.Series) -> bool:
        return (_date_key(r["date"]), str(r["team_a"]), str(r["team_b"])) in played_keys

    mask_played = rows.apply(_is_played, axis=1)
    return rows[~mask_played].copy(), rows[mask_played].copy()


def validate_team_names(rows: pd.DataFrame, loader: DataLoader) -> list[str]:
    """Return bracket team names absent from the canonical table.

    A name the model has never seen has no ratings/features and would score a
    flat 33/33/33, so the caller should abort if this is non-empty.
    """
    canonical = loader.load_matches()
    known = set(
        pd.concat([canonical["team_a"], canonical["team_b"]]).dropna().astype(str)
    )
    used = set(pd.concat([rows["team_a"], rows["team_b"]]).dropna().astype(str))
    return sorted(used - known)


def main() -> int:
    """Ingest the filled knockout ties into the canonical table."""
    setup_logging(log_file="logs/data/ingest_knockout_bracket.log")
    logger = get_logger(__name__)
    config = load_config()
    args = parse_args()
    tournament_year = int(config.get("tournament.year", 2026))

    path = Path(args.bracket_csv)
    if not path.exists():
        logger.error("Knockout-bracket CSV not found: %s", path)
        return 1

    bracket = pd.read_csv(path)
    rows = build_canonical_rows(bracket, tournament_year)
    if rows.empty:
        logger.warning("No filled ties found in %s; nothing to ingest", path)
        return 0

    loader = DataLoader()

    # Never overwrite a tie that already has a result (entered via the matchday
    # flow): re-ingesting a scoreless fixture for it would wipe the score.
    rows, played = drop_already_played(rows, loader)
    if not played.empty:
        logger.info(
            "Skipping %d already-played tie(s) so their results are preserved: %s",
            len(played),
            ", ".join(f"{r.team_a} vs {r.team_b}" for r in played.itertuples()),
        )
    if rows.empty:
        logger.warning("All filled ties are already played; nothing to (re)ingest.")
        return 0

    unknown = validate_team_names(rows, loader)
    if unknown:
        logger.error(
            "These bracket team names are not in the canonical table and would "
            "score 33/33/33: %s. Fix the names in %s to match the canonical "
            "spelling before ingesting.",
            unknown,
            path,
        )
        return 1

    logger.info(
        "Prepared %d knockout fixtures (%s)",
        len(rows),
        ", ".join(sorted(rows["stage"].astype(str).unique())),
    )
    for r in rows.itertuples():
        date_str = "date TBD" if pd.isna(r.date) else str(r.date)
        logger.info("  %-8s %-18s %s  %s vs %s", r.match_id, r.stage, date_str, r.team_a, r.team_b)

    if args.dry_run:
        logger.info("Dry run: no changes written.")
        return 0

    updater = TournamentUpdater(loader=loader)
    summary = updater.append_results(
        rows,
        source=SOURCE_TAG,
        replace_source=not args.append_only,
    )
    logger.info(
        "Ingested knockout bracket: %d net new row(s); canonical table now has %d rows (snapshot: %s)",
        summary.matches_appended,
        summary.matches_total,
        summary.snapshot_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
