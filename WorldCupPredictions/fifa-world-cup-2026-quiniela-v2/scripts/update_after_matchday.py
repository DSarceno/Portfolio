"""Refresh ratings, features and predictions after a World Cup matchday."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.football_data_client import FootballDataClient
from src.data.results_collector import CANONICAL_COLUMNS
from src.prediction.daily_update import DailyUpdater
from src.utils.logging_config import get_logger, setup_logging


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Update World Cup after a matchday")
    parser.add_argument("--date", required=False, help="ISO date YYYY-MM-DD")
    parser.add_argument(
        "--manual-csv",
        required=False,
        help="Optional CSV path with new matches in canonical schema",
    )
    parser.add_argument(
        "--append-only",
        action="store_true",
        help=(
            "Disable the default 'purge + re-ingest' for --manual-csv. By default "
            "the manual CSV is treated as the authoritative record: all existing "
            "manual ('tournament_update') rows are replaced by the CSV, so edits "
            "(fixed dates/teams) propagate and no orphan rows survive. Use this "
            "flag to only upsert the CSV rows without purging (e.g. ingesting a "
            "partial CSV)."
        ),
    )
    return parser.parse_args()


def main() -> int:
    """Run the matchday update."""
    setup_logging(log_file="logs/updates/update_after_matchday.log")
    logger = get_logger(__name__)
    args = parse_args()

    new_matches = pd.DataFrame(columns=CANONICAL_COLUMNS)
    replace_source = False
    if args.manual_csv:
        new_matches = pd.read_csv(args.manual_csv)
        replace_source = not args.append_only
        logger.info(
            "Loaded %d matches from manual CSV %s (mode=%s)",
            len(new_matches),
            args.manual_csv,
            "append-only" if args.append_only else "replace (purge + re-ingest)",
        )
    elif args.date:
        logger.info("Attempting to fetch matches for %s from football-data.org", args.date)
        df = FootballDataClient().fetch_matches("WC", date_from=args.date, date_to=args.date)
        if not df.empty:
            new_matches = pd.DataFrame(
                {
                    "match_id": df.get("match_id"),
                    "date": df.get("date"),
                    "competition": df.get("competition"),
                    "season": df.get("season"),
                    "stage": df.get("stage"),
                    "team_a": df.get("home_team"),
                    "team_b": df.get("away_team"),
                    "score_a": df.get("home_score"),
                    "score_b": df.get("away_score"),
                    "neutral_venue": df.get("neutral_venue", True),
                    "host_country": df.get("venue", ""),
                    "source": "football_data_update",
                }
            )

    updater = DailyUpdater()
    result = updater.update(new_matches=new_matches, replace_source=replace_source)
    logger.info("Update complete: %s", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
