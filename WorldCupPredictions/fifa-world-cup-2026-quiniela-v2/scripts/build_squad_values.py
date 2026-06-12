"""Build the canonical squad-value snapshot table (A.2).

Aggregates Kaggle ``player-scores`` valuations into dated national-squad-value
snapshots (citizenship proxy, same method for every date) and merges the
optional manual override on top. Output feeds the squad-value features.

    python scripts/build_squad_values.py \
        --as-of-dates 2018-06-01 2022-06-01 2026-06-01

Idempotent: reads local Kaggle files, never downloads. Re-run only when the
Kaggle dump or the manual override changes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.squad_value_client import (
    CANONICAL_SQUAD_VALUE_COLUMNS,
    SQUAD_VALUES_PATH,
    build_citizenship_snapshot,
)
from src.utils.io import ensure_dir, save_csv
from src.utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)

DEFAULT_KAGGLE_DIR = "data/raw/kaggle/players-scores"
DEFAULT_MANUAL = "data/raw/squad_values/squad_values_2026_manual.csv"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build squad-value snapshots")
    parser.add_argument(
        "--as-of-dates",
        nargs="+",
        default=["2018-06-01", "2022-06-01", "2026-06-01"],
        help="Snapshot cut-off dates (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--kaggle-dir",
        default=DEFAULT_KAGGLE_DIR,
        help="Directory with players.csv and player_valuations.csv.",
    )
    parser.add_argument(
        "--manual",
        default=DEFAULT_MANUAL,
        help="Optional manual override CSV (same schema; non-empty rows win).",
    )
    parser.add_argument(
        "--output",
        default=SQUAD_VALUES_PATH,
        help="Output canonical snapshot table.",
    )
    return parser.parse_args()


def _load_manual_override(path: str) -> pd.DataFrame:
    """Load the manual override, keeping only rows with at least one value."""
    p = Path(path)
    if not p.exists():
        logger.info("No manual override at %s; skipping", path)
        return pd.DataFrame(columns=CANONICAL_SQUAD_VALUE_COLUMNS)
    df = pd.read_csv(p)
    for col in ("squad_value_total_meur", "squad_value_top11_meur"):
        if col not in df.columns:
            df[col] = pd.NA
        df[col] = pd.to_numeric(df[col], errors="coerce")
    has_value = df["squad_value_total_meur"].notna() | df["squad_value_top11_meur"].notna()
    df = df[has_value].copy()
    if df.empty:
        logger.info("Manual override present but empty; skipping")
        return pd.DataFrame(columns=CANONICAL_SQUAD_VALUE_COLUMNS)
    df["as_of_date"] = pd.to_datetime(df["as_of_date"], errors="coerce").dt.date.astype(str)
    df["source"] = "manual"
    logger.info("Manual override: %d non-empty rows", len(df))
    return df[CANONICAL_SQUAD_VALUE_COLUMNS]


def main() -> int:
    """Build and persist the squad-value snapshots."""
    setup_logging(log_file="logs/pipeline/build_squad_values.log")
    logger_ = get_logger(__name__)
    args = parse_args()

    kaggle = Path(args.kaggle_dir)
    valuations_path = kaggle / "player_valuations.csv"
    players_path = kaggle / "players.csv"
    if not valuations_path.exists() or not players_path.exists():
        logger_.error(
            "Missing Kaggle files in %s (need players.csv + player_valuations.csv)",
            args.kaggle_dir,
        )
        return 1

    logger_.info("Loading Kaggle player-scores from %s", args.kaggle_dir)
    valuations = pd.read_csv(
        valuations_path,
        usecols=["player_id", "date", "market_value_in_eur"],
        parse_dates=["date"],
    )
    players = pd.read_csv(players_path, usecols=["player_id", "country_of_citizenship"])

    snapshots = []
    for date in args.as_of_dates:
        snap = build_citizenship_snapshot(valuations, players, date)
        logger_.info("Snapshot %s: %d teams", date, len(snap))
        snapshots.append(snap)

    derived = (
        pd.concat(snapshots, ignore_index=True)
        if snapshots
        else pd.DataFrame(columns=CANONICAL_SQUAD_VALUE_COLUMNS)
    )

    manual = _load_manual_override(args.manual)
    # Manual rows win: append after derived, drop duplicate (team, as_of_date)
    # keeping the last (manual) occurrence. Skip the concat entirely when the
    # override is empty to avoid pandas' all-NA concat warning.
    combined = derived if manual.empty else pd.concat([derived, manual], ignore_index=True)
    combined["as_of_date"] = combined["as_of_date"].astype(str)
    combined = (
        combined.drop_duplicates(subset=["team", "as_of_date"], keep="last")
        .sort_values(["team", "as_of_date"])
        .reset_index(drop=True)
    )

    ensure_dir(Path(args.output).parent)
    save_csv(combined, args.output)
    logger_.info(
        "Wrote %s: %d snapshots across %d teams (%d manual overrides)",
        args.output,
        len(combined),
        combined["team"].nunique(),
        int((combined["source"] == "manual").sum()),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
