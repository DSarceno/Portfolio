"""Bootstrap the historical match database from every configured source.

By default the script:

1. Loads the Kaggle ``results.csv`` dataset if present
   (``data/raw/kaggle/results.csv``).
2. Pulls the configured competitions from football-data.org (when an API key
   is set; otherwise it falls back to the manual CSV).
3. Loads any StatsBomb open data found locally.
4. Merges all frames into ``data/interim/matches_unified.csv``.

Use ``--start-year`` and ``--end-year`` to limit the historical window.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.results_collector import ResultsCollector
from src.utils.config import load_config
from src.utils.logging_config import get_logger, setup_logging


def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(description="Bootstrap historical match data")
    parser.add_argument(
        "--start-year",
        type=int,
        default=None,
        help="Minimum year to include (inclusive). Defaults to data.historical_start_year.",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=None,
        help="Maximum year to include (inclusive). Defaults to data.historical_end_year.",
    )
    parser.add_argument(
        "--competitions",
        nargs="+",
        default=None,
        help="football-data.org competition codes (default: WC EC CA).",
    )
    parser.add_argument(
        "--no-kaggle",
        action="store_true",
        help="Skip the Kaggle dataset even if present.",
    )
    parser.add_argument(
        "--no-friendlies",
        action="store_true",
        help="Drop friendly matches from Kaggle.",
    )
    parser.add_argument(
        "--no-qualifications",
        action="store_true",
        help="Drop qualification matches from Kaggle.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace the existing canonical table instead of merging.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the bootstrap.

    Returns:
        Process exit code (0 on success).
    """
    setup_logging(log_file="logs/pipeline/bootstrap_historical.log")
    logger = get_logger(__name__)
    config = load_config()
    args = parse_args()

    start_year = args.start_year or config.get("data.historical_start_year", 2014)
    end_year = args.end_year or config.get("data.historical_end_year", 2026)
    competitions = args.competitions or ["WC", "EC", "CA"]

    logger.info(
        "Bootstrapping historical data: years %s-%s, competitions=%s, include_kaggle=%s, "
        "include_friendlies=%s, include_qualifications=%s, merge=%s",
        start_year,
        end_year,
        competitions,
        not args.no_kaggle,
        not args.no_friendlies,
        not args.no_qualifications,
        not args.replace,
    )

    collector = ResultsCollector()
    df = collector.collect_all(
        competitions=competitions,
        start_year=start_year,
        end_year=end_year,
        include_kaggle=not args.no_kaggle,
        include_friendlies=not args.no_friendlies,
        include_qualifications=not args.no_qualifications,
        merge_with_existing=not args.replace,
    )
    logger.info("Bootstrap complete: %d matches in canonical table", len(df))

    if not df.empty:
        import pandas as pd

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        per_year = df.groupby(df["date"].dt.year).size().to_dict()
        per_comp = df["competition"].value_counts().to_dict()
        logger.info("Per-year counts: %s", per_year)
        logger.info("Per-competition counts: %s", per_comp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
