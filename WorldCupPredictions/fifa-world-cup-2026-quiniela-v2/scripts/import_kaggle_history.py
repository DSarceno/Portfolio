"""Import historical match data from the Kaggle international results CSV.

The Kaggle dataset lives at:

    https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017

After downloading ``results.csv`` (and optionally ``shootouts.csv`` /
``goalscorers.csv``) place ``results.csv`` under ``data/raw/kaggle/`` and run::

    python scripts/import_kaggle_history.py --csv data/raw/kaggle/results.csv --start-year 2014
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.kaggle_results_client import KaggleResultsClient
from src.data.results_collector import ResultsCollector
from src.utils.config import load_config
from src.utils.logging_config import get_logger, setup_logging


def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(description="Import historical data from Kaggle CSV")
    parser.add_argument(
        "--csv",
        default="data/raw/kaggle/results.csv",
        help="Path to results.csv (default: data/raw/kaggle/results.csv)",
    )
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
        "--no-friendlies",
        action="store_true",
        help="Drop friendly matches.",
    )
    parser.add_argument(
        "--no-qualifications",
        action="store_true",
        help="Drop qualification matches.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace the existing canonical table instead of merging.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the Kaggle import."""
    setup_logging(log_file="logs/pipeline/import_kaggle_history.log")
    logger = get_logger(__name__)
    config = load_config()
    args = parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        logger.error(
            "Kaggle results.csv not found at %s. Download it from "
            "https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017",
            csv_path,
        )
        return 1

    start_year = args.start_year or config.get("data.historical_start_year", 2014)
    end_year = args.end_year or config.get("data.historical_end_year", 2026)

    collector = ResultsCollector(kaggle_client=KaggleResultsClient(csv_path=csv_path))
    df = collector.collect_kaggle_only(
        start_year=start_year,
        end_year=end_year,
        include_friendlies=not args.no_friendlies,
        include_qualifications=not args.no_qualifications,
        merge_with_existing=not args.replace,
    )
    logger.info(
        "Kaggle import complete: %d matches in canonical table (years %s-%s)",
        len(df),
        start_year,
        end_year,
    )
    if not df.empty:
        import pandas as pd

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        per_comp = df["competition"].value_counts().to_dict()
        logger.info("Per-competition counts: %s", per_comp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
