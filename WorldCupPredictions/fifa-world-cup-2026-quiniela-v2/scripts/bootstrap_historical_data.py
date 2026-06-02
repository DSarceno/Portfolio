"""Bootstrap the historical match database from every configured source."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.results_collector import ResultsCollector
from src.utils.config import load_config
from src.utils.logging_config import get_logger, setup_logging


def main() -> int:
    """Run the bootstrap.

    Returns:
        Process exit code (0 on success).
    """
    setup_logging(log_file="logs/pipeline/bootstrap_historical.log")
    logger = get_logger(__name__)
    config = load_config()

    competitions = ["WC", "EC", "CA"]
    logger.info("Bootstrapping historical data for competitions=%s", competitions)
    collector = ResultsCollector()
    df = collector.collect_all(competitions)
    logger.info("Bootstrap complete: %d matches in canonical table", len(df))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
