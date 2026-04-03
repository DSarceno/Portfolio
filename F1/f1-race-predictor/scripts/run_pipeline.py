#!/usr/bin/env python3
"""Run the complete F1 data extraction and processing pipeline."""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_loader import DataLoader
from src.data.data_splitter import DataSplitter
from src.data.data_validator import DataValidator
from src.data.fastf1_client import FastF1Client
from src.features.build_features import FeatureBuilder
from src.utils.config import Config
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="F1 Race Predictor - Data Pipeline"
    )
    parser.add_argument(
        "--start-season",
        type=int,
        default=2020,
        help="First season year to include (default: 2020)",
    )
    parser.add_argument(
        "--end-season",
        type=int,
        default=2025,
        help="Last season year to include (default: 2025)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to main config file",
    )
    parser.add_argument(
        "--force-reload",
        action="store_true",
        help="Force re-download even if cached data exists",
    )
    return parser.parse_args()


def main() -> None:
    """Execute the full data pipeline.

    Steps:
        1. Setup logging and config
        2. Fetch multi-season race data
        3. Validate the dataset
        4. Build feature matrix
        5. Temporal split into train/val/test
        6. Save processed splits to data/processed/
    """
    args = parse_args()
    setup_logging(level="INFO", log_file="pipeline.log", log_dir="logs/pipeline")
    logger.info(
        "Starting pipeline: seasons %d-%d", args.start_season, args.end_season
    )

    try:
        config = Config(config_path=args.config)
        data_cfg = config.data_config

        client = FastF1Client(
            cache_dir=data_cfg.get("cache_dir", "data/raw/fastf1_cache"),
            enabled=not args.force_reload,
        )
        loader = DataLoader(config=data_cfg, fastf1_client=client)

        logger.info("Loading data for seasons %d-%d...", args.start_season, args.end_season)
        raw_data = loader.load_multiple_seasons(args.start_season, args.end_season)

        if raw_data.empty:
            logger.error("No data loaded. Check your FastF1 connection.")
            sys.exit(1)

        logger.info("Loaded %d rows. Validating...", len(raw_data))
        validator = DataValidator(config=data_cfg)
        is_valid, errors = validator.validate(raw_data)
        if not is_valid:
            logger.warning("Validation issues: %s", errors)
        report = validator.generate_validation_report(raw_data)
        logger.info("Validation report: %s seasons, %s races",
                    report.get("seasons", []), report.get("races", 0))

        loader.save_raw_data(raw_data, "data/raw/all_seasons.parquet")

        logger.info("Building features...")
        feature_builder = FeatureBuilder(config=config._config)
        # For pipeline, build basic features from race data
        feature_data = {
            "race": raw_data,
            "laps": __import__("pandas").DataFrame(),
            "weather": __import__("pandas").DataFrame(),
            "historical": raw_data,
        }
        features_df = feature_builder.build_features(feature_data)
        logger.info("Feature matrix: %d rows x %d cols", *features_df.shape)

        splitter = DataSplitter(config=config._config)
        split_info = splitter.get_split_info(features_df)
        logger.info("Split info: %s", split_info)

        train, val, test = splitter.temporal_split(features_df)
        splitter.save_splits(train, val, test, "data/processed")

        logger.info(
            "Pipeline complete! Train=%d, Val=%d, Test=%d",
            len(train),
            len(val),
            len(test),
        )

    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error("Pipeline failed: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
