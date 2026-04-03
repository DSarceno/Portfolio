#!/usr/bin/env python3
"""Predict F1 race results using a trained model."""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.prediction.predictor import RacePredictor
from src.utils.config import Config
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="F1 Race Predictor - Race Prediction"
    )
    parser.add_argument(
        "--season",
        type=int,
        required=True,
        help="F1 season year (e.g. 2025)",
    )
    parser.add_argument(
        "--round",
        type=int,
        required=True,
        dest="round_num",
        help="Race round number (e.g. 1)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="best_model",
        help="Model filename stem (default: best_model)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/prediction.json",
        help="Output file path (default: outputs/prediction.json)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "csv", "table"],
        default="table",
        help="Output format (default: table)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to main config file",
    )
    return parser.parse_args()


def format_table_output(predictions_df: pd.DataFrame) -> str:
    """Format predictions as a pretty-printed table string.

    Args:
        predictions_df: Predictions DataFrame with standard columns.

    Returns:
        Formatted string table.
    """
    lines = [
        "",
        "=" * 65,
        f"{'Pos':>4}  {'Driver':<8}  {'Team':<25}  {'Confidence':>10}",
        "-" * 65,
    ]
    for _, row in predictions_df.iterrows():
        lines.append(
            f"{row['predicted_position']:>4}  "
            f"{str(row['driver']):<8}  "
            f"{str(row['team']):<25}  "
            f"{float(row['confidence']):>10.1%}"
        )
    lines.append("=" * 65)
    return "\n".join(lines)


def main() -> None:
    """Generate and output race predictions.

    Steps:
        1. Load config and model
        2. Fetch race data and build features
        3. Generate position predictions
        4. Output in requested format
    """
    args = parse_args()
    setup_logging(
        level="INFO", log_file="prediction.log", log_dir="logs/prediction"
    )
    logger.info(
        "Predicting: season=%d round=%d", args.season, args.round_num
    )

    try:
        config = Config(config_path=args.config)
        predictor = RacePredictor(config=config._config)

        model_path = Path(
            config.get("paths.models_dir", "models")
        ) / f"{args.model}.pkl"

        if not model_path.exists():
            logger.error(
                "Model not found: %s. Run train_model.py first.", model_path
            )
            sys.exit(1)

        predictor.load_model(str(model_path))
        predictions_df = predictor.predict_race(args.season, args.round_num)

        if args.format == "table":
            print(format_table_output(predictions_df))
        elif args.format == "json":
            records = predictions_df.to_dict(orient="records")
            print(json.dumps(records, indent=2))
        elif args.format == "csv":
            print(predictions_df.to_csv(index=False))

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        predictions_df.to_json(output_path, orient="records", indent=2)
        logger.info("Predictions saved to %s", output_path)

    except Exception as e:
        logger.error("Prediction failed: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
