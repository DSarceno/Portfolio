#!/usr/bin/env python3
"""Train F1 race prediction models."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.models.model_factory import ModelFactory
from src.training.evaluator import Evaluator
from src.training.trainer import Trainer
from src.utils.config import Config
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="F1 Race Predictor - Model Training"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["xgboost", "neural_net", "baseline", "all"],
        default="all",
        help="Model to train (default: all)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to main config file",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Directory with processed train/val/test splits",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save trained models",
    )
    return parser.parse_args()


def load_splits(
    data_dir: str,
) -> tuple:
    """Load train/val/test splits from parquet files.

    Args:
        data_dir: Directory containing split parquet files.

    Returns:
        Tuple of (train_df, val_df, test_df).

    Raises:
        FileNotFoundError: If split files are missing.
    """
    path = Path(data_dir)
    train = pd.read_parquet(path / "train.parquet")
    val = pd.read_parquet(path / "val.parquet")
    test = pd.read_parquet(path / "test.parquet")
    logger.info(
        "Loaded splits: train=%d val=%d test=%d",
        len(train),
        len(val),
        len(test),
    )
    return train, val, test


def prepare_arrays(
    df: pd.DataFrame, target_col: str = "Position"
) -> tuple:
    """Extract feature matrix and target array from DataFrame.

    Args:
        df: Feature DataFrame.
        target_col: Column name for target labels.

    Returns:
        Tuple of (X, y) numpy arrays.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not in DataFrame")
    y = pd.to_numeric(df[target_col], errors="coerce").fillna(10).values
    X = df.select_dtypes(include=[np.number]).drop(
        columns=[target_col], errors="ignore"
    ).fillna(0).values
    return X, y


def print_results_table(results: dict) -> None:
    """Print evaluation results as a formatted table.

    Args:
        results: Dict of model_name -> metrics dict.
    """
    print("\n" + "=" * 70)
    print(f"{'Model':<15} {'MAE':>8} {'RMSE':>8} {'Top-3':>8} {'Top-5':>8}")
    print("-" * 70)
    for model_name, metrics in results.items():
        if not metrics:
            continue
        print(
            f"{model_name:<15} "
            f"{metrics.get('mae', float('nan')):>8.3f} "
            f"{metrics.get('rmse', float('nan')):>8.3f} "
            f"{metrics.get('top_3_accuracy', float('nan')):>8.3f} "
            f"{metrics.get('top_5_accuracy', float('nan')):>8.3f}"
        )
    print("=" * 70 + "\n")


def main() -> None:
    """Train and evaluate F1 prediction models.

    Steps:
        1. Load processed data splits
        2. Train specified model(s)
        3. Evaluate on test set
        4. Save best model
        5. Print results table
    """
    args = parse_args()
    setup_logging(level="INFO", log_file="training.log", log_dir="logs/training")
    logger.info("Starting training: model=%s", args.model)

    try:
        config = Config(config_path=args.config)
        config._config["paths"] = config._config.get("paths", {})
        config._config["paths"]["models_dir"] = args.output_dir

        train_df, val_df, test_df = load_splits(args.data_dir)
        X_train, y_train = prepare_arrays(train_df)
        X_val, y_val = prepare_arrays(val_df)
        X_test, y_test = prepare_arrays(test_df)

        trainer = Trainer(config=config._config)
        evaluator = Evaluator(config=config._config)

        if args.model == "all":
            results = trainer.train_all_models(X_train, y_train, X_val, y_val)
        else:
            model = ModelFactory.create(args.model, config._config)
            metrics = trainer.train_single_model(
                model, X_train, y_train, X_val, y_val
            )
            results = {args.model: metrics}

        logger.info("Evaluating on test set...")
        test_results = {}
        for name, _ in results.items():
            if name in trainer._trained_models:
                test_metrics = evaluator.evaluate(
                    trainer._trained_models[name], X_test, y_test
                )
                test_results[name] = test_metrics
                trainer.save_model(trainer._trained_models[name], name)

        print_results_table(test_results)
        evaluator.generate_evaluation_report(test_results, "outputs/evaluation_report.json")

        if results:
            best_name, best_model = trainer.get_best_model(results)
            trainer.save_model(best_model, "best_model")
            logger.info("Best model saved: %s", best_name)

    except FileNotFoundError as e:
        logger.error("Data not found: %s. Run run_pipeline.py first.", e)
        sys.exit(1)
    except Exception as e:
        logger.error("Training failed: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
