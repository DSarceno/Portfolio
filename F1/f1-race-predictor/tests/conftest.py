"""Shared pytest fixtures for F1 Race Predictor tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_race_data() -> pd.DataFrame:
    """60-row race DataFrame (3 races x 20 drivers).

    Returns:
        DataFrame with standard race result columns.
    """
    drivers = [
        "VER", "HAM", "LEC", "NOR", "PIA", "SAI", "RUS", "ALO", "STR", "OCO",
        "ALB", "BOT", "ZHO", "HUL", "MAG", "TSU", "RIC", "LAW", "SAR", "BEA",
    ]
    teams = [
        "Red Bull", "Mercedes", "Ferrari", "McLaren", "McLaren",
        "Ferrari", "Mercedes", "Aston Martin", "Aston Martin", "Alpine",
        "Williams", "Alfa Romeo", "Alfa Romeo", "Haas", "Haas",
        "RB", "RB", "RB", "Williams", "Haas",
    ]

    rows = []
    for year in [2022, 2023, 2024]:
        for round_num in [1]:
            for i, (drv, team) in enumerate(zip(drivers, teams)):
                rows.append(
                    {
                        "Year": year,
                        "Round": round_num,
                        "Abbreviation": drv,
                        "TeamName": team,
                        "GridPosition": i + 1,
                        "Position": i + 1,
                        "Points": max(0, 26 - i * 2),
                        "Status": "Finished",
                        "Location": "Bahrain",
                    }
                )
    df = pd.DataFrame(rows)
    # Introduce a few DNFs
    df.loc[df["Abbreviation"] == "SAR", "Status"] = "DNF"
    df.loc[df["Abbreviation"] == "SAR", "Position"] = 20
    return df


@pytest.fixture
def sample_lap_data() -> pd.DataFrame:
    """Synthetic lap-level DataFrame.

    Returns:
        DataFrame with lap time and telemetry columns.
    """
    np.random.seed(42)
    drivers = ["VER", "HAM", "LEC", "NOR"]
    rows = []
    for driver in drivers:
        base_time = pd.Timedelta(seconds=90 + np.random.uniform(-2, 2))
        for lap in range(1, 21):
            degradation = pd.Timedelta(seconds=lap * 0.05)
            rows.append(
                {
                    "Driver": driver,
                    "LapNumber": lap,
                    "LapTime": base_time + degradation,
                    "SpeedST": 320 + np.random.uniform(-10, 10),
                    "SpeedFL": 200 + np.random.uniform(-10, 10),
                    "SpeedI1": 260 + np.random.uniform(-10, 10),
                    "Throttle": np.random.uniform(0.6, 0.9),
                    "Brake": np.random.uniform(0.1, 0.3),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def sample_weather_data() -> pd.DataFrame:
    """Synthetic weather time-series DataFrame.

    Returns:
        DataFrame with weather measurement columns.
    """
    rows = [
        {
            "TrackTemp": 35.0 + i * 0.1,
            "AirTemp": 25.0,
            "Humidity": 52.0,
            "WindSpeed": 12.0,
            "Rainfall": 0.0,
        }
        for i in range(10)
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def mock_config() -> dict:
    """Return a minimal configuration dict.

    Returns:
        Dictionary matching config.yaml structure.
    """
    return {
        "data": {
            "start_season": 2020,
            "end_season": 2025,
            "cache_dir": "data/raw/fastf1_cache",
            "validation": {
                "max_null_percentage": 0.3,
                "min_laps_per_race": 10,
                "min_drivers_per_race": 10,
            },
        },
        "training": {
            "test_size": 0.2,
            "val_size": 0.15,
            "random_seed": 42,
        },
        "models": {
            "xgboost": {
                "enabled": True,
                "hyperparameters": {
                    "n_estimators": 10,
                    "max_depth": 3,
                    "learning_rate": 0.1,
                    "objective": "multi:softprob",
                    "num_class": 20,
                    "random_state": 42,
                    "n_jobs": 1,
                },
                "training": {
                    "early_stopping_rounds": 5,
                    "verbose": 0,
                },
            },
            "neural_net": {
                "enabled": True,
                "architecture": {
                    "hidden_layers": [32, 16],
                    "output_classes": 20,
                    "dropout_rate": 0.1,
                    "batch_norm": False,
                },
                "hyperparameters": {
                    "learning_rate": 0.01,
                    "weight_decay": 0.0,
                },
                "training": {
                    "epochs": 2,
                    "batch_size": 16,
                    "early_stopping": 5,
                    "device": "cpu",
                },
            },
            "baseline": {
                "enabled": True,
                "parameters": {
                    "use_historical_delta": True,
                    "dnf_threshold": 0.2,
                    "max_position_change": 5,
                },
            },
        },
        "paths": {
            "models_dir": "models",
            "outputs_dir": "outputs",
        },
    }
