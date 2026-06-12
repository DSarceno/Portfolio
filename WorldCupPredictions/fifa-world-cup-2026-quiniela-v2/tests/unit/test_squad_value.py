"""Tests for the squad-value source and features (A.2)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.squad_value_client import (
    build_citizenship_snapshot,
    canonical_team_name,
)
from src.features.squad_value_features import compute_squad_value_features


def test_canonical_team_name_maps_known_aliases() -> None:
    assert canonical_team_name("Cote d'Ivoire") == "Ivory Coast"
    assert canonical_team_name("Korea, South") == "South Korea"
    assert canonical_team_name("Czech Republic") == "Czechia"
    # Unmapped names pass through unchanged.
    assert canonical_team_name("Brazil") == "Brazil"


def test_canonical_team_name_is_accent_insensitive() -> None:
    # The accented and ascii spellings must both resolve to Turkey.
    assert canonical_team_name("Türkiye") == "Turkey"
    assert canonical_team_name("Turkiye") == "Turkey"


def _valuations() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": [1, 1, 2, 3, 4],
            "date": pd.to_datetime(
                ["2017-01-01", "2021-01-01", "2021-06-01", "2021-06-01", "2030-01-01"]
            ),
            "market_value_in_eur": [10e6, 50e6, 30e6, 20e6, 99e6],
        }
    )


def _players() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": [1, 2, 3, 4],
            "country_of_citizenship": ["Brazil", "Brazil", "Brazil", "Brazil"],
        }
    )


def test_build_citizenship_snapshot_uses_latest_value_before_cutoff() -> None:
    snap = build_citizenship_snapshot(_valuations(), _players(), "2022-06-01")
    row = snap[snap["team"] == "Brazil"].iloc[0]
    # Player 1 uses its 2021 value (50), players 2+3 (30+20), player 4 is in the
    # future (2030) and must be excluded.
    assert row["squad_value_total_meur"] == 100.0
    assert row["squad_value_top11_meur"] == 100.0
    assert row["source"] == "kaggle_citizenship"


def test_build_citizenship_snapshot_excludes_future_only_players() -> None:
    # As of 2018, only player 1 (value 10 in 2017) qualifies.
    snap = build_citizenship_snapshot(_valuations(), _players(), "2018-06-01")
    row = snap[snap["team"] == "Brazil"].iloc[0]
    assert row["squad_value_total_meur"] == 10.0


def _snapshots() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "team": ["Brazil", "Brazil", "Argentina"],
            "as_of_date": ["2018-06-01", "2022-06-01", "2022-06-01"],
            "squad_value_total_meur": [1000.0, 1200.0, 800.0],
            "squad_value_top11_meur": [700.0, 800.0, 600.0],
            "source": ["kaggle_citizenship"] * 3,
        }
    )


def test_compute_squad_value_features_asof_join() -> None:
    matches = pd.DataFrame(
        {
            "date": pd.to_datetime(["2019-07-01", "2023-07-01"]),
            "team_a": ["Brazil", "Brazil"],
            "team_b": ["Argentina", "Argentina"],
        }
    )
    out = compute_squad_value_features(matches, _snapshots())
    assert "squad_value_top11_diff" in out.columns
    # 2019 match: Brazil uses 2018 snapshot (700), Argentina has no 2018 snapshot
    # -> filled with confederation median, but the column is finite either way.
    assert np.isfinite(out["squad_value_top11_diff"]).all()
    # 2023 match: both have 2022 snapshots; Brazil (800) > Argentina (600) -> diff > 0.
    second = out.iloc[1]
    assert second["squad_value_top11_diff"] > 0


def test_compute_squad_value_features_noop_without_data() -> None:
    matches = pd.DataFrame(
        {"date": pd.to_datetime(["2019-07-01"]), "team_a": ["Brazil"], "team_b": ["Chile"]}
    )
    out = compute_squad_value_features(matches, None)
    assert "squad_value_top11_diff" not in out.columns
    out_empty = compute_squad_value_features(matches, pd.DataFrame())
    assert "squad_value_top11_diff" not in out_empty.columns
