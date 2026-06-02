"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def synthetic_matches() -> pd.DataFrame:
    """Return a deterministic synthetic match table."""
    rng = np.random.default_rng(42)
    teams = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta"]
    rows: list[dict] = []
    dates = pd.date_range("2022-01-01", periods=120, freq="3D")
    for i, day in enumerate(dates):
        a, b = rng.choice(teams, size=2, replace=False)
        rows.append(
            {
                "match_id": i,
                "date": day,
                "competition": "WC" if i % 3 == 0 else "FRIENDLY",
                "season": "2022",
                "stage": "group" if i % 4 != 0 else "round_of_16",
                "team_a": a,
                "team_b": b,
                "score_a": int(rng.poisson(1.4)),
                "score_b": int(rng.poisson(1.1)),
                "neutral_venue": True,
                "host_country": "Synthetic",
                "source": "test",
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def synthetic_fixtures() -> pd.DataFrame:
    """Return a 3-group, 3-team group-stage fixture set."""
    return pd.DataFrame(
        [
            {"group": "A", "team_a": "Alpha", "team_b": "Beta"},
            {"group": "A", "team_a": "Beta", "team_b": "Gamma"},
            {"group": "A", "team_a": "Alpha", "team_b": "Gamma"},
            {"group": "B", "team_a": "Delta", "team_b": "Epsilon"},
            {"group": "B", "team_a": "Epsilon", "team_b": "Zeta"},
            {"group": "B", "team_a": "Delta", "team_b": "Zeta"},
            {"group": "C", "team_a": "Eta", "team_b": "Theta"},
            {"group": "C", "team_a": "Theta", "team_b": "Iota"},
            {"group": "C", "team_a": "Eta", "team_b": "Iota"},
        ]
    )
