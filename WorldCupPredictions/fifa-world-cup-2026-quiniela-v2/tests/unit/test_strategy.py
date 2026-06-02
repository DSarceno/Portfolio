"""Tests for the pick-optimizer and quiniela strategy modules."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.ensemble.pick_optimizer import PickOptimizer
from src.prediction.quiniela_strategy import QuinielaStrategy


def test_pick_optimizer_all_profiles() -> None:
    optimizer = PickOptimizer()
    proba = np.array([0.45, 0.30, 0.25])
    for profile in ("safe", "balanced", "aggressive", "contrarian"):
        label, conf, rationale = optimizer.pick(proba, risk_profile=profile)
        assert label in {"H", "D", "A"}
        assert 0 <= conf <= 1
        assert isinstance(rationale, str)


def test_pick_optimizer_aggressive_triggers_upset() -> None:
    optimizer = PickOptimizer()
    proba = np.array([0.50, 0.20, 0.30])
    label, _, rationale = optimizer.pick(
        proba,
        risk_profile="aggressive",
        upset_window_score=0.8,
        favorite_fragility_score=0.7,
    )
    assert label == "A"
    assert "upset" in rationale


def test_quiniela_strategy_generates_all_profiles() -> None:
    strategy = QuinielaStrategy()
    probs = pd.DataFrame(
        [
            {
                "team_a": "Alpha",
                "team_b": "Beta",
                "p_home": 0.5,
                "p_draw": 0.3,
                "p_away": 0.2,
                "upset_window_score": 0.4,
                "favorite_fragility_score": 0.3,
                "public_bias_proxy": 0.1,
            }
        ]
    )
    sheets = strategy.generate_all(probs, include_scoreline=False)
    assert set(sheets.keys()) == {"safe", "balanced", "aggressive", "contrarian"}
    for sheet in sheets.values():
        assert {"pick", "confidence", "rationale"}.issubset(sheet.columns)
