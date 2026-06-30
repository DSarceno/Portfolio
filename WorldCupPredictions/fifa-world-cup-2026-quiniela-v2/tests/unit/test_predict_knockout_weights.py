"""Regression test: predict_knockout must blend with the configured weights.

The knockout prediction script once built its ``MatchPredictor`` without passing
a blender, so it silently fell back to ``BlendWeights()`` defaults (multinomial
0.20) instead of the config — which still carried the *degenerate* multinomial
into the live ``knockout_predictions.csv`` even though the config had set its
weight to 0.0. This pins that the script reads the blend weights from config.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

predict_knockout = importlib.import_module("predict_knockout")


class _SpyPredictor:
    """Captures the blender it was constructed with; emits a trivial frame."""

    last_blender = None

    def __init__(self, *_, blender=None, **__) -> None:
        type(self).last_blender = blender

    def predict_matches(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {"p_home": [0.4] * len(fixtures), "p_draw": [0.3] * len(fixtures),
             "p_away": [0.3] * len(fixtures)}
        )


def test_predict_knockout_uses_config_blend_weights(monkeypatch) -> None:
    from src.ensemble.blender import BlendWeights

    fixtures = pd.DataFrame(
        {
            "date": ["2026-07-01"],
            "stage": ["ROUND_OF_32"],
            "team_a": ["England"],
            "team_b": ["Panama"],
            "outcome": [float("nan")],
        }
    )
    monkeypatch.setattr(predict_knockout, "build_inference_feature_matrix", lambda **_: fixtures)
    monkeypatch.setattr(predict_knockout, "filter_upcoming", lambda fm, **_: fm)
    monkeypatch.setattr(predict_knockout.BaseOutcomeModel, "load", staticmethod(lambda *_: None))
    monkeypatch.setattr(predict_knockout, "load_pickle", lambda *_: None)
    monkeypatch.setattr(predict_knockout, "save_csv", lambda *a, **k: None)
    monkeypatch.setattr(predict_knockout, "MatchPredictor", _SpyPredictor)

    # Sentinel weights distinct from the BlendWeights() defaults: if the script
    # builds its blender from config (the fix), the predictor must receive *these*;
    # if it falls back to the default constructor (the bug), it would not.
    sentinel = BlendWeights(ratings=0.1, multinomial=0.07, xgboost=0.4, poisson=0.43)
    monkeypatch.setattr(predict_knockout.BlendWeights, "from_config", staticmethod(lambda _cfg: sentinel))

    assert predict_knockout.main() == 0

    blender = _SpyPredictor.last_blender
    assert blender is not None, "predictor built without an explicit blender"
    assert blender.weights is sentinel, "predictor did not use BlendWeights.from_config"
