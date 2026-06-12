"""Tests for the ensemble blending layer."""

from __future__ import annotations

import numpy as np

from src.ensemble.blender import BlendWeights, ProbabilityBlender


def test_from_model_params_reads_configured_weights() -> None:
    model_params = {
        "ensemble": {
            "weights": {
                "ratings": 0.35,
                "multinomial": 0.20,
                "xgboost": 0.30,
                "poisson": 0.15,
            }
        }
    }
    weights = BlendWeights.from_model_params(model_params)
    assert weights.ratings == 0.35
    assert weights.multinomial == 0.20
    assert weights.xgboost == 0.30
    assert weights.poisson == 0.15


def test_from_model_params_falls_back_to_defaults_on_missing_keys() -> None:
    defaults = BlendWeights()
    weights = BlendWeights.from_model_params({"ensemble": {"weights": {"poisson": 0.10}}})
    assert weights.poisson == 0.10
    assert weights.ratings == defaults.ratings
    assert weights.xgboost == defaults.xgboost


def test_from_model_params_handles_malformed_config() -> None:
    defaults = BlendWeights()
    for bad in (None, {}, {"ensemble": None}, {"ensemble": {"weights": "nope"}}):
        weights = BlendWeights.from_model_params(bad)
        assert weights == defaults


def test_from_model_params_ignores_non_numeric_value() -> None:
    defaults = BlendWeights()
    weights = BlendWeights.from_model_params({"ensemble": {"weights": {"ratings": "high"}}})
    assert weights.ratings == defaults.ratings


def test_blend_respects_relative_weights() -> None:
    # Two models, sharply opposed predictions. The heavier weight must dominate.
    a = np.array([[0.8, 0.1, 0.1]])
    b = np.array([[0.1, 0.1, 0.8]])
    blender = ProbabilityBlender(
        weights=BlendWeights(ratings=0.9, multinomial=0.0, xgboost=0.0, poisson=0.1)
    )
    blended = blender.blend(ratings=a, poisson=b)
    np.testing.assert_allclose(blended.sum(axis=1), 1.0, atol=1e-6)
    # ratings (weight 0.9) favours home, so home prob stays the largest.
    assert blended[0, 0] > blended[0, 2]
