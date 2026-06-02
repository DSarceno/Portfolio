"""Project-wide constants."""

from __future__ import annotations

from typing import Final

OUTCOME_HOME_WIN: Final[str] = "H"
OUTCOME_DRAW: Final[str] = "D"
OUTCOME_AWAY_WIN: Final[str] = "A"

OUTCOMES: Final[tuple[str, str, str]] = (OUTCOME_HOME_WIN, OUTCOME_DRAW, OUTCOME_AWAY_WIN)
OUTCOME_TO_INDEX: Final[dict[str, int]] = {
    OUTCOME_HOME_WIN: 0,
    OUTCOME_DRAW: 1,
    OUTCOME_AWAY_WIN: 2,
}
INDEX_TO_OUTCOME: Final[dict[int, str]] = {v: k for k, v in OUTCOME_TO_INDEX.items()}

STAGE_GROUP: Final[str] = "group"
STAGE_R32: Final[str] = "r32"
STAGE_R16: Final[str] = "r16"
STAGE_QF: Final[str] = "qf"
STAGE_SF: Final[str] = "sf"
STAGE_THIRD: Final[str] = "third"
STAGE_FINAL: Final[str] = "final"

KNOCKOUT_STAGES: Final[tuple[str, ...]] = (
    STAGE_R32,
    STAGE_R16,
    STAGE_QF,
    STAGE_SF,
    STAGE_THIRD,
    STAGE_FINAL,
)

POINTS_WIN: Final[int] = 3
POINTS_DRAW: Final[int] = 1
POINTS_LOSS: Final[int] = 0

WORLD_CUP_2026_HOSTS: Final[tuple[str, ...]] = ("United States", "Canada", "Mexico")

CONFEDERATIONS: Final[tuple[str, ...]] = (
    "UEFA",
    "CONMEBOL",
    "CONCACAF",
    "AFC",
    "CAF",
    "OFC",
)

CONFEDERATION_STRENGTH_PRIOR: Final[dict[str, float]] = {
    "UEFA": 1.00,
    "CONMEBOL": 0.95,
    "CONCACAF": 0.78,
    "AFC": 0.74,
    "CAF": 0.76,
    "OFC": 0.60,
}

RISK_PROFILES: Final[tuple[str, ...]] = ("safe", "balanced", "aggressive", "contrarian")

DEFAULT_RANDOM_SEED: Final[int] = 42
