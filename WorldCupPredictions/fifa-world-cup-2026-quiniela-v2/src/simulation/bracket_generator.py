"""Round-of-32 bracket generation from group-stage outcomes."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class BracketPair:
    """A single round-of-32 pairing."""

    slot: int
    team_a: str
    team_b: str


def build_round_of_32_bracket(
    standings: pd.DataFrame,
    best_thirds: list[str],
) -> list[BracketPair]:
    """Build a deterministic round-of-32 bracket.

    The pairing rule keeps it simple but consistent across simulations:
    winners of consecutive groups (1A, 1B, 1C, ...) face runners-up from
    rotated groups (2B, 2C, ...), and best-third teams fill the remaining
    slots in order.

    Args:
        standings: Full group standings DataFrame.
        best_thirds: List of eight best-third team names.

    Returns:
        List of :class:`BracketPair` (length 16).
    """
    winners = (
        standings.sort_values(["group", "rank"]).groupby("group").first().sort_index()
    )["team"].tolist()
    runners = (
        standings[standings["rank"] == 2].sort_values("group")["team"].tolist()
    )

    runners_rotated = runners[1:] + runners[:1] if runners else []
    pairs: list[BracketPair] = []
    slot = 1

    for w, r in zip(winners, runners_rotated):
        pairs.append(BracketPair(slot=slot, team_a=w, team_b=r))
        slot += 1

    extras = [t for t in best_thirds if t not in {p.team_a for p in pairs} | {p.team_b for p in pairs}]
    while len(pairs) < 16 and extras:
        team_a = extras.pop(0)
        team_b = extras.pop(0) if extras else (winners[0] if winners else team_a)
        pairs.append(BracketPair(slot=slot, team_a=team_a, team_b=team_b))
        slot += 1

    return pairs
