"""Round-of-32 bracket generation from group-stage outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class BracketPair:
    """A single round-of-32 pairing."""

    slot: int
    team_a: str
    team_b: str


def load_known_round_of_32(
    bracket_csv: str | Path = "data/raw/manual/wc2026_knockout_bracket.csv",
) -> list[BracketPair]:
    """Load the *real* round-of-32 pairings from the manual bracket CSV.

    Once the group stage is over the round-of-32 matchups are known, so the
    Monte-Carlo simulator should no longer re-simulate the groups — it should
    seed the knockout from these actual ties. The CSV's ``match_id`` order
    (``R32_01``..``R32_16``) is the bracket's slot order, and it lines up with
    the binary feed mapping (``R32_01`` + ``R32_02`` -> ``R16_01``, ...), which
    is exactly how :func:`src.simulation.knockout.simulate_knockout` pairs the
    list, so the bracket tree is preserved.

    Args:
        bracket_csv: Path to the knockout-bracket CSV.

    Returns:
        A length-16 list of :class:`BracketPair` in slot order.

    Raises:
        FileNotFoundError: If the CSV does not exist.
        ValueError: If the round of 32 is not fully and consistently filled
            (not 16 ties, a missing team, or a duplicated team).
    """
    path = Path(bracket_csv)
    if not path.exists():
        raise FileNotFoundError(f"Knockout-bracket CSV not found: {path}")

    df = pd.read_csv(path)
    r32 = df[df["stage"].astype(str).str.upper() == "ROUND_OF_32"].copy()
    r32 = r32.sort_values("match_id").reset_index(drop=True)

    def _filled(value: object) -> bool:
        return not (pd.isna(value) or str(value).strip() == "")

    pairs: list[BracketPair] = []
    for slot, row in enumerate(r32.itertuples(), start=1):
        if not (_filled(row.team_a) and _filled(row.team_b)):
            raise ValueError(
                f"Round-of-32 tie {row.match_id} is not filled in yet "
                f"(team_a={row.team_a!r}, team_b={row.team_b!r}); "
                "complete the bracket before running the knockout simulation."
            )
        pairs.append(BracketPair(slot=slot, team_a=str(row.team_a).strip(), team_b=str(row.team_b).strip()))

    if len(pairs) != 16:
        raise ValueError(
            f"Expected 16 round-of-32 ties, found {len(pairs)} in {path}."
        )

    teams = [t for p in pairs for t in (p.team_a, p.team_b)]
    if len(set(teams)) != 32:
        from collections import Counter

        dupes = sorted(t for t, c in Counter(teams).items() if c > 1)
        raise ValueError(
            f"Round of 32 must have 32 distinct teams; found {len(set(teams))} "
            f"(duplicated: {dupes})."
        )

    logger.info("Loaded a complete real round-of-32 (%d ties) from %s", len(pairs), path)
    return pairs


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
