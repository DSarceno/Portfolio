"""Loader for the Kaggle 'International football results from 1872 to 2024' dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils.io import ensure_dir
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

EXPECTED_KAGGLE_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "city",
    "country",
    "neutral",
]

CANONICAL_COLUMNS = [
    "match_id",
    "date",
    "competition",
    "season",
    "stage",
    "group",
    "team_a",
    "team_b",
    "score_a",
    "score_b",
    "neutral_venue",
    "host_country",
    "source",
]

TOURNAMENT_TO_COMPETITION = {
    "FIFA World Cup": "WC",
    "FIFA World Cup qualification": "WCQ",
    "UEFA Euro": "EURO",
    "UEFA Euro qualification": "QUAL",
    "Copa América": "COPA",
    "Copa America": "COPA",
    "African Cup of Nations": "AFCON",
    "African Cup of Nations qualification": "QUAL",
    "Africa Cup of Nations": "AFCON",
    "Africa Cup of Nations qualification": "QUAL",
    "AFC Asian Cup": "ASIAN",
    "AFC Asian Cup qualification": "QUAL",
    "CONCACAF Gold Cup": "GOLD",
    "CONCACAF Championship": "GOLD",
    "Gold Cup": "GOLD",
    "CONCACAF Nations League": "CNL",
    "UEFA Nations League": "UNL",
    "Confederations Cup": "CONFED",
    "FIFA Confederations Cup": "CONFED",
    "Olympic Games": "OLYM",
    "Friendly": "FRIENDLY",
}


def _map_competition(tournament: str) -> str:
    """Map a Kaggle ``tournament`` value to a canonical competition code.

    Args:
        tournament: Raw tournament string from the Kaggle CSV.

    Returns:
        Canonical competition code (defaults to ``"QUAL"`` for any
        unrecognized qualification, ``"OTHER"`` otherwise).
    """
    if not isinstance(tournament, str):
        return "OTHER"
    if tournament in TOURNAMENT_TO_COMPETITION:
        return TOURNAMENT_TO_COMPETITION[tournament]
    lowered = tournament.lower()
    if "world cup" in lowered and "qualif" in lowered:
        return "WCQ"
    if "world cup" in lowered:
        return "WC"
    if "euro" in lowered and "qualif" in lowered:
        return "QUAL"
    if "qualif" in lowered:
        return "QUAL"
    if "friendly" in lowered:
        return "FRIENDLY"
    if "nations league" in lowered:
        return "UNL"
    if "olympic" in lowered:
        return "OLYM"
    return "OTHER"


class KaggleResultsClient:
    """Loads and normalizes the Kaggle international results dataset.

    The Kaggle dataset (``results.csv``) is available at:
    https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017
    """

    def __init__(
        self,
        csv_path: str | Path = "data/raw/kaggle/results.csv",
        cache_dir: str | Path = "data/raw/kaggle",
    ) -> None:
        """Initialize the client.

        Args:
            csv_path: Path to the Kaggle ``results.csv`` file.
            cache_dir: Directory where the dataset lives (created if missing).
        """
        self.csv_path = Path(csv_path)
        self.cache_dir = ensure_dir(cache_dir)

    def is_available(self) -> bool:
        """Return ``True`` when the dataset CSV is present."""
        return self.csv_path.exists()

    def load(
        self,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        include_friendlies: bool = True,
        include_qualifications: bool = True,
    ) -> pd.DataFrame:
        """Load and normalize the Kaggle dataset.

        Args:
            start_year: Optional minimum year (inclusive).
            end_year: Optional maximum year (inclusive).
            include_friendlies: When ``False`` drop friendlies.
            include_qualifications: When ``False`` drop qualification matches.

        Returns:
            DataFrame with the canonical schema. Returns an empty DataFrame
            if the file is missing.
        """
        if not self.is_available():
            logger.warning("Kaggle dataset not found at %s", self.csv_path)
            return pd.DataFrame(columns=CANONICAL_COLUMNS)

        raw = pd.read_csv(self.csv_path)
        missing = [c for c in EXPECTED_KAGGLE_COLUMNS if c not in raw.columns]
        if missing:
            logger.error("Kaggle CSV missing columns %s", missing)
            return pd.DataFrame(columns=CANONICAL_COLUMNS)

        raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
        raw = raw.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"])

        if start_year is not None:
            raw = raw[raw["date"].dt.year >= int(start_year)]
        if end_year is not None:
            raw = raw[raw["date"].dt.year <= int(end_year)]

        raw = raw.reset_index(drop=True)

        out = pd.DataFrame()
        out["match_id"] = [f"kaggle_{i}" for i in range(len(raw))]
        out["date"] = raw["date"].dt.strftime("%Y-%m-%d")
        out["competition"] = raw["tournament"].map(_map_competition)
        out["season"] = raw["date"].dt.year.astype(str)
        out["stage"] = "historical"
        out["group"] = ""
        out["team_a"] = raw["home_team"].astype(str)
        out["team_b"] = raw["away_team"].astype(str)
        out["score_a"] = raw["home_score"].astype(int)
        out["score_b"] = raw["away_score"].astype(int)
        out["neutral_venue"] = raw["neutral"].astype(bool)
        out["host_country"] = raw["country"].astype(str)
        out["source"] = "kaggle_international"

        if not include_friendlies:
            out = out[out["competition"] != "FRIENDLY"]
        if not include_qualifications:
            out = out[~out["competition"].isin(["QUAL", "WCQ"])]

        out = out.reset_index(drop=True)
        logger.info(
            "Loaded %d Kaggle matches (years %s-%s)",
            len(out),
            start_year if start_year is not None else "min",
            end_year if end_year is not None else "max",
        )
        return out

    def summary(self, df: pd.DataFrame) -> dict[str, int]:
        """Quick distribution summary for a normalized DataFrame."""
        if df.empty:
            return {}
        return {
            "total": int(len(df)),
            "teams": int(pd.unique(pd.concat([df["team_a"], df["team_b"]])).size),
            "competitions": int(df["competition"].nunique()),
            "year_min": int(pd.to_datetime(df["date"]).dt.year.min()),
            "year_max": int(pd.to_datetime(df["date"]).dt.year.max()),
        }
