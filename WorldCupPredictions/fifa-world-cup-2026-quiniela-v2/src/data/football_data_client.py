"""Client for football-data.org with offline-cached fallback."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import httpx
import pandas as pd

from src.utils.io import ensure_dir, save_csv, snapshot_path
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

BASE_URL = "https://api.football-data.org/v4"
EXPECTED_COLUMNS = [
    "match_id",
    "date",
    "competition",
    "season",
    "stage",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "status",
    "venue",
    "neutral_venue",
]


class FootballDataClient:
    """Lightweight client around football-data.org with an offline cache mode.

    When ``api_key`` is missing the client still works: it reads cached
    snapshots from ``cache_dir`` and from ``manual_dir`` when available.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: str | Path = "data/raw/football_data",
        manual_dir: str | Path = "data/raw/manual",
        timeout: float = 20.0,
    ) -> None:
        """Initialize the client.

        Args:
            api_key: football-data.org API key. Defaults to the
                ``FOOTBALL_DATA_API_KEY`` environment variable.
            cache_dir: Directory storing timestamped raw snapshots.
            manual_dir: Directory storing manual CSV overrides.
            timeout: Network timeout in seconds.
        """
        self.api_key = api_key or os.getenv("FOOTBALL_DATA_API_KEY", "")
        self.cache_dir = ensure_dir(cache_dir)
        self.manual_dir = ensure_dir(manual_dir)
        self.timeout = timeout

    def fetch_matches(
        self,
        competition: str,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch matches for a competition.

        Falls back to manual CSV when no API key is configured or when the
        remote call fails.

        Args:
            competition: Competition code (e.g. ``"WC"``).
            date_from: ISO start date (inclusive).
            date_to: ISO end date (inclusive).

        Returns:
            DataFrame with :data:`EXPECTED_COLUMNS`.
        """
        if not self.api_key:
            logger.info("No API key set, reading cached/manual fixtures for %s", competition)
            return self._read_local(competition)

        params: dict[str, str] = {}
        if date_from:
            params["dateFrom"] = date_from
        if date_to:
            params["dateTo"] = date_to

        url = f"{BASE_URL}/competitions/{competition}/matches"
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(
                    url,
                    headers={"X-Auth-Token": self.api_key},
                    params=params,
                )
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            logger.error("football-data.org request failed: %s", exc)
            return self._read_local(competition)

        df = self._normalize(payload, competition)
        save_csv(df, snapshot_path(self.cache_dir, "football_data_matches"))
        return df

    def _read_local(self, competition: str) -> pd.DataFrame:
        """Read the most recent cached or manual snapshot for *competition*.

        Args:
            competition: Competition code.

        Returns:
            DataFrame; empty with the expected schema when nothing is found.
        """
        manual = self.manual_dir / f"manual_matches_{competition}.csv"
        if manual.exists():
            df = pd.read_csv(manual)
            missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
            if missing:
                logger.warning("Manual matches CSV missing columns %s", missing)
            return df

        files = sorted(self.cache_dir.glob("football_data_matches_*.csv"))
        if files:
            return pd.read_csv(files[-1])

        logger.warning("No cached or manual matches for %s; returning template", competition)
        template = pd.DataFrame(columns=EXPECTED_COLUMNS)
        save_csv(template, self.manual_dir / f"manual_matches_{competition}.csv")
        return template

    @staticmethod
    def _normalize(payload: dict, competition: str) -> pd.DataFrame:
        """Normalize a football-data.org match payload into the canonical schema.

        Args:
            payload: Raw JSON dictionary from the API.
            competition: Competition code injected as a column.

        Returns:
            DataFrame with :data:`EXPECTED_COLUMNS`.
        """
        rows: list[dict] = []
        for match in payload.get("matches", []):
            score = match.get("score", {}).get("fullTime", {}) or {}
            rows.append(
                {
                    "match_id": match.get("id"),
                    "date": (match.get("utcDate") or "")[:10],
                    "competition": competition,
                    "season": match.get("season", {}).get("startDate", "")[:4],
                    "stage": match.get("stage", ""),
                    "home_team": match.get("homeTeam", {}).get("name"),
                    "away_team": match.get("awayTeam", {}).get("name"),
                    "home_score": score.get("home"),
                    "away_score": score.get("away"),
                    "status": match.get("status", ""),
                    "venue": match.get("venue", ""),
                    "neutral_venue": False,
                }
            )
        return pd.DataFrame(rows, columns=EXPECTED_COLUMNS)
