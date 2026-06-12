"""Squad-value data source (A.2 — valor de plantilla).

Loads the canonical snapshot table produced by ``scripts/build_squad_values.py``
and exposes it to the feature layer. Squad value (especially the top-11 sum) is
the most predictive covariate after Elo in the academic literature
(Groll-Schauberger-Tutz 2015, Groll-Ley 2019); it captures *current talent*,
which a results-based rating cannot see when a power rotates its squad or hits a
poor run of form.

The canonical table is a set of **dated snapshots**:

    team, as_of_date, squad_value_total_meur, squad_value_top11_meur, source

Values are in millions of EUR. The feature layer performs an *as-of* join so a
match on date ``D`` uses the most recent snapshot with ``as_of_date <= D``.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

SQUAD_VALUES_PATH = "data/raw/squad_values/squad_values.csv"

CANONICAL_SQUAD_VALUE_COLUMNS = [
    "team",
    "as_of_date",
    "squad_value_total_meur",
    "squad_value_top11_meur",
    "source",
]

# Transfermarkt/Kaggle country labels -> canonical team names used in
# ``matches_unified.csv``. Only the genuine mismatches are listed; everything
# else matches directly. Verified against the 48 WC-2026 participants.
SQUAD_VALUE_NAME_MAP: dict[str, str] = {
    "Cape Verde": "Cape Verde Islands",
    "DR Congo": "Congo DR",
    "Curacao": "Curaçao",
    "Czech Republic": "Czechia",
    "Cote d'Ivoire": "Ivory Coast",
    "Côte d'Ivoire": "Ivory Coast",
    "Korea, South": "South Korea",
    "Türkiye": "Turkey",
}


def _strip_accents(text: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch)
    )


def canonical_team_name(name: str) -> str:
    """Map a Kaggle citizenship/country label to a canonical team name.

    Falls back to an accent-insensitive lookup before returning *name*
    unchanged, so encoding quirks (e.g. ``Türkiye``) still resolve.

    Args:
        name: Raw country/citizenship string from the Kaggle dataset.

    Returns:
        Canonical team name as used in the project's match table.
    """
    if name in SQUAD_VALUE_NAME_MAP:
        return SQUAD_VALUE_NAME_MAP[name]
    stripped = _strip_accents(str(name))
    for raw, canonical in SQUAD_VALUE_NAME_MAP.items():
        if _strip_accents(raw) == stripped:
            return canonical
    return name


def build_citizenship_snapshot(
    valuations: pd.DataFrame,
    players: pd.DataFrame,
    as_of_date: str | pd.Timestamp,
) -> pd.DataFrame:
    """Aggregate a national-squad-value snapshot by player citizenship.

    For each player, the most recent market valuation with ``date <= as_of_date``
    is taken, players are grouped by canonical national team (via
    :func:`canonical_team_name` on ``country_of_citizenship``), and the per-team
    total and top-11 sums (in millions of EUR) are returned.

    This is a *proxy*: citizenship is not the same as a tournament call-up, so
    the total is noisy (it counts every citizen pro). The **top-11** is far more
    robust — it tracks the would-be starting XI — and is the signal the feature
    layer leans on. Using the same method for every snapshot keeps the feature on
    a consistent scale across train (historical) and prediction (2026), which is
    what lets the model transfer.

    Args:
        valuations: ``player_valuations`` with ``player_id``, ``date`` (datetime)
            and ``market_value_in_eur``.
        players: ``players`` with ``player_id`` and ``country_of_citizenship``.
        as_of_date: Snapshot cut-off date.

    Returns:
        DataFrame with :data:`CANONICAL_SQUAD_VALUE_COLUMNS`
        (``source="kaggle_citizenship"``). Empty if no valuations precede the
        cut-off.
    """
    asof = pd.Timestamp(as_of_date)
    sub = valuations[valuations["date"] <= asof]
    if sub.empty:
        return pd.DataFrame(columns=CANONICAL_SQUAD_VALUE_COLUMNS)

    latest = sub.sort_values("date").groupby("player_id")["market_value_in_eur"].last()
    merged = latest.reset_index().merge(
        players[["player_id", "country_of_citizenship"]], on="player_id", how="left"
    )
    merged = merged.dropna(subset=["country_of_citizenship"])
    merged["team"] = merged["country_of_citizenship"].astype(str).map(canonical_team_name)
    merged["market_value_in_eur"] = pd.to_numeric(
        merged["market_value_in_eur"], errors="coerce"
    ).fillna(0.0)

    grouped = merged.groupby("team")["market_value_in_eur"]
    total = grouped.sum() / 1e6
    top11 = grouped.apply(lambda s: s.nlargest(11).sum()) / 1e6
    out = pd.DataFrame(
        {
            "team": total.index.astype(str),
            "as_of_date": asof.date().isoformat(),
            "squad_value_total_meur": total.to_numpy().round(3),
            "squad_value_top11_meur": top11.reindex(total.index).to_numpy().round(3),
            "source": "kaggle_citizenship",
        }
    )
    return out.reset_index(drop=True)


@dataclass
class SquadValueClient:
    """Loader for the canonical squad-value snapshot table."""

    path: str = SQUAD_VALUES_PATH

    def is_available(self) -> bool:
        """Return whether the snapshot table exists on disk."""
        return Path(self.path).exists()

    def load(self) -> pd.DataFrame:
        """Load and validate the squad-value snapshots.

        Returns:
            DataFrame with :data:`CANONICAL_SQUAD_VALUE_COLUMNS`, ``as_of_date``
            parsed to ``datetime64`` and the two value columns coerced to float.
            Returns an empty (correctly-typed) frame when the file is missing.

        Raises:
            ValueError: If required columns are absent from the file.
        """
        if not self.is_available():
            logger.warning("Squad-value table %s not found; returning empty", self.path)
            return pd.DataFrame(columns=CANONICAL_SQUAD_VALUE_COLUMNS)

        df = pd.read_csv(self.path)
        missing = [c for c in CANONICAL_SQUAD_VALUE_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Squad-value table missing columns: {missing}")

        df = df.copy()
        df["team"] = df["team"].astype(str)
        df["as_of_date"] = pd.to_datetime(df["as_of_date"], errors="coerce")
        for col in ("squad_value_total_meur", "squad_value_top11_meur"):
            df[col] = pd.to_numeric(df[col], errors="coerce").clip(lower=0.0)
        df = df.dropna(subset=["team", "as_of_date"])
        logger.info(
            "Loaded squad values: %d snapshots, %d teams, dates %s",
            len(df),
            df["team"].nunique(),
            sorted(df["as_of_date"].dt.date.unique().tolist()),
        )
        return df.sort_values(["team", "as_of_date"]).reset_index(drop=True)
