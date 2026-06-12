"""Match-level squad-value features (A.2).

Squad value depends on the **match date** (a team's talent in 2018 differs from
2026), so it is a match-context feature rather than a static team feature. For
each fixture we as-of join the most recent squad-value snapshot per team
(``as_of_date <= match date``), log-transform (values are heavy-tailed) and emit
the home-minus-away differences consumed by the models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.ratings.shrinkage import TEAM_CONFEDERATION
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

SQUAD_VALUE_FEATURE_COLUMNS = ["squad_value_total_diff", "squad_value_top11_diff"]


def _asof_lookup(
    dates: pd.Series,
    teams: pd.Series,
    snapshots: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """As-of join: latest snapshot per (team, date) with ``as_of_date <= date``.

    Args:
        dates: Per-row match dates (datetime).
        teams: Per-row team names, index-aligned with *dates*.
        snapshots: Squad-value table sorted by ``as_of_date``.

    Returns:
        ``(total, top11)`` arrays aligned to the input order; ``NaN`` where no
        snapshot precedes the date or the team is absent.
    """
    n = len(dates)
    total = np.full(n, np.nan)
    top11 = np.full(n, np.nan)
    frame = pd.DataFrame(
        {"_i": np.arange(n), "_date": dates.to_numpy(), "team": teams.astype(str).to_numpy()}
    )
    valid = frame.dropna(subset=["_date"]).sort_values("_date")
    if valid.empty:
        return total, top11
    merged = pd.merge_asof(
        valid,
        snapshots[["as_of_date", "team", "squad_value_total_meur", "squad_value_top11_meur"]],
        left_on="_date",
        right_on="as_of_date",
        by="team",
        direction="backward",
    )
    idx = merged["_i"].to_numpy()
    total[idx] = merged["squad_value_total_meur"].to_numpy()
    top11[idx] = merged["squad_value_top11_meur"].to_numpy()
    return total, top11


def _fill_by_confederation(
    values: np.ndarray,
    teams: pd.Series,
    conf_median: pd.Series,
    global_median: float,
) -> np.ndarray:
    """Fill missing values with the team's confederation median, else global."""
    confs = teams.astype(str).map(lambda t: TEAM_CONFEDERATION.get(t, "UNKNOWN"))
    fallback = confs.map(conf_median).to_numpy(dtype=float)
    fallback = np.where(np.isfinite(fallback), fallback, global_median)
    return np.where(np.isfinite(values), values, fallback)


def compute_squad_value_features(
    match_features: pd.DataFrame,
    squad_values: pd.DataFrame | None,
) -> pd.DataFrame:
    """Add squad-value difference features to a match-level frame.

    Args:
        match_features: Match feature frame with ``date``, ``team_a``,
            ``team_b``.
        squad_values: Canonical squad-value snapshots (see
            :class:`~src.data.squad_value_client.SquadValueClient`). ``None`` or
            empty leaves *match_features* unchanged (clean degradation).

    Returns:
        *match_features* with ``squad_value_total_diff`` and
        ``squad_value_top11_diff`` (log home minus log away). Missing teams are
        filled with the confederation median before the log transform.
    """
    out = match_features.copy()
    if squad_values is None or squad_values.empty:
        logger.info("No squad values supplied; skipping squad-value features")
        return out
    if not {"team_a", "team_b", "date"}.issubset(out.columns):
        logger.warning("match_features lacks team_a/team_b/date; skipping squad-value features")
        return out

    snap = squad_values.copy()
    snap["as_of_date"] = pd.to_datetime(snap["as_of_date"], errors="coerce")
    snap = snap.dropna(subset=["as_of_date"]).sort_values("as_of_date")

    dates = pd.to_datetime(out["date"], errors="coerce")
    total_a, top11_a = _asof_lookup(dates, out["team_a"], snap)
    total_b, top11_b = _asof_lookup(dates, out["team_b"], snap)

    # Confederation / global medians from the values actually present, so the
    # fill tracks the era covered by this matrix.
    present = pd.DataFrame(
        {
            "team": pd.concat([out["team_a"], out["team_b"]], ignore_index=True).astype(str),
            "total": np.concatenate([total_a, total_b]),
            "top11": np.concatenate([top11_a, top11_b]),
        }
    )
    present["conf"] = present["team"].map(lambda t: TEAM_CONFEDERATION.get(t, "UNKNOWN"))
    conf_total = present.groupby("conf")["total"].median()
    conf_top11 = present.groupby("conf")["top11"].median()
    glob_total = float(np.nanmedian(present["total"])) if present["total"].notna().any() else 0.0
    glob_top11 = float(np.nanmedian(present["top11"])) if present["top11"].notna().any() else 0.0

    total_a = _fill_by_confederation(total_a, out["team_a"], conf_total, glob_total)
    total_b = _fill_by_confederation(total_b, out["team_b"], conf_total, glob_total)
    top11_a = _fill_by_confederation(top11_a, out["team_a"], conf_top11, glob_top11)
    top11_b = _fill_by_confederation(top11_b, out["team_b"], conf_top11, glob_top11)

    out["squad_value_total_diff"] = np.log1p(total_a) - np.log1p(total_b)
    out["squad_value_top11_diff"] = np.log1p(top11_a) - np.log1p(top11_b)
    logger.info(
        "Squad-value features added (%d matches, %.1f%% from snapshots)",
        len(out),
        100.0 * np.isfinite(np.concatenate([total_a, total_b])).mean(),
    )
    return out
