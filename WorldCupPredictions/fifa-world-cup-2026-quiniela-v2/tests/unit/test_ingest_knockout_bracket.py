"""Tests for the knockout-bracket ingestion helpers."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

ingest = importlib.import_module("ingest_knockout_bracket")
simulate = importlib.import_module("simulate_tournament")


def test_select_fixtures_excludes_scoreless_knockout_rows() -> None:
    """Knockout rows (blank group, no score) must not be tagged group-stage.

    Regression: ``str(NaN) == 'nan'`` is non-empty, so a naive ``group != ''``
    check wrongly classified scoreless round-of-32 rows as group fixtures,
    which blocked the post-group knockout-only auto-detection.
    """
    fm = pd.DataFrame(
        [
            {"date": "2026-06-15", "competition": "WC", "stage": "GROUP_STAGE",
             "group": "A", "team_a": "Brazil", "team_b": "Haiti", "score_a": float("nan")},
            {"date": "2026-06-29", "competition": "WC", "stage": "ROUND_OF_32",
             "group": float("nan"), "team_a": "Germany", "team_b": "Paraguay",
             "score_a": float("nan")},
        ]
    )
    selected = simulate._select_fixtures(fm, tournament_year=2026, competition="WC")
    assert list(selected["stage"]) == ["GROUP_STAGE"]


def _bracket() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # Filled tie -> ingested.
            {"match_id": "R32_01", "stage": "ROUND_OF_32", "feeds_winner_into": "R16_01",
             "feeds_loser_into": "", "date": "2026-06-29", "team_a": "Germany",
             "team_b": "Paraguay", "neutral_venue": False},
            # Not-yet-known tie -> skipped.
            {"match_id": "R16_01", "stage": "ROUND_OF_16", "feeds_winner_into": "QF_01",
             "feeds_loser_into": "", "date": "", "team_a": "", "team_b": "",
             "neutral_venue": False},
        ]
    )


class _FakeLoader:
    def __init__(self, known: list[str]) -> None:
        self._known = known

    def load_matches(self) -> pd.DataFrame:
        return pd.DataFrame({"team_a": self._known, "team_b": self._known})


def test_build_canonical_rows_keeps_only_filled_ties() -> None:
    rows = ingest.build_canonical_rows(_bracket(), tournament_year=2026)
    assert len(rows) == 1
    row = rows.iloc[0]
    assert row["team_a"] == "Germany" and row["team_b"] == "Paraguay"
    assert row["competition"] == "WC" and row["season"] == 2026
    assert row["stage"] == "ROUND_OF_32" and row["group"] == ""
    # Fixtures are unplayed: scores must be null.
    assert pd.isna(row["score_a"]) and pd.isna(row["score_b"])


def test_validate_team_names_flags_unknown() -> None:
    rows = ingest.build_canonical_rows(_bracket(), tournament_year=2026)
    # Loader knows Germany but not Paraguay -> Paraguay flagged.
    unknown = ingest.validate_team_names(rows, _FakeLoader(["Germany"]))
    assert unknown == ["Paraguay"]


def test_validate_team_names_passes_when_all_known() -> None:
    rows = ingest.build_canonical_rows(_bracket(), tournament_year=2026)
    assert ingest.validate_team_names(rows, _FakeLoader(["Germany", "Paraguay"])) == []


class _ResultLoader:
    """Loader whose canonical table already holds one played tie."""

    def __init__(self, played: pd.DataFrame) -> None:
        self._played = played

    def load_matches(self) -> pd.DataFrame:
        return self._played


def test_drop_already_played_holds_back_results() -> None:
    rows = ingest.build_canonical_rows(_bracket(), tournament_year=2026)  # Germany vs Paraguay, 2026-06-29
    canonical = pd.DataFrame(
        [
            # Same (date, team_a, team_b) key as the bracket tie, now with a score.
            {"date": "2026-06-29", "team_a": "Germany", "team_b": "Paraguay",
             "score_a": 2.0, "score_b": 1.0},
        ]
    )
    kept, played = ingest.drop_already_played(rows, _ResultLoader(canonical))
    assert len(played) == 1 and len(kept) == 0


def test_drop_already_played_ignores_unplayed_and_other_dates() -> None:
    rows = ingest.build_canonical_rows(_bracket(), tournament_year=2026)
    canonical = pd.DataFrame(
        [
            # Same teams but a different date (e.g. a group-stage meeting) -> not the tie.
            {"date": "2026-06-15", "team_a": "Germany", "team_b": "Paraguay",
             "score_a": 0.0, "score_b": 0.0},
            # The actual tie, still unplayed.
            {"date": "2026-06-29", "team_a": "Germany", "team_b": "Paraguay",
             "score_a": pd.NA, "score_b": pd.NA},
        ]
    )
    kept, played = ingest.drop_already_played(rows, _ResultLoader(canonical))
    assert len(played) == 0 and len(kept) == 1


def _penalty_bracket(tmp_path: Path, r16_filled: bool) -> Path:
    """R32_01 (Alpha-Beta) and R32_02 (Gamma-Delta) both feed R16_01."""
    rows = [
        "match_id,stage,feeds_winner_into,feeds_loser_into,date,team_a,team_b,neutral_venue",
        "R32_01,ROUND_OF_32,R16_01,,2026-06-30,Alpha,Beta,False",
        "R32_02,ROUND_OF_32,R16_01,,2026-06-30,Gamma,Delta,False",
    ]
    # R16_01 is filled with the two advancers (Alpha from R32_01, Gamma from R32_02)
    rows.append(
        "R16_01,ROUND_OF_16,QF_01,,2026-07-05,Alpha,Gamma,False"
        if r16_filled
        else "R16_01,ROUND_OF_16,QF_01,,2026-07-05,,,False"
    )
    path = tmp_path / "bracket.csv"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_bracket_advancers_reads_winner_from_filled_next_round(tmp_path) -> None:
    adv = simulate._bracket_advancers(_penalty_bracket(tmp_path, r16_filled=True))
    assert adv[frozenset({"Alpha", "Beta"})] == "Alpha"
    assert adv[frozenset({"Gamma", "Delta"})] == "Gamma"


def test_bracket_advancers_empty_when_next_round_unfilled(tmp_path) -> None:
    adv = simulate._bracket_advancers(_penalty_bracket(tmp_path, r16_filled=False))
    assert frozenset({"Alpha", "Beta"}) not in adv
    assert frozenset({"Gamma", "Delta"}) not in adv


class _MatchLoader:
    """Minimal DataLoader stub returning a fixed canonical table."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def load_matches(self) -> pd.DataFrame:
        return self._df


def _drawn_r32() -> pd.DataFrame:
    return pd.DataFrame(
        [{"date": "2026-06-30", "stage": "ROUND_OF_32", "team_a": "Alpha",
          "team_b": "Beta", "score_a": 1, "score_b": 1}]
    )


def test_played_knockout_winners_infers_penalty_winner(tmp_path, monkeypatch) -> None:
    """A drawn (penalty) tie's winner is recovered from the bracket's next round."""
    monkeypatch.setattr(simulate, "DataLoader", lambda: _MatchLoader(_drawn_r32()))
    winners = simulate._played_knockout_winners(
        2026, bracket_csv=_penalty_bracket(tmp_path, r16_filled=True)
    )
    assert winners == {frozenset({"Alpha", "Beta"}): "Alpha"}


def test_played_knockout_winners_leaves_draw_to_sim_when_unresolvable(tmp_path, monkeypatch) -> None:
    """Without a filled next round (or no bracket), a draw stays unforced."""
    monkeypatch.setattr(simulate, "DataLoader", lambda: _MatchLoader(_drawn_r32()))
    assert simulate._played_knockout_winners(2026, bracket_csv=None) == {}
    assert simulate._played_knockout_winners(
        2026, bracket_csv=_penalty_bracket(tmp_path, r16_filled=False)
    ) == {}
