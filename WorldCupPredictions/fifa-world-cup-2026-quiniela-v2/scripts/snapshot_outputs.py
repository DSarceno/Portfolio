"""Archive a dated copy of ``outputs/`` so each matchday's forecast is preserved.

The live ``outputs/`` tree (predictions, simulations, picks, diagnostics) is
gitignored and is *overwritten* on every matchday run, so the forecast as it
stood at any earlier point in the tournament is otherwise lost. That history is
exactly what the planned pre-tournament-vs-actual comparison needs, so this
script copies the current outputs into ``outputs/snapshots/<label>/`` together
with a small ``manifest.json`` describing the state that produced them.

Snapshots are **committable** (exempted from ``.gitignore``) on purpose: the
whole point is that they survive the next overwrite and can be diffed later.

Default label is today's date (``YYYY-MM-DD``); re-running the same day refreshes
that day's snapshot in place (the latest run of a matchday is the canonical one).
Pass ``--label`` for a custom name (e.g. ``pre_tournament``, ``2026-07-09_R16``).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.utils.config import load_config
from src.utils.logging_config import get_logger, setup_logging

# Output sub-directories that make up a forecast snapshot.
SNAPSHOT_SUBDIRS = ("predictions", "simulations", "picks", "diagnostics")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Archive a dated copy of outputs/")
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Snapshot folder name (default: today's date, YYYY-MM-DD).",
    )
    parser.add_argument(
        "--outputs-dir",
        type=str,
        default=None,
        help="Override the outputs directory (default: paths.outputs_dir from config).",
    )
    parser.add_argument(
        "--note",
        type=str,
        default="",
        help="Free-text note stored in the snapshot manifest (e.g. 'after R32').",
    )
    return parser.parse_args()


def _git_commit() -> str:
    """Best-effort short git commit hash; empty string if unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:  # noqa: BLE001 - git is optional context, never fatal
        return ""


def _played_wc_matches(tournament_year: int) -> int:
    """Count played World Cup matches in the canonical table (best-effort).

    Filters on the **date's year**, not the ``season`` column: the live WC 2026
    rows are ingested from the manual results CSV / bracket and carry a blank
    ``season`` (only the historical Kaggle feed populates it), so a season-based
    filter would miss every actual tournament match.
    """
    try:
        from src.data.data_loader import DataLoader

        matches = DataLoader().load_matches()
        if matches.empty or "score_a" not in matches.columns:
            return 0
        is_wc = matches["competition"].astype(str).str.upper().eq("WC")
        is_year = pd.to_datetime(matches["date"], errors="coerce").dt.year.eq(tournament_year)
        return int((is_wc & is_year & matches["score_a"].notna()).sum())
    except Exception:  # noqa: BLE001 - manifest context only, never fatal
        return -1


def _champion_top5(outputs_dir: Path) -> list[dict[str, object]]:
    """Read the current championship top-5 for the manifest (best-effort)."""
    path = outputs_dir / "simulations" / "championship_probabilities.csv"
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path)
        prob_col = "championship_prob" if "championship_prob" in df.columns else df.columns[-1]
        top = df.sort_values(prob_col, ascending=False).head(5)
        return [
            {"team": str(r["team"]), "prob": round(float(r[prob_col]), 4)}
            for _, r in top.iterrows()
        ]
    except Exception:  # noqa: BLE001
        return []


def main() -> int:
    """Copy the live outputs into a dated snapshot folder with a manifest."""
    setup_logging(log_file="logs/updates/snapshot_outputs.log")
    logger = get_logger(__name__)
    config = load_config()
    args = parse_args()

    outputs_dir = Path(args.outputs_dir or config.get("paths.outputs_dir", "outputs"))
    if not outputs_dir.is_absolute():
        outputs_dir = PROJECT_ROOT / outputs_dir
    if not outputs_dir.exists():
        logger.error("Outputs directory not found: %s", outputs_dir)
        return 1

    label = args.label or datetime.now().strftime("%Y-%m-%d")
    dest = outputs_dir / "snapshots" / label
    if dest.exists():
        logger.info("Refreshing existing snapshot '%s' (overwriting)", label)
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    copied = 0
    for sub in SNAPSHOT_SUBDIRS:
        src = outputs_dir / sub
        if not src.is_dir():
            continue
        files = sorted(src.glob("*.csv"))
        if not files:
            continue
        (dest / sub).mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy2(f, dest / sub / f.name)
            copied += 1

    if copied == 0:
        logger.warning("No CSV outputs found under %s; snapshot is empty.", outputs_dir)

    tournament_year = int(config.get("tournament.year", 2026))
    manifest = {
        "label": label,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "note": args.note,
        "files_copied": copied,
        "played_wc_matches": _played_wc_matches(tournament_year),
        "championship_top5": _champion_top5(outputs_dir),
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    logger.info(
        "Snapshot '%s' written to %s (%d file(s), %d WC matches played)",
        label,
        dest,
        copied,
        manifest["played_wc_matches"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
