"""Tests for the dated outputs-snapshot helper."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

snapshot = importlib.import_module("snapshot_outputs")


def _make_outputs(tmp_path: Path) -> Path:
    """Build a minimal outputs/ tree with one CSV per snapshot subdir."""
    outputs = tmp_path / "outputs"
    for sub in snapshot.SNAPSHOT_SUBDIRS:
        d = outputs / sub
        d.mkdir(parents=True)
        (d / f"{sub}_sample.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    # championship file the manifest reads for its top-5.
    (outputs / "simulations" / "championship_probabilities.csv").write_text(
        "team,championship_prob\nArgentina,0.18\nFrance,0.15\n", encoding="utf-8"
    )
    return outputs


def test_snapshot_copies_all_subdirs_and_writes_manifest(tmp_path, monkeypatch) -> None:
    """A snapshot mirrors every output subdir and records a manifest."""
    outputs = _make_outputs(tmp_path)
    monkeypatch.setattr(sys, "argv", ["snapshot_outputs.py", "--label", "t1",
                                      "--outputs-dir", str(outputs)])
    assert snapshot.main() == 0

    dest = outputs / "snapshots" / "t1"
    for sub in snapshot.SNAPSHOT_SUBDIRS:
        assert (dest / sub).is_dir()
        assert list((dest / sub).glob("*.csv")), f"{sub} not copied"

    manifest = json.loads((dest / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["label"] == "t1"
    assert manifest["files_copied"] >= len(snapshot.SNAPSHOT_SUBDIRS)
    assert manifest["championship_top5"][0]["team"] == "Argentina"


def test_rerun_same_label_refreshes_in_place(tmp_path, monkeypatch) -> None:
    """Re-running with the same label overwrites rather than nesting/duplicating."""
    outputs = _make_outputs(tmp_path)
    argv = ["snapshot_outputs.py", "--label", "day", "--outputs-dir", str(outputs)]
    monkeypatch.setattr(sys, "argv", argv)
    assert snapshot.main() == 0

    # Add a stray file inside the snapshot; a refresh must clear it.
    stray = outputs / "snapshots" / "day" / "stray.txt"
    stray.write_text("x", encoding="utf-8")
    assert snapshot.main() == 0
    assert not stray.exists()
    # Still exactly one snapshot folder for that label.
    assert [p.name for p in (outputs / "snapshots").iterdir()] == ["day"]
