"""Halt artifact must be written atomically and FAIL CLOSED on corruption."""

from __future__ import annotations

from pathlib import Path

from quant.governance.halt import halt_path, load_halt, set_halt


def test_corrupt_halt_reads_as_active(tmp_path: Path) -> None:
    # A truncated/garbage halt.json must be treated as HALTED, not raise / not open.
    p = halt_path(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{ this is not json", encoding="utf-8")
    state = load_halt(tmp_path)
    assert state.active is True
    assert "corrupt" in state.reason.lower()


def test_non_dict_halt_reads_as_active(tmp_path: Path) -> None:
    p = halt_path(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("[1, 2, 3]", encoding="utf-8")
    assert load_halt(tmp_path).active is True


def test_set_halt_leaves_no_tmp_file(tmp_path: Path) -> None:
    set_halt(tmp_path, reason="test")
    files = sorted(p.name for p in (tmp_path / "governance").iterdir())
    assert files == ["halt.json"], f"stray temp files: {files}"


def test_valid_halt_roundtrips(tmp_path: Path) -> None:
    set_halt(tmp_path, reason="drift breach")
    s = load_halt(tmp_path)
    assert s.active is True and s.reason == "drift breach"
