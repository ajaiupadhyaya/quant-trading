"""Tests for atomic JSON/text writes."""

from __future__ import annotations

import json
from pathlib import Path

from quant.util.atomic import atomic_write_text, write_json_atomic


def test_write_json_atomic_roundtrips(tmp_path: Path) -> None:
    p = tmp_path / "sub" / "x.json"
    write_json_atomic(p, {"b": 2, "a": 1})
    assert json.loads(p.read_text()) == {"a": 1, "b": 2}  # sorted keys, parses


def test_write_json_atomic_leaves_no_tmp_file(tmp_path: Path) -> None:
    p = tmp_path / "x.json"
    write_json_atomic(p, {"k": 1})
    siblings = list(tmp_path.iterdir())
    assert siblings == [p], f"stray temp files: {siblings}"


def test_atomic_write_text_overwrites(tmp_path: Path) -> None:
    p = tmp_path / "y.txt"
    atomic_write_text(p, "first")
    atomic_write_text(p, "second")
    assert p.read_text() == "second"
    assert list(tmp_path.iterdir()) == [p]
