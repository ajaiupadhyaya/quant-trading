"""Append-only research log: round-trip, idempotency, robust reads."""

from __future__ import annotations

import json
from pathlib import Path

from quant.research.signals import (
    append_signals,
    build_market_signals,
    from_json_dict,
    read_latest_signals,
    to_json_dict,
)
from tests.research.conftest import close_panel, macro_series


def _rec(asof_offset: int = 0):
    panel = close_panel(seed=4)
    idx = panel.index
    vix = macro_series(18.0, idx)
    d10 = macro_series(4.0, idx)
    d2 = macro_series(4.2, idx)
    asof = idx[-1 - asof_offset].date()
    return build_market_signals(closes=panel, vix=vix, dgs10=d10, dgs2=d2, asof=asof)


def test_to_from_json_dict_roundtrip() -> None:
    rec = _rec()
    back = from_json_dict(to_json_dict(rec))
    assert to_json_dict(back) == to_json_dict(rec)


def test_append_and_read_latest(tmp_path: Path) -> None:
    p = tmp_path / "signals.jsonl"
    rec = _rec()
    append_signals(p, rec)
    latest = read_latest_signals(p)
    assert latest is not None
    assert latest.asof == rec.asof
    assert to_json_dict(latest) == to_json_dict(rec)


def test_append_lines_sorted_keys_no_nan(tmp_path: Path) -> None:
    p = tmp_path / "signals.jsonl"
    append_signals(p, _rec())
    line = p.read_text(encoding="utf-8").splitlines()[0]
    # sorted_keys + allow_nan=False were used: line must round-trip strictly.
    json.loads(line)
    assert line == json.dumps(json.loads(line), sort_keys=True)


def test_append_idempotent_same_asof(tmp_path: Path) -> None:
    p = tmp_path / "signals.jsonl"
    rec = _rec()
    append_signals(p, rec)
    append_signals(p, rec)  # same asof -> must not duplicate
    assert len([ln for ln in p.read_text().splitlines() if ln.strip()]) == 1


def test_append_distinct_asofs_accumulate(tmp_path: Path) -> None:
    p = tmp_path / "signals.jsonl"
    append_signals(p, _rec(asof_offset=1))
    append_signals(p, _rec(asof_offset=0))
    assert len([ln for ln in p.read_text().splitlines() if ln.strip()]) == 2
    assert read_latest_signals(p).asof == _rec(asof_offset=0).asof  # type: ignore[union-attr]


def test_read_latest_absent_returns_none(tmp_path: Path) -> None:
    assert read_latest_signals(tmp_path / "nope.jsonl") is None


def test_read_latest_skips_blank_and_malformed(tmp_path: Path) -> None:
    p = tmp_path / "signals.jsonl"
    append_signals(p, _rec())
    with p.open("a", encoding="utf-8") as fh:
        fh.write("\n")
        fh.write("{not valid json\n")
    latest = read_latest_signals(p)  # must fall back to the last valid line
    assert latest is not None


def test_append_best_effort_on_unwritable_path(tmp_path: Path) -> None:
    # A path whose parent is a FILE cannot be created; append must swallow, not raise.
    blocker = tmp_path / "blocker"
    blocker.write_text("x")
    append_signals(blocker / "sub" / "signals.jsonl", _rec())  # no exception
