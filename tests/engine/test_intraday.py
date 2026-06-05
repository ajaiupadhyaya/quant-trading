"""Phase 7A intraday signals: snapshot math, fail-open fetch, MarketState + events."""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from quant.engine import intraday as it
from quant.engine.events import EventConfig, detect_events
from quant.engine.intraday import compute_intraday_signals, live_intraday_signals
from quant.engine.state import build_market_state, render_state
from tests.engine.conftest import mk_state

CFG = EventConfig()


def _snap(**rows):
    base = {
        "SPY": {"price": 100.0, "prev_close": 100.0, "high": 101.0, "low": 99.0, "minute_ts": "t1"},
        "GLD": {"price": 100.0, "prev_close": 100.0, "high": 100.5, "low": 99.5, "minute_ts": "t1"},
    }
    base.update(rows)
    return base


def test_intraday_return_and_breadth() -> None:
    snap = _snap(
        SPY={"price": 98.0, "prev_close": 100.0, "high": 100.0, "low": 97.5, "minute_ts": "t"},
        GLD={"price": 102.0, "prev_close": 100.0, "high": 102.5, "low": 99.5, "minute_ts": "t"},
    )
    s = compute_intraday_signals(snap)
    assert abs(s.spy_ret - (-0.02)) < 1e-9
    assert s.n_up == 1 and s.n_down == 1
    assert abs(s.breadth - 0.5) < 1e-9


def test_parkinson_range_vol_matches_formula() -> None:
    snap = _snap(
        SPY={"price": 100.0, "prev_close": 100.0, "high": 105.0, "low": 95.0, "minute_ts": "t"}
    )
    s = compute_intraday_signals(snap)
    expected = (math.log(105.0 / 95.0) / (2 * math.sqrt(math.log(2)))) * math.sqrt(252)
    assert abs(s.range_vol - expected) < 1e-9


def test_dispersion_and_asof_from_market() -> None:
    snap = _snap(
        SPY={
            "price": 101.0,
            "prev_close": 100.0,
            "high": 101.0,
            "low": 100.0,
            "minute_ts": "spy-ts",
        }
    )
    s = compute_intraday_signals(snap)
    assert s.dispersion is not None and s.dispersion >= 0
    assert s.asof_minute == "spy-ts"  # prefers the market symbol's stamp


def test_failopen_on_none_and_empty() -> None:
    assert compute_intraday_signals(None).spy_ret is None
    assert compute_intraday_signals({}).n_symbols == 0


def test_missing_prev_close_is_skipped() -> None:
    snap = {
        "SPY": {"price": 100.0, "prev_close": None, "high": 101.0, "low": 99.0, "minute_ts": "t"}
    }
    s = compute_intraday_signals(snap)
    assert s.spy_ret is None  # no valid return without prev_close
    assert s.n_symbols == 0


def test_zero_or_negative_prev_close_guarded() -> None:
    snap = {
        "SPY": {"price": 100.0, "prev_close": 0.0, "high": 101.0, "low": 99.0, "minute_ts": "t"}
    }
    assert compute_intraday_signals(snap).spy_ret is None


def test_fetch_failopen_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    # The data client import/call raising must degrade to None, never raise.
    settings = SimpleNamespace(alpaca_api_key="k", alpaca_secret_key="s")
    monkeypatch.setattr(
        it, "_with_timeout", lambda fn, seconds: (_ for _ in ()).throw(RuntimeError())
    )
    assert it.fetch_intraday_snapshot(settings, ["SPY"]) is None


def test_live_intraday_failopen(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(it, "fetch_intraday_snapshot", lambda *a, **k: None)
    s = live_intraday_signals(SimpleNamespace())
    assert s.spy_ret is None and s.n_symbols == 0


# --- MarketState + render integration ---
def test_build_state_carries_intraday(tmp_path: Path) -> None:
    intra = compute_intraday_signals(
        {"SPY": {"price": 97.0, "prev_close": 100.0, "high": 100.0, "low": 96.5, "minute_ts": "t"}}
    )
    s = build_market_state(tmp_path, asof=date(2026, 6, 3), intraday=intra)
    assert s.intraday_spy_ret is not None and abs(s.intraday_spy_ret - (-0.03)) < 1e-9
    assert "SPYday=" in render_state(s)


# --- new event detectors ---
def test_intraday_selloff_warn_and_critical() -> None:
    warn = detect_events(mk_state(), mk_state(intraday_spy_ret=-0.02), CFG)
    assert next(e for e in warn if e.code == "intraday_selloff").severity == "warn"
    crit = detect_events(mk_state(), mk_state(intraday_spy_ret=-0.04), CFG)
    assert next(e for e in crit if e.code == "intraday_selloff").severity == "critical"
    assert detect_events(mk_state(), mk_state(intraday_spy_ret=0.005), CFG) == []


def test_intraday_breadth_break() -> None:
    evs = detect_events(mk_state(), mk_state(intraday_breadth=0.2), CFG)
    assert any(e.code == "intraday_breadth_break" for e in evs)
    assert not any(
        e.code == "intraday_breadth_break"
        for e in detect_events(mk_state(), mk_state(intraday_breadth=0.6), CFG)
    )


def test_intraday_range_spike() -> None:
    warn = detect_events(mk_state(), mk_state(intraday_range_vol=0.35), CFG)
    assert next(e for e in warn if e.code == "intraday_range_spike").severity == "warn"
    crit = detect_events(mk_state(), mk_state(intraday_range_vol=0.55), CFG)
    assert next(e for e in crit if e.code == "intraday_range_spike").severity == "critical"


def test_intraday_detectors_are_absolute_on_first_cycle() -> None:
    # prev=None: an already-selling-off tape surfaces immediately.
    codes = {e.code for e in detect_events(None, mk_state(intraday_spy_ret=-0.04), CFG)}
    assert "intraday_selloff" in codes
