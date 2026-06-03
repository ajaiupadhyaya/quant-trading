"""MarketState: session phase, fail-open build, serialization, render."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from quant.engine import state as st
from quant.engine.state import (
    build_market_state,
    from_json_dict,
    render_state,
    session_phase,
    to_json_dict,
)
from tests.engine.conftest import mk_state


@pytest.mark.parametrize(
    "utc_hour, expected",
    [
        (8, "premarket"),  # 04:00 ET  -> before 07:00 is closed... 08 UTC = 04 ET -> closed
        (12, "premarket"),  # 08:00 ET
        (14, "rth"),  # 10:00 ET
        (19, "rth"),  # 15:00 ET
        (21, "afterhours"),  # 17:00 ET
        (3, "closed"),  # 23:00 ET prior day-ish / overnight
    ],
)
def test_session_phase_trading_day(utc_hour: int, expected: str) -> None:
    d = date(2026, 6, 3)  # a Wednesday (trading day)
    now = datetime(2026, 6, 3, utc_hour, 0, tzinfo=UTC)
    phase = session_phase(now, d)
    # 08 UTC = 04:00 ET which is before the 07:00 premarket start -> closed
    if utc_hour == 8:
        assert phase == "closed"
    else:
        assert phase == expected


def test_session_phase_weekend_is_closed() -> None:
    sat = date(2026, 6, 6)
    assert session_phase(datetime(2026, 6, 6, 14, 0, tzinfo=UTC), sat) == "closed"


def test_build_state_failopen_on_empty_data_dir(tmp_path: Path) -> None:
    s = build_market_state(tmp_path, asof=date(2026, 6, 3), now_utc=datetime(2026, 6, 3, 14, tzinfo=UTC))
    assert s.halt_active is False
    assert s.composite_label is None  # no signals logged
    assert "signals" in s.degraded
    assert s.session_phase == "rth"


def test_build_state_never_raises_when_context_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import quant.analyst.context as ctxmod

    def _boom(*a: object, **k: object) -> object:
        raise RuntimeError("context exploded")

    monkeypatch.setattr(ctxmod, "gather_analyst_context", _boom)
    s = build_market_state(tmp_path, asof=date(2026, 6, 3))
    assert "context" in s.degraded  # degraded, but no exception escaped


def test_build_state_reads_logged_signals(tmp_path: Path) -> None:
    # Seed a signals log the way the live system does, then confirm it flows in.
    import numpy as np
    import pandas as pd

    from quant.research.signals import append_signals, build_market_signals, signals_path

    idx = pd.bdate_range("2022-01-03", periods=560)
    syms = ["SPY", "TLT", "IEF", "GLD", "DBC", "VNQ", "EFA", "EEM"]
    closes = pd.DataFrame(
        {s: 100.0 * np.exp(np.cumsum(np.full(len(idx), 0.0003))) for s in syms}, index=idx
    )
    rec = build_market_signals(
        closes=closes,
        vix=pd.Series(17.0, index=idx),
        dgs10=pd.Series(4.0, index=idx),
        dgs2=pd.Series(4.2, index=idx),
        asof=idx[-1].date(),
    )
    append_signals(signals_path(tmp_path), rec)
    s = build_market_state(tmp_path, asof=idx[-1].date())
    assert s.composite_label is not None
    assert "signals" not in s.degraded


def test_reads_halt_from_monitor_status(tmp_path: Path) -> None:
    import json

    from quant.monitor.status import monitor_status_path

    p = monitor_status_path(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"halt_active": True, "worst_severity": "halt"}), encoding="utf-8")
    s = build_market_state(tmp_path, asof=date(2026, 6, 3))
    assert s.halt_active is True
    assert s.worst_severity == "halt"


def test_json_roundtrip() -> None:
    s = mk_state()
    assert from_json_dict(to_json_dict(s)) == s


def test_render_contains_key_fields() -> None:
    text = render_state(mk_state(halt_active=True))
    assert "posture=neutral" in text
    assert "HALT" in text


def test_non_finite_sanitized_to_none() -> None:
    # NaN/inf risk metrics (degenerate covariance, bad broker equity) must become
    # None so state.json is valid JSON and the drawdown guard never sees NaN.
    assert st._f(float("nan")) is None
    assert st._f(float("inf")) is None
    assert st._f(float("-inf")) is None
    assert st._f(3.5) == 3.5


def test_build_state_sanitizes_nan_equity(tmp_path: Path) -> None:
    import json

    s = build_market_state(tmp_path, asof=date(2026, 6, 3), equity=float("nan"))
    assert s.equity is None  # routed through _f, not persisted as NaN
    json.dumps(to_json_dict(s), allow_nan=False)  # would raise if a NaN slipped through


def test_monitor_read_failopen_on_garbage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from quant.monitor.status import monitor_status_path

    p = monitor_status_path(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{garbage", encoding="utf-8")
    halt, sev = st._read_monitor(tmp_path)
    assert halt is False and sev is None
