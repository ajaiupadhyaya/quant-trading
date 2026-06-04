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


def test_fundamentals_flow_into_state(tmp_path: Path) -> None:
    from quant.fundamentals.factors import FundamentalRow, compute_fundamentals

    rows = [FundamentalRow(f"S{i}", 100.0, 1e12, 0.08, 0.5, 0.40, 0.05) for i in range(8)]
    read = compute_fundamentals(rows, asof=date(2026, 6, 3))
    s = build_market_state(tmp_path, asof=date(2026, 6, 3), fundamentals=read)
    assert s.valuation_label == "cheap"
    assert s.fund_quality_label == "strong"
    assert s.fund_coverage == 1.0
    assert abs(s.equity_earnings_yield - 0.08) < 1e-12
    # render shows the valuation posture
    assert "val=cheap" in render_state(s)


def test_fundamentals_absent_leaves_fields_none(tmp_path: Path) -> None:
    s = build_market_state(tmp_path, asof=date(2026, 6, 3))  # no fundamentals passed
    assert s.valuation_label is None
    assert s.fund_coverage is None


def test_macro_nowcast_flows_into_state(tmp_path: Path) -> None:
    from quant.macro.nowcast import compute_macro_nowcast

    nowcast = compute_macro_nowcast(
        date(2026, 6, 3), t10y3m=-0.4, hy_oas=4.5, nfci=0.1, claims=230_000,
        claims_year_low=210_000, sahm=0.2, baa=5.6, aaa=4.7,
    )
    s = build_market_state(tmp_path, asof=date(2026, 6, 3), macro_nowcast=nowcast)
    assert s.macro_cycle_label == "late-cycle"
    assert s.hy_oas == 4.5
    assert abs(s.credit_spread_baa_aaa - 0.9) < 1e-9
    assert s.term_spread_10y3m == -0.4
    assert "cycle=late-cycle" in render_state(s)


def test_macro_nowcast_absent_leaves_fields_none(tmp_path: Path) -> None:
    s = build_market_state(tmp_path, asof=date(2026, 6, 3))
    assert s.macro_cycle_label is None
    assert s.recession_risk is None


def test_vol_surface_flows_into_state(tmp_path: Path) -> None:
    from datetime import timedelta

    from quant.options.pricing import bs_price
    from quant.options.surface import OptionQuote, compute_vol_surface

    asof = date(2026, 6, 3)

    def mk(dte, strike, right, vol):
        return OptionQuote(asof + timedelta(days=dte), strike, right, bs_price(750, strike, dte / 365, vol, 0.045, 0.013, right))

    quotes = [mk(28, 750, "call", 0.16), mk(28, 712.5, "put", 0.22), mk(28, 787.5, "call", 0.14), mk(88, 750, "call", 0.17)]
    vs = compute_vol_surface(quotes, 750.0, asof)
    s = build_market_state(tmp_path, asof=asof, vol_surface=vs)
    assert s.iv_regime == "normal"
    assert abs(s.iv_atm_30d - 0.16) < 1e-3
    assert s.vol_tail_label == "elevated"
    assert "iv=normal" in render_state(s)


def test_vol_surface_absent_leaves_fields_none(tmp_path: Path) -> None:
    s = build_market_state(tmp_path, asof=date(2026, 6, 3))
    assert s.iv_regime is None and s.iv_atm_30d is None


def test_vol_forecast_flows_into_state(tmp_path: Path) -> None:
    import numpy as np

    from quant.forecast.vol import compute_vol_forecast

    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(0.01 * rng.standard_normal(400)))
    fc = compute_vol_forecast(close, date(2026, 6, 3), symbol="SPY")
    s = build_market_state(tmp_path, asof=date(2026, 6, 3), vol_forecast=fc)
    assert s.vol_forecast_ann is not None and s.vol_forecast_ann > 0
    assert s.vol_forecast_regime in {"calm", "normal", "elevated", "stressed"}
    assert "fcast_vol=" in render_state(s)


def test_vol_forecast_absent_leaves_fields_none(tmp_path: Path) -> None:
    s = build_market_state(tmp_path, asof=date(2026, 6, 3))
    assert s.vol_forecast_ann is None and s.vol_forecast_regime is None


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
