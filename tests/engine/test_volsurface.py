"""Track F engine integration: vol-surface flow into state + tape-active gating."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from quant.engine.loop import EngineConfig, engine_dir, run_engine
from quant.options.surface import VolSurface
from tests.engine.conftest import fake_settings


def _clock(start: datetime, step: float):
    t = {"v": start - timedelta(seconds=step)}

    def now() -> datetime:
        t["v"] += timedelta(seconds=step)
        return t["v"]

    return now


def _surface() -> VolSurface:
    return VolSurface(
        asof="2026-06-04",
        spot=750.0,
        near_dte=30,
        far_dte=90,
        atm_iv_30d=0.17,
        atm_iv_90d=0.18,
        term_slope=0.01,
        put_skew=0.05,
        iv_regime="normal",
        term_label="contango",
        tail_label="elevated",
        n_quotes=120,
        n_expiries=6,
    )


def test_vol_surface_flows_into_state_and_throttles(tmp_path: Path) -> None:
    calls = {"n": 0}

    def vol_fn(_d: date):
        calls["n"] += 1
        return _surface()

    run_engine(
        fake_settings(tmp_path),
        max_cycles=3,
        dry_run=True,
        sleep=lambda _x: None,
        now_fn=_clock(datetime(2026, 6, 4, 14, tzinfo=UTC), 46),  # RTH
        positions_fn=lambda: None,
        equity_fn=lambda: None,
        intraday_fn=lambda: None,
        news_fn=lambda: None,
        eventrisk_fn=lambda _d: None,
        vol_surface_fn=vol_fn,
        config=EngineConfig(vol_surface_refresh_s=1800.0),  # 46s steps -> fetched once
    )
    assert calls["n"] == 1  # throttled across the 3 cycles
    state = json.loads((engine_dir(tmp_path) / "state.json").read_text())
    assert state["iv_regime"] == "normal"
    assert state["iv_atm_30d"] == 0.17
    assert state["vol_tail_label"] == "elevated"


def test_vol_surface_skipped_when_market_closed(tmp_path: Path) -> None:
    calls = {"n": 0}

    def vol_fn(_d: date):
        calls["n"] += 1
        return _surface()

    run_engine(
        fake_settings(tmp_path),
        max_cycles=1,
        dry_run=True,
        sleep=lambda _x: None,
        now_fn=_clock(
            datetime(2026, 6, 4, 3, tzinfo=UTC), 46
        ),  # 03:00 UTC -> closed (pre-premarket ET)
        positions_fn=lambda: None,
        equity_fn=lambda: None,
        intraday_fn=lambda: None,
        news_fn=lambda: None,
        eventrisk_fn=lambda _d: None,
        vol_surface_fn=vol_fn,
    )
    assert calls["n"] == 0  # the heavy chain fetch is skipped overnight
    state = json.loads((engine_dir(tmp_path) / "state.json").read_text())
    assert state["iv_regime"] is None
