"""Shared helpers for the continuous-engine tests (offline; no network/Claude)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from quant.engine.state import MarketState


@pytest.fixture(autouse=True)
def _hermetic_heavy_readers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep loop tests offline: the default fundamentals_fn / macro_nowcast_fn /
    vol_surface_fn would otherwise issue ~20 cold SEC EDGAR fetches, ~10 FRED
    reads, and a SPY option-chain pull per refresh. Tests that need specific
    values pass an explicit *_fn (which bypasses these module globals); tests
    that don't never assert on these reads."""
    import quant.engine.loop as lp

    monkeypatch.setattr(lp, "live_fundamentals", lambda *_a, **_k: None, raising=False)
    monkeypatch.setattr(lp, "live_macro_nowcast", lambda *_a, **_k: None, raising=False)
    monkeypatch.setattr(lp, "live_vol_surface", lambda *_a, **_k: None, raising=False)
    monkeypatch.setattr(lp, "live_vol_forecast", lambda *_a, **_k: None, raising=False)


def mk_state(**over: Any) -> MarketState:
    """A baseline calm-market MarketState; override any field for a scenario."""
    base: dict[str, Any] = dict(
        at="2026-06-03T14:00:00+00:00",
        asof="2026-06-03",
        session_phase="rth",
        is_trading_day=True,
        composite_score=-0.05,
        composite_label="neutral",
        coverage=1.0,
        breadth=0.75,
        median_mom=0.05,
        avg_corr=0.20,
        regime_label="calm",
        p_crisis=0.05,
        vix=16.0,
        realized_vol=0.10,
        vol_regime="calm",
        curve_label="normal",
        term_spread=0.40,
        intraday_spy_ret=0.001,
        intraday_breadth=0.625,
        intraday_range_vol=0.10,
        intraday_dispersion=0.004,
        intraday_asof="2026-06-03T14:00:00+00:00",
        news_sentiment=0.05,
        news_n_items=12,
        news_negative_frac=0.25,
        news_top_negative=None,
        next_event="FOMC",
        days_to_event=10,
        in_event_window=False,
        policy_uncertainty=150.0,
        financial_conditions=-0.4,
        vix_term_structure=1.15,
        macro_risk_label="calm",
        fund_coverage=0.9,
        equity_earnings_yield=0.045,
        equity_book_to_market=0.35,
        equity_gross_profitability=0.30,
        equity_asset_growth=0.06,
        valuation_label="fair",
        fund_quality_label="neutral",
        macro_cycle_label="expansion",
        recession_risk=0.15,
        recession_risk_label="low",
        hy_oas=3.2,
        credit_spread_baa_aaa=0.8,
        term_spread_10y3m=0.55,
        sahm=0.10,
        iv_atm_30d=0.165,
        iv_term_slope=0.01,
        put_skew=0.045,
        iv_regime="normal",
        vol_tail_label="benign",
        vol_forecast_ann=0.14,
        vol_forecast_regime="normal",
        vol_forecast_vs_realized=0.05,
        equity=1_000_000.0,
        n_positions=3,
        port_ann_vol=0.12,
        port_var_95=0.018,
        port_cvar_95=0.026,
        port_beta=0.60,
        top_name_weight=0.34,
        halt_active=False,
        worst_severity="ok",
        live_strategies=("defensive-etf-allocation",),
        degraded=(),
    )
    base.update(over)
    return MarketState(**base)


class SpySlack:
    def __init__(self) -> None:
        self.msgs: list[str] = []

    def send_slack(self, text: str, blocks: Any | None = None) -> bool:
        self.msgs.append(text)
        return True


def fake_settings(tmp_path: Path, **over: Any) -> SimpleNamespace:
    base: dict[str, Any] = dict(
        data_dir=tmp_path,
        slack_webhook_url=None,
        anthropic_api_key=None,
        anthropic_model_fast="claude-haiku-4-5",
        anthropic_model="claude-opus-4-8",
    )
    base.update(over)
    return SimpleNamespace(**base)
