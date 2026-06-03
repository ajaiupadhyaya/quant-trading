"""Shared helpers for the continuous-engine tests (offline; no network/Claude)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from quant.engine.state import MarketState


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
