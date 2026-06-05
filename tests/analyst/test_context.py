"""Read-only analyst context gathering — fail-open on every source."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from quant.analyst.context import AnalystContext, gather_analyst_context, render_context
from quant.governance.models import GovernanceState, StrategyState, ValidationEvidence
from quant.governance.store import (
    allocation_path,
    strategy_states_path,
    validation_manifest_path,
    write_allocation,
    write_strategy_states,
    write_validation_manifest,
)
from quant.risk.portfolio import PortfolioRisk

ASOF = date(2026, 6, 2)
SLUG = "defensive-etf-allocation"


def test_empty_data_dir_is_failopen(tmp_path: Path) -> None:
    # Nothing on disk → context is fully default, never raises.
    ctx = gather_analyst_context(tmp_path, ASOF, include_macro=False)
    assert ctx.asof == ASOF
    assert ctx.regime is None
    assert ctx.allocation == {}
    assert ctx.evidence == []
    assert ctx.recon is None
    # render still works on an empty context
    text = render_context(ctx)
    assert "Regime: unavailable" in text


def _write_regime(data_dir: Path) -> None:
    rp = data_dir / "regime"
    rp.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "p_calm": [0.7, 0.6],
            "p_choppy": [0.2, 0.25],
            "p_crisis": [0.1, 0.15],
            "label": ["calm", "calm"],
        },
        index=pd.to_datetime(["2026-05-29", "2026-06-01"]),
    )
    df.to_parquet(rp / "regime_series.parquet")


def test_gathers_regime_allocation_and_evidence(tmp_path: Path) -> None:
    _write_regime(tmp_path)
    write_allocation(allocation_path(tmp_path), {SLUG: 1.0})
    write_validation_manifest(
        validation_manifest_path(tmp_path),
        {
            SLUG: ValidationEvidence(
                slug=SLUG,
                run_date=date(2026, 5, 1),
                data_start=date(2018, 1, 1),
                data_end=date(2026, 4, 30),
                gate_deflated_sharpe=True,
                gate_probabilistic_sharpe=True,
                gate_bootstrap_lower=True,
                gate_regime=True,
                gate_holdout=True,
                deflated_sharpe=0.85,
                probabilistic_sharpe=0.97,
                bootstrap_total_return_p05=0.05,
                n_positive_regimes=3,
                n_tested_regimes=3,
                holdout_total_return=0.1,
                chosen_params_path="x.json",
                walkforward_path="y.parquet",
                provenance="test",
            )
        },
    )
    write_strategy_states(
        strategy_states_path(tmp_path),
        {
            SLUG: StrategyState(
                slug=SLUG,
                state=GovernanceState.LIVE,
                evaluated_at=datetime(2026, 5, 1, tzinfo=UTC),
                validation_age_days=12,
                reason="all gates pass",
                code_enabled_live=True,
            )
        },
    )

    ctx = gather_analyst_context(tmp_path, ASOF, include_macro=False)

    assert ctx.regime is not None
    assert ctx.regime.label == "calm"
    assert ctx.regime.p_calm == 0.6  # last row
    assert ctx.allocation == {SLUG: 1.0}
    assert len(ctx.evidence) == 1
    ev = ctx.evidence[0]
    assert ev.slug == SLUG
    assert ev.state == "live"
    assert (ev.gates_passed, ev.gates_total) == (5, 5)
    assert ev.deflated_sharpe == 0.85
    assert ev.validation_age_days == 12

    text = render_context(ctx)
    assert "Regime: calm" in text
    assert SLUG in text
    assert "gates 5/5" in text
    assert "DSR 0.85" in text


def test_render_includes_portfolio_risk() -> None:
    pr = PortfolioRisk(
        n_positions=3,
        gross_exposure=1.0,
        net_exposure=1.0,
        ann_vol=0.12,
        var_95=0.02,
        cvar_95=0.03,
        beta_to_benchmark=0.8,
        top_name_weight=0.4,
        lookback_days=180,
    )
    text = render_context(AnalystContext(asof=ASOF, portfolio_risk=pr))
    assert "Portfolio risk:" in text
    assert "VaR95" in text


def test_render_includes_fundamentals() -> None:
    from quant.fundamentals.factors import FundamentalRow, compute_fundamentals

    rows = [FundamentalRow(f"S{i}", 100.0, 1e12, 0.08, 0.5, 0.40, 0.05) for i in range(8)]
    read = compute_fundamentals(rows, asof=ASOF)
    text = render_context(AnalystContext(asof=ASOF, fundamentals=read))
    assert "Fundamentals:" in text
    assert "valuation=cheap" in text


def test_render_includes_macro_nowcast() -> None:
    from quant.macro.nowcast import compute_macro_nowcast

    n = compute_macro_nowcast(
        ASOF, t10y3m=-0.4, hy_oas=4.5, nfci=0.1, claims=230_000, claims_year_low=210_000, sahm=0.2
    )
    text = render_context(AnalystContext(asof=ASOF, macro_nowcast=n))
    assert "Macro nowcast:" in text
    assert "cycle=late-cycle" in text


def test_render_includes_vol_surface() -> None:
    from datetime import timedelta

    from quant.options.pricing import bs_price
    from quant.options.surface import OptionQuote, compute_vol_surface

    def mk(dte, strike, right, vol):
        return OptionQuote(
            ASOF + timedelta(days=dte),
            strike,
            right,
            bs_price(750, strike, dte / 365, vol, 0.045, 0.013, right),
        )

    quotes = [mk(28, 750, "call", 0.16), mk(28, 712.5, "put", 0.22), mk(28, 787.5, "call", 0.14)]
    vs = compute_vol_surface(quotes, 750.0, ASOF)
    text = render_context(AnalystContext(asof=ASOF, vol_surface=vs))
    assert "Vol surface:" in text
    assert "iv_regime=normal" in text


def test_render_includes_vol_forecast() -> None:
    import numpy as np

    from quant.forecast.vol import compute_vol_forecast

    rng = np.random.default_rng(1)
    close = 100 * np.exp(np.cumsum(0.01 * rng.standard_normal(400)))
    fc = compute_vol_forecast(close, ASOF, symbol="SPY", oos_skill="beats EWMA")
    text = render_context(AnalystContext(asof=ASOF, vol_forecast=fc))
    assert "Vol forecast:" in text
    assert "1d-ahead vol=" in text
