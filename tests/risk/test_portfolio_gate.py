"""Portfolio risk GATE (Guard 5): pure WARN/BLOCK evaluator, defensive-sleeve calibration."""

from __future__ import annotations

from typing import Any

from quant.risk.portfolio import (
    PortfolioRisk,
    PortfolioRiskLimits,
    RiskGateMode,
    build_portfolio_risk_gate,
)


def _risk(**overrides: Any) -> PortfolioRisk:
    """A PortfolioRisk seeded with the live defensive-etf measured values."""
    base: dict[str, Any] = dict(
        n_positions=3,
        gross_exposure=1.0,
        net_exposure=1.0,
        ann_vol=0.170,
        var_95=0.0183,
        cvar_95=0.0264,
        beta_to_benchmark=0.586,
        top_name_weight=0.34,
        lookback_days=180,
        sector_exposure={"commodity": 0.333, "equity": 0.333, "gold": 0.334},
        computable=True,
        degraded_metrics=(),
    )
    base.update(overrides)
    return PortfolioRisk(**base)


def test_gate_ok_on_live_defensive_sleeve() -> None:
    """Bake-in criterion: defensive-etf records OK under default limits every run."""
    g = build_portfolio_risk_gate(_risk(), limits=PortfolioRiskLimits(), mode=RiskGateMode.WARN)
    assert g.ok is True
    assert g.severity == "ok"
    assert g.violations == ()
    assert "ok" in g.detail


def test_gate_each_metric_trips_independently() -> None:
    lim = PortfolioRiskLimits()
    cases = {
        "ann_vol": _risk(ann_vol=0.40),
        "var_95": _risk(var_95=0.06),
        "cvar_95": _risk(cvar_95=0.08),
        "beta": _risk(beta_to_benchmark=-1.6),  # checked on abs()
    }
    for code, risk in cases.items():
        g = build_portfolio_risk_gate(risk, limits=lim)
        assert {v.code for v in g.violations} == {code}, code
        assert g.ok is False
        assert g.severity == "warn"  # WARN mode never escalates to block

    g = build_portfolio_risk_gate(_risk(sector_exposure={"equity": 1.2}), limits=lim)
    assert [v.code for v in g.violations] == ["asset_class"]
    assert g.violations[0].bucket == "equity"


def test_gate_none_metrics_never_violate() -> None:
    g = build_portfolio_risk_gate(
        _risk(
            ann_vol=None,
            var_95=None,
            cvar_95=None,
            beta_to_benchmark=None,
            computable=False,
            sector_exposure={},
        ),
        limits=PortfolioRiskLimits(),
    )
    assert g.ok is True
    assert g.severity == "ok"


def test_gate_uncomputable_fail_closed_only_when_enabled() -> None:
    degraded = _risk(
        ann_vol=None,
        var_95=None,
        cvar_95=None,
        beta_to_benchmark=None,
        computable=False,
        sector_exposure={},
    )
    assert (
        build_portfolio_risk_gate(
            degraded, limits=PortfolioRiskLimits(fail_closed_on_uncomputable=False)
        ).ok
        is True
    )
    g = build_portfolio_risk_gate(
        degraded, limits=PortfolioRiskLimits(fail_closed_on_uncomputable=True)
    )
    assert g.ok is False
    assert g.violations[0].code == "uncomputable"


def test_gate_asset_class_cap_allows_100pct_single_class() -> None:
    """Risk-off is 100% defensive by design — a single asset class at 1.0 must pass."""
    lim = PortfolioRiskLimits()  # max_asset_class_weight=1.0
    assert build_portfolio_risk_gate(_risk(sector_exposure={"bond": 1.0}), limits=lim).ok is True
    assert build_portfolio_risk_gate(_risk(sector_exposure={"bond": 1.01}), limits=lim).ok is False


def test_gate_block_mode_marks_severity_block_but_mutates_nothing() -> None:
    risk = _risk(ann_vol=0.40)
    g = build_portfolio_risk_gate(risk, limits=PortfolioRiskLimits(), mode=RiskGateMode.BLOCK)
    assert g.ok is False
    assert g.severity == "block"  # blocking is the caller's job; the gate only flags it
    assert g.risk is risk  # echoes the input untouched
