"""Pure unit tests for the scenario/stress engine (no data/network)."""

from __future__ import annotations

import pandas as pd

from quant.risk.scenarios import (
    HistoricalScenario,
    HypotheticalScenario,
    StressReport,
    compute_stress,
    default_scenarios,
)


def _panel(dates, data):
    return pd.DataFrame(data, index=pd.to_datetime(dates))


def test_hypothetical_class_level_shock():
    # 50% SPY (equity), 50% TLT (bond); equity -20%, bond +5%
    weights = {"SPY": 0.5, "TLT": 0.5}
    scen = HypotheticalScenario("eq-crash", {"equity": -0.20, "bond": 0.05})
    rep = compute_stress(weights, returns=pd.DataFrame(), scenarios=[scen])
    res = rep.results[0]
    assert res.computable
    assert res.pnl_pct == 0.5 * -0.20 + 0.5 * 0.05  # = -0.075
    assert res.by_class["equity"] == 0.5 * -0.20
    assert res.by_class["bond"] == 0.5 * 0.05


def test_symbol_override_beats_class():
    # rate shock: bond class -7% but TLT overridden to -15%
    weights = {"TLT": 1.0}
    scen = HypotheticalScenario("rate", {"bond": -0.07, "TLT": -0.15})
    rep = compute_stress(weights, returns=pd.DataFrame(), scenarios=[scen])
    assert rep.results[0].pnl_pct == -0.15


def test_unshocked_asset_contributes_zero():
    weights = {"SPY": 0.5, "GLD": 0.5}
    scen = HypotheticalScenario("eq-only", {"equity": -0.10})  # no gold shock
    rep = compute_stress(weights, returns=pd.DataFrame(), scenarios=[scen])
    res = rep.results[0]
    assert res.pnl_pct == 0.5 * -0.10  # GLD contributes 0
    assert "GLD" in res.missing_symbols


def test_historical_replay_cumulative_return():
    # SPY daily returns over the window: +10% then -20% => cumret = 1.1*0.8-1 = -0.12
    dates = ["2008-09-01", "2008-09-02", "2008-09-03"]
    returns = _panel(dates, {"SPY": [0.0, 0.10, -0.20]})
    weights = {"SPY": 1.0}
    scen = HistoricalScenario(
        "gfc",
        start=pd.Timestamp("2008-09-02").date(),
        end=pd.Timestamp("2008-09-03").date(),
    )
    rep = compute_stress(weights, returns=returns, scenarios=[scen])
    res = rep.results[0]
    assert res.computable
    assert abs(res.pnl_pct - (1.10 * 0.80 - 1.0)) < 1e-12  # -0.12


def test_historical_missing_symbol_degrades_but_still_computes():
    dates = ["2008-09-02", "2008-09-03"]
    returns = _panel(dates, {"SPY": [0.10, -0.20]})  # no TLT column
    weights = {"SPY": 0.5, "TLT": 0.5}
    scen = HistoricalScenario(
        "gfc",
        start=pd.Timestamp("2008-09-02").date(),
        end=pd.Timestamp("2008-09-03").date(),
    )
    rep = compute_stress(weights, returns=returns, scenarios=[scen])
    res = rep.results[0]
    assert res.computable  # SPY computed
    assert "TLT" in res.missing_symbols
    assert abs(res.pnl_pct - 0.5 * (1.10 * 0.80 - 1.0)) < 1e-12  # only SPY half


def test_worst_loss_and_scenario_selection():
    weights = {"SPY": 1.0}
    scens = [
        HypotheticalScenario("mild", {"equity": -0.05}),
        HypotheticalScenario("severe", {"equity": -0.30}),
        HypotheticalScenario("gain", {"equity": 0.10}),
    ]
    rep = compute_stress(weights, returns=pd.DataFrame(), scenarios=scens)
    assert rep.worst_scenario == "severe"
    assert abs(rep.worst_loss - 0.30) < 1e-12  # positive = loss
    assert rep.computable


def test_all_gains_worst_loss_is_negative():
    weights = {"SPY": 1.0}
    rep = compute_stress(
        weights,
        returns=pd.DataFrame(),
        scenarios=[HypotheticalScenario("g", {"equity": 0.10})],
    )
    assert rep.worst_loss < 0  # best-case-only => "loss" is negative


def test_empty_weights_not_computable():
    rep = compute_stress(
        {},
        returns=pd.DataFrame(),
        scenarios=[HypotheticalScenario("x", {"equity": -0.2})],
    )
    assert not rep.computable
    assert rep.worst_loss is None
    assert rep.worst_scenario is None


def test_default_scenarios_library_shape():
    scens = default_scenarios()
    names = {s.name for s in scens}
    assert {"2008-GFC", "2020-COVID", "2022-rate-selloff", "2013-taper-tantrum"} <= names
    assert {
        "equity-crash-20",
        "rate-shock-+100bp",
        "stagflation",
        "risk-off-flight",
    } <= names
    for s in scens:
        if isinstance(s, HistoricalScenario):
            assert s.start < s.end
        else:
            assert isinstance(s, HypotheticalScenario) and s.shocks


def test_render_non_empty_and_degrades():
    weights = {"SPY": 1.0}
    rep = compute_stress(
        weights,
        returns=pd.DataFrame(),
        scenarios=[HypotheticalScenario("severe", {"equity": -0.30})],
    )
    text = rep.render()
    assert "severe" in text
    # a fully-degraded report renders an n/a sentinel, not a crash
    empty = StressReport(results=(), worst_loss=None, worst_scenario=None, computable=False)
    assert "n/a" in empty.render()


# --- gate violation mapping (build_portfolio_risk_gate stress kwarg) ----------

from quant.risk.portfolio import (  # noqa: E402
    PortfolioRisk,
    PortfolioRiskLimits,
    RiskGateMode,
    build_portfolio_risk_gate,
)


def _flat_risk():
    return PortfolioRisk(
        n_positions=1,
        gross_exposure=1.0,
        net_exposure=1.0,
        ann_vol=0.10,
        var_95=0.01,
        cvar_95=0.02,
        beta_to_benchmark=0.5,
        top_name_weight=1.0,
        lookback_days=180,
        sector_exposure={"equity": 1.0},
        computable=True,
    )


def test_stress_within_limit_no_violation():
    rep = StressReport(results=(), worst_loss=0.20, worst_scenario="x", computable=True)
    gate = build_portfolio_risk_gate(
        _flat_risk(), limits=PortfolioRiskLimits(), mode=RiskGateMode.WARN, stress=rep
    )
    assert gate.ok
    assert all(v.code != "stress" for v in gate.violations)
    assert gate.stress is rep


def test_stress_over_limit_warn_violation():
    rep = StressReport(results=(), worst_loss=0.45, worst_scenario="2008-GFC", computable=True)
    gate = build_portfolio_risk_gate(
        _flat_risk(),
        limits=PortfolioRiskLimits(max_scenario_loss=0.30),
        mode=RiskGateMode.WARN,
        stress=rep,
    )
    assert not gate.ok
    assert any(v.code == "stress" for v in gate.violations)
    assert gate.severity == "warn"  # WARN never blocks


def test_stress_none_worst_never_violates():
    rep = StressReport(results=(), worst_loss=None, worst_scenario=None, computable=False)
    gate = build_portfolio_risk_gate(
        _flat_risk(), limits=PortfolioRiskLimits(), mode=RiskGateMode.WARN, stress=rep
    )
    assert all(v.code != "stress" for v in gate.violations)


def test_gate_back_compat_no_stress():
    # Omitting stress preserves prior behavior and gate.stress is None.
    gate = build_portfolio_risk_gate(
        _flat_risk(), limits=PortfolioRiskLimits(), mode=RiskGateMode.WARN
    )
    assert gate.stress is None
    assert gate.ok
