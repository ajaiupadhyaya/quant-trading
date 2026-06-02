"""Portfolio risk analytics: exposures, tail risk (VaR/CVaR), beta, fail-open."""

from __future__ import annotations

import pandas as pd
import pytest

from quant.risk.portfolio import compute_portfolio_risk, weights_from_positions


def _panel() -> pd.DataFrame:
    a = [
        0.01, -0.01, 0.02, -0.02, 0.0, 0.01, -0.01, 0.03, -0.03, 0.0,
        0.01, -0.01, 0.02, -0.02, 0.0, 0.01, -0.01, 0.02, -0.02, 0.0,
    ]
    b = [
        -0.005, 0.005, -0.01, 0.01, 0.0, -0.005, 0.005, -0.015, 0.015, 0.0,
        -0.005, 0.005, -0.01, 0.01, 0.0, -0.005, 0.005, -0.01, 0.01, 0.0,
    ]
    idx = pd.bdate_range("2024-01-01", periods=len(a))
    return pd.DataFrame({"A": a, "B": b}, index=idx)


def test_weights_from_positions() -> None:
    w = weights_from_positions({"SPY": 10, "QQQ": -5, "ZERO": 0}, {"SPY": 100, "QQQ": 200}, 10_000)
    assert w == {"SPY": pytest.approx(0.1), "QQQ": pytest.approx(-0.1)}
    assert weights_from_positions({"SPY": 10}, {"SPY": 100}, 0) == {}


def test_exposures_and_tail_risk() -> None:
    res = compute_portfolio_risk({"A": 0.6, "B": -0.4}, _panel())
    assert res.n_positions == 2
    assert res.gross_exposure == pytest.approx(1.0)
    assert res.net_exposure == pytest.approx(0.2)
    assert res.top_name_weight == pytest.approx(0.6)
    assert res.ann_vol is not None and res.ann_vol > 0
    assert res.var_95 is not None and res.cvar_95 is not None
    # Expected shortfall is never below the VaR quantile loss.
    assert res.cvar_95 >= res.var_95 - 1e-9
    assert res.lookback_days == len(_panel())
    assert "VaR95" in res.render()


def test_beta_unit_when_portfolio_equals_benchmark() -> None:
    r = _panel()
    res = compute_portfolio_risk({"A": 1.0}, r, benchmark=r["A"])
    assert res.beta_to_benchmark == pytest.approx(1.0, abs=1e-6)


def test_degenerate_inputs_failopen() -> None:
    # Empty returns: exposures still computed from weights, tail metrics None.
    res = compute_portfolio_risk({"A": 1.0}, pd.DataFrame())
    assert res.gross_exposure == pytest.approx(1.0)
    assert res.var_95 is None and res.ann_vol is None and res.cvar_95 is None
    # Weights referencing symbols absent from the panel.
    res2 = compute_portfolio_risk({"ZZZ": 1.0}, _panel())
    assert res2.var_95 is None
    # No positions at all.
    res3 = compute_portfolio_risk({}, _panel())
    assert res3.n_positions == 0 and res3.var_95 is None
