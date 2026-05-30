import pandas as pd

from quant.intraday.sim.result import BacktestResult, CostBreakdown


def _equity():
    idx = pd.to_datetime(["2023-06-01T20:00:00Z", "2023-06-02T20:00:00Z", "2023-06-05T20:00:00Z"])
    return pd.Series([100_000.0, 101_000.0, 100_500.0], index=idx)


def test_daily_returns_from_equity():
    r = BacktestResult(
        equity_curve=_equity(), fills=[], costs=CostBreakdown(0, 0, 0, 0)
    ).daily_returns()
    assert len(r) == 2
    assert round(r.iloc[0], 5) == round(1_000 / 100_000, 5)


def test_sharpe_runs_on_daily_returns():
    res = BacktestResult(equity_curve=_equity(), fills=[], costs=CostBreakdown(1, 2, 3, 4))
    s = res.sharpe()
    assert isinstance(s, float)
    assert res.costs.total == 10.0
