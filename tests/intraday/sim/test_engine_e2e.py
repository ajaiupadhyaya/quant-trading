from datetime import UTC, datetime, timedelta

from quant.intraday.data.events import QuoteBar
from quant.intraday.sim.engine import BacktestEngine, EngineConfig
from quant.intraday.strategy import Order, Side


def _q(day, sec, bid, ask):
    return QuoteBar(
        ts=datetime(2023, 6, day, 20, 0, sec, tzinfo=UTC),
        symbol="AAPL",
        bid=bid,
        ask=ask,
        bid_size=100,
        ask_size=100,
    )


class BuyDay1SellDay2:
    def on_event(self, event, ctx):
        if event.ts.day == 1 and ctx.position("AAPL") == 0:
            return [Order("AAPL", Side.BUY, 100)]
        if event.ts.day == 2 and ctx.position("AAPL") > 0:
            return [Order("AAPL", Side.SELL, 100)]
        return []


def _events():
    return [
        _q(1, 0, 99.98, 100.02),
        _q(1, 1, 99.98, 100.02),
        _q(2, 0, 100.98, 101.02),
        _q(2, 1, 100.98, 101.02),
    ]


def test_costs_are_always_charged_and_result_is_deterministic():
    cfg = EngineConfig(latency=timedelta(milliseconds=250), commission_per_share=0.01)
    r1 = BacktestEngine(cfg).run(
        BuyDay1SellDay2(), _events(), adv_dollar={"AAPL": 1e9}, impact_coef_bps=10.0
    )
    r2 = BacktestEngine(cfg).run(
        BuyDay1SellDay2(), _events(), adv_dollar={"AAPL": 1e9}, impact_coef_bps=10.0
    )
    assert len(r1.fills) == 2
    assert r1.costs.commission == 2.0  # 2 fills * 100 sh * 0.01
    assert r1.costs.spread > 0.0  # crossed the spread both ways
    assert r1.equity_curve.equals(r2.equity_curve)  # deterministic
    assert r1.daily_returns().abs().sum() > 0.0
