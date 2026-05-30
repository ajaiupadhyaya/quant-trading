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


class ShortAndHold:
    def on_event(self, event, ctx):
        if event.ts.day == 1 and ctx.position("AAPL") == 0:
            return [Order("AAPL", Side.SELL, 100)]  # open a short, carry it overnight
        return []


def test_overnight_short_incurs_borrow_financing():
    events = [
        _q(1, 0, 99.98, 100.02),
        _q(1, 1, 99.98, 100.02),
        _q(2, 0, 99.98, 100.02),
        _q(2, 1, 99.98, 100.02),
    ]
    cfg = EngineConfig(
        latency=timedelta(milliseconds=250), commission_per_share=0.0, annual_borrow_bps=50.0
    )
    res = BacktestEngine(cfg).run(
        ShortAndHold(), events, adv_dollar={"AAPL": 1e12}, impact_coef_bps=0.0
    )
    assert res.costs.financing > 0.0  # one overnight borrow charge on the short notional
