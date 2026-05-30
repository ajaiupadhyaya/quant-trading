from datetime import UTC, datetime, timedelta

from quant.intraday.data.events import QuoteBar
from quant.intraday.sim.engine import BacktestEngine, EngineConfig
from quant.intraday.strategy import Order, Side


def _q(sec, bid, ask):
    return QuoteBar(
        ts=datetime(2023, 6, 1, 13, 30, sec, tzinfo=UTC),
        symbol="AAPL",
        bid=bid,
        ask=ask,
        bid_size=10,
        ask_size=10,
    )


class BuyOnceStrategy:
    def __init__(self):
        self.fired = False

    def on_event(self, event, ctx):
        if not self.fired:
            self.fired = True
            return [Order("AAPL", Side.BUY, 100)]
        return []


def test_order_fills_at_post_latency_nbbo_not_signal_time():
    # order signalled at :00 (effective :00.250) fills against the :01 NBBO.
    events = [_q(0, 99.98, 100.02), _q(1, 100.40, 100.50)]
    eng = BacktestEngine(
        EngineConfig(latency=timedelta(milliseconds=250), commission_per_share=0.0)
    )
    res = eng.run(BuyOnceStrategy(), events, adv_dollar={"AAPL": 0.0}, impact_coef_bps=0.0)
    assert len(res.fills) == 1
    assert res.fills[0].price == 100.50  # the :01 ask, NOT the :00 ask


class PeekStrategy:
    def __init__(self):
        self.seen_asks = []

    def on_event(self, event, ctx):
        q = ctx.nbbo("AAPL")
        self.seen_asks.append(None if q is None else q.ask)
        return []


def test_context_never_sees_future_quote():
    events = [_q(0, 99.98, 100.02), _q(1, 100.40, 100.50)]
    strat = PeekStrategy()
    BacktestEngine(EngineConfig()).run(strat, events, adv_dollar={"AAPL": 0.0}, impact_coef_bps=0.0)
    assert strat.seen_asks[0] == 100.02
    assert strat.seen_asks[1] == 100.50
