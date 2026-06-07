from datetime import UTC, datetime

from quant.intraday.data.events import QuoteBar
from quant.intraday.live.config import SleeveConfig
from quant.intraday.live.strategy import MeanReversionStrategy
from quant.intraday.strategy import Order, OrderType, Side


class _Ctx:
    def __init__(self, pos=0):
        self._pos = pos

    def position(self, symbol):
        return self._pos

    def cash(self):
        return 0.0

    def nbbo(self, symbol):
        return None

    def now(self):
        return datetime(2026, 6, 8, 15, 0, tzinfo=UTC)


def _qb(price, t):
    return QuoteBar(
        ts=datetime(2026, 6, 8, 15, t, tzinfo=UTC),
        symbol="QQQ",
        bid=price - 0.01,
        ask=price + 0.01,
        bid_size=100,
        ask_size=100,
    )


def test_no_order_before_window_full():
    cfg = SleeveConfig(mean_reversion_lookback=5)
    strat = MeanReversionStrategy(cfg, unit_shares=10)
    ctx = _Ctx()
    orders = []
    for i in range(4):
        orders = strat.on_event(_qb(100.0, i), ctx)
    assert orders == []  # window not yet full


def test_fades_upward_deviation_with_short():
    cfg = SleeveConfig(mean_reversion_lookback=5, entry_z=2.0)
    strat = MeanReversionStrategy(cfg, unit_shares=10)
    ctx = _Ctx(pos=0)
    for i in range(5):
        strat.on_event(_qb(100.0, i), ctx)  # flat history, ~zero vol -> spike z huge
    orders = strat.on_event(_qb(101.0, 5), ctx)  # jump up -> fade = SELL
    assert len(orders) == 1
    o = orders[0]
    assert isinstance(o, Order) and o.side is Side.SELL and o.symbol == "QQQ"
    assert o.type is OrderType.MARKET and o.qty == 10


def test_exits_when_reverted_inside_exit_band():
    cfg = SleeveConfig(mean_reversion_lookback=5, entry_z=2.0, exit_z=0.5)
    strat = MeanReversionStrategy(cfg, unit_shares=10)
    ctx = _Ctx(pos=-10)  # already short from a prior fade
    for i in range(5):
        strat.on_event(_qb(100.0, i), ctx)
    orders = strat.on_event(_qb(100.0, 5), ctx)  # back at mean -> exit short = BUY 10
    assert len(orders) == 1 and orders[0].side is Side.BUY and orders[0].qty == 10


def test_satisfies_intraday_strategy_protocol():
    from quant.intraday.strategy import IntradayStrategy

    assert isinstance(MeanReversionStrategy(SleeveConfig()), IntradayStrategy)
