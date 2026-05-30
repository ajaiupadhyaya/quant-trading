from datetime import UTC, datetime

from quant.intraday.data.events import QuoteBar
from quant.intraday.sim.fills import marketable_fill
from quant.intraday.strategy import Order, Side


def _ts():
    return datetime(2023, 6, 1, 13, 30, tzinfo=UTC)


def _nbbo():
    return QuoteBar(ts=_ts(), symbol="AAPL", bid=99.98, ask=100.02, bid_size=10, ask_size=10)


def test_market_buy_fills_at_ask_plus_costs():
    o = Order("AAPL", Side.BUY, 100)
    f = marketable_fill(
        o, _nbbo(), _ts(), adv_dollar=0.0, impact_coef_bps=0.0, commission_per_share=0.005
    )
    assert f is not None
    assert f.price == 100.02  # far touch (ask), zero impact
    assert f.commission == 0.5  # 100 * 0.005
    assert round(f.spread_cost, 4) == 2.0  # (ask - mid) * qty = 0.02 * 100
    assert f.impact_cost == 0.0


def test_market_sell_fills_at_bid():
    o = Order("AAPL", Side.SELL, 50)
    f = marketable_fill(
        o, _nbbo(), _ts(), adv_dollar=0.0, impact_coef_bps=0.0, commission_per_share=0.0
    )
    assert f is not None
    assert f.price == 99.98


def test_impact_raises_buy_price_with_size():
    o = Order("AAPL", Side.BUY, 100)
    f = marketable_fill(
        o, _nbbo(), _ts(), adv_dollar=100_000.0, impact_coef_bps=10.0, commission_per_share=0.0
    )
    assert f is not None
    assert f.price > 100.02  # impact pushes the buy fill above the ask
    assert f.impact_cost > 0.0


def test_no_nbbo_returns_none():
    o = Order("AAPL", Side.BUY, 100)
    assert (
        marketable_fill(
            o, None, _ts(), adv_dollar=0.0, impact_coef_bps=0.0, commission_per_share=0.0
        )
        is None
    )
