from datetime import UTC, datetime

from quant.intraday.sim.fills import Fill
from quant.intraday.sim.portfolio import Portfolio
from quant.intraday.strategy import Side


def _f(side, qty, price, commission=0.0):
    return Fill(
        ts=datetime(2023, 6, 1, tzinfo=UTC),
        symbol="AAPL",
        side=side,
        qty=qty,
        price=price,
        commission=commission,
        impact_cost=0.0,
        spread_cost=0.0,
    )


def test_buy_then_sell_round_trip_pnl():
    p = Portfolio(cash=100_000.0)
    p.apply_fill(_f(Side.BUY, 100, 100.0, commission=1.0))  # spend 10000 + 1 fee
    assert p.position("AAPL") == 100
    assert p.cash == 100_000.0 - 10_000.0 - 1.0
    p.apply_fill(_f(Side.SELL, 100, 101.0, commission=1.0))  # receive 10100 - 1 fee
    assert p.position("AAPL") == 0
    # realized = (101-100)*100 - 2 fees = 98
    assert round(p.realized_pnl, 2) == 98.0
    assert round(p.cash, 2) == round(100_000.0 + 98.0, 2)


def test_mark_to_market_equity():
    p = Portfolio(cash=100_000.0)
    p.apply_fill(_f(Side.BUY, 100, 100.0))
    eq = p.equity({"AAPL": 102.0})  # 90000 cash + 100*102 = 100200
    assert round(eq, 2) == 100_200.0
