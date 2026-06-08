from datetime import UTC, datetime, timedelta

from quant.intraday.data.events import QuoteBar
from quant.intraday.execution.evaluate import LiquidationStrategy, evaluate_schedule
from quant.intraday.strategy import Side


def _events(n, price=100.0):
    t0 = datetime(2026, 6, 8, 14, 30, tzinfo=UTC)
    out = []
    for i in range(n):
        out.append(QuoteBar(ts=t0 + timedelta(minutes=i), symbol="QQQ",
                            bid=price - 0.01, ask=price + 0.01, bid_size=500, ask_size=500))
    return out


def test_liquidation_strategy_emits_schedule_in_order():
    strat = LiquidationStrategy(symbol="QQQ", side=Side.BUY, child_sizes=[10, 20, 30])
    emitted = []

    class _Ctx:
        def position(self, s): return 0
        def cash(self): return 0.0
        def nbbo(self, s): return None
        def now(self): return datetime(2026, 6, 8, 14, 30, tzinfo=UTC)

    for ev in _events(5):
        for o in strat.on_event(ev, _Ctx()):
            emitted.append(o.qty)
    assert emitted == [10, 20, 30]


def test_evaluate_schedule_returns_realized_cost():
    res = evaluate_schedule(
        events=_events(6), symbol="QQQ", side=Side.BUY, child_sizes=[20, 20, 20],
        adv_dollar={"QQQ": 5_000_000_000.0}, impact_coef_bps=10.0,
    )
    assert "total_cost" in res and res["total_cost"] >= 0.0
    assert "commission" in res and "impact" in res and "spread" in res
    assert res["filled_shares"] == 60
