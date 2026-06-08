"""Evaluate an execution schedule against the real intraday sim fill model. A
LiquidationStrategy emits one child per event until the schedule is exhausted;
evaluate_schedule runs it through BacktestEngine and reports realized cost."""

from __future__ import annotations

from typing import Any

from quant.intraday.data.events import Event, QuoteBar
from quant.intraday.sim.engine import BacktestEngine
from quant.intraday.strategy import Order, OrderType, Side, StrategyContext


class LiquidationStrategy:
    """Works a fixed parent along `child_sizes`, one slice per QuoteBar event."""

    def __init__(self, *, symbol: str, side: Side, child_sizes: list[int]) -> None:
        self._symbol = symbol
        self._side = side
        self._sizes = list(child_sizes)
        self._i = 0

    def on_event(self, event: Event, ctx: StrategyContext) -> list[Order]:
        if not isinstance(event, QuoteBar) or event.symbol != self._symbol:
            return []
        if self._i >= len(self._sizes):
            return []
        qty = self._sizes[self._i]
        self._i += 1
        if qty <= 0:
            return []
        return [Order(self._symbol, self._side, qty, OrderType.MARKET)]


def evaluate_schedule(
    *,
    events: list[Event],
    symbol: str,
    side: Side,
    child_sizes: list[int],
    adv_dollar: dict[str, float],
    impact_coef_bps: float,
) -> dict[str, Any]:
    """Run the schedule through the sim; return realized cost components + fills."""
    strat = LiquidationStrategy(symbol=symbol, side=side, child_sizes=child_sizes)
    result = BacktestEngine().run(
        strat, events, adv_dollar=adv_dollar, impact_coef_bps=impact_coef_bps
    )
    # signed_qty is a property on Fill, not a method
    filled = sum(abs(f.signed_qty) for f in result.fills)
    return {
        "total_cost": result.costs.total,
        "commission": result.costs.commission,
        "impact": result.costs.impact,
        "spread": result.costs.spread,
        "filled_shares": filled,
    }
