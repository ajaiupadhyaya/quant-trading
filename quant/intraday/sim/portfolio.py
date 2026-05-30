"""Position / cash / P&L accounting (average-cost). Pure: no market data access
beyond marks passed in. Long and short supported."""

from __future__ import annotations

from dataclasses import dataclass, field

from quant.intraday.sim.fills import Fill
from quant.intraday.strategy import Side


@dataclass
class Portfolio:
    cash: float
    positions: dict[str, int] = field(default_factory=dict)
    avg_cost: dict[str, float] = field(default_factory=dict)
    realized_pnl: float = 0.0

    def position(self, symbol: str) -> int:
        return self.positions.get(symbol, 0)

    def apply_fill(self, fill: Fill) -> None:
        sym = fill.symbol
        pos = self.positions.get(sym, 0)
        avg = self.avg_cost.get(sym, 0.0)
        signed = fill.signed_qty

        # Cash: buys pay price*qty, sells receive price*qty; commission always paid.
        cash_delta = -fill.price * fill.qty if fill.side is Side.BUY else fill.price * fill.qty
        self.cash += cash_delta - fill.commission

        new_pos = pos + signed
        # Realize P&L on the portion that reduces/closes an existing position.
        if pos != 0 and (pos > 0) != (signed > 0):
            closed = min(abs(signed), abs(pos))
            direction = 1 if pos > 0 else -1
            self.realized_pnl += direction * (fill.price - avg) * closed
            self.realized_pnl -= fill.commission  # fee attributed to the closing trade
        else:
            # opening or adding: blend average cost, fee not yet realized
            total_qty = abs(pos) + abs(signed)
            if total_qty > 0:
                avg = (abs(pos) * avg + abs(signed) * fill.price) / total_qty
            self.realized_pnl -= fill.commission

        if new_pos == 0:
            self.avg_cost.pop(sym, None)
            self.positions.pop(sym, None)
        else:
            # if the position flipped direction (or opened from flat), reset avg cost
            self.avg_cost[sym] = fill.price if (pos == 0 or (pos > 0) != (new_pos > 0)) else avg
            self.positions[sym] = new_pos

    def market_value(self, marks: dict[str, float]) -> float:
        return sum(qty * marks.get(sym, 0.0) for sym, qty in self.positions.items())

    def equity(self, marks: dict[str, float]) -> float:
        return self.cash + self.market_value(marks)
