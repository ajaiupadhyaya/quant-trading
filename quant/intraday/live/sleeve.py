"""Internal sleeve ledger: positions and realized/unrealized P&L computed from the
sleeve's OWN fills, independent of the Alpaca aggregate. Average-cost, long/short."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Fill:
    symbol: str
    qty: int          # signed: +buy, -sell
    price: float


@dataclass
class _Lot:
    qty: int = 0      # signed net position
    avg_price: float = 0.0


@dataclass
class SleeveLedger:
    realized_pnl: float = 0.0
    round_trips: int = 0
    _lots: dict[str, _Lot] = field(default_factory=dict)

    def position(self, symbol: str) -> int:
        return self._lots.get(symbol, _Lot()).qty

    def positions(self) -> dict[str, int]:
        return {s: lot.qty for s, lot in self._lots.items() if lot.qty != 0}

    def record(self, fill: Fill) -> None:
        lot = self._lots.setdefault(fill.symbol, _Lot())
        old_qty = lot.qty
        if old_qty == 0:
            self.round_trips += 1  # opening a new position
        # Same direction (or opening): blend average price.
        if old_qty == 0 or (old_qty > 0) == (fill.qty > 0):
            new_qty = old_qty + fill.qty
            lot.avg_price = (
                (abs(old_qty) * lot.avg_price + abs(fill.qty) * fill.price)
                / (abs(old_qty) + abs(fill.qty))
            )
            lot.qty = new_qty
            return
        # Opposite direction: realize against existing average for the closed amount.
        closed = min(abs(fill.qty), abs(old_qty))
        direction = 1.0 if old_qty > 0 else -1.0
        self.realized_pnl += direction * (fill.price - lot.avg_price) * closed
        lot.qty = old_qty + fill.qty
        if lot.qty == 0:
            lot.avg_price = 0.0
        elif (lot.qty > 0) != (old_qty > 0):
            # Flipped through zero: leftover opens a new position at fill price.
            lot.avg_price = fill.price
            self.round_trips += 1

    def unrealized_pnl(self, marks: dict[str, float]) -> float:
        total = 0.0
        for sym, lot in self._lots.items():
            if lot.qty == 0:
                continue
            total += (marks[sym] - lot.avg_price) * lot.qty
        return total

    def gross_notional(self, marks: dict[str, float]) -> float:
        return sum(abs(lot.qty) * marks[sym] for sym, lot in self._lots.items() if lot.qty)

    def day_pnl(self, marks: dict[str, float]) -> float:
        return self.realized_pnl + self.unrealized_pnl(marks)
