"""ExecutionManager: holds at most one active ExecutionProgram per symbol, builds
A-C programs from parent entries, and yields due child Orders per tick. It never
submits — the loop submits what due_slices() returns and reports fills back."""

from __future__ import annotations

from quant.intraday.execution.almgren_chriss import optimal_schedule
from quant.intraday.execution.config import ExecConfig
from quant.intraday.execution.scheduler import ExecutionProgram
from quant.intraday.strategy import Order, OrderType


class ExecutionManager:
    def __init__(self, config: ExecConfig) -> None:
        self._cfg = config
        self._programs: dict[str, ExecutionProgram] = {}

    def has_active(self, symbol: str) -> bool:
        prog = self._programs.get(symbol)
        return prog is not None and not prog.is_complete

    def start_entry(
        self,
        parent: Order,
        *,
        tick_index: int,
        sigma: float,
        eta: float,
        gamma: float,
    ) -> bool:
        if self.has_active(parent.symbol):
            return False
        plan = optimal_schedule(
            total_shares=parent.qty,
            n_intervals=self._cfg.horizon_ticks,
            tau=1.0,
            sigma=sigma,
            eta=eta,
            gamma=gamma,
            risk_aversion=self._cfg.risk_aversion,
        )
        self._programs[parent.symbol] = ExecutionProgram(
            symbol=parent.symbol,
            side=parent.side,
            total_qty=parent.qty,
            child_sizes=plan.child_sizes,
            start_tick=tick_index,
        )
        return True

    def due_slices(self, tick_index: int) -> list[Order]:
        orders: list[Order] = []
        for prog in self._programs.values():
            qty = prog.slice_due(tick_index)
            if qty > 0:
                orders.append(Order(prog.symbol, prog.side, qty, OrderType.MARKET))
        return orders

    def record_fill(self, symbol: str, qty: int) -> None:
        prog = self._programs.get(symbol)
        if prog is not None:
            prog.record_fill(qty)
            if prog.is_complete:
                del self._programs[symbol]

    def cancel(self, symbol: str) -> None:
        prog = self._programs.get(symbol)
        if prog is not None:
            prog.cancel()
            del self._programs[symbol]
