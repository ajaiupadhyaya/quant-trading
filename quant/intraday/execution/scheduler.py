"""ExecutionProgram: one parent order worked along a fixed child-size schedule, one
slice per tick. Schedule-source-agnostic (Almgren-Chriss or any baseline)."""

from __future__ import annotations

from dataclasses import dataclass, field

from quant.intraday.strategy import Side


@dataclass
class ExecutionProgram:
    symbol: str
    side: Side
    total_qty: int
    child_sizes: list[int]
    start_tick: int
    filled: int = field(default=0)
    cancelled: bool = field(default=False)

    def __post_init__(self) -> None:
        if sum(self.child_sizes) != self.total_qty:
            raise ValueError(
                f"child_sizes sum {sum(self.child_sizes)} != total_qty {self.total_qty}"
            )

    def slice_due(self, tick_index: int) -> int:
        if self.cancelled:
            return 0
        offset = tick_index - self.start_tick
        if 0 <= offset < len(self.child_sizes):
            return self.child_sizes[offset]
        return 0

    def record_fill(self, qty: int) -> None:
        self.filled += qty

    @property
    def remaining(self) -> int:
        return self.total_qty - self.filled

    @property
    def is_complete(self) -> bool:
        return self.cancelled or self.filled >= self.total_qty

    def cancel(self) -> None:
        self.cancelled = True
