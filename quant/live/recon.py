"""Pure reconciliation logic: join local trade intents against Alpaca fill outcomes.

Inputs come in as already-loaded DataFrames + iterables — no Alpaca calls, no
file I/O, no bar fetching. The orchestrator in scripts/reconcile_live.py wires
those in. Keep this module side-effect-free so tests can stay synchronous and
network-free.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from quant.execution.alpaca import OrderRow

BarFetcher = Callable[[str, date], float | None]
MidFetcher = Callable[[OrderRow], float | None]
"""(symbol, signal_date) -> close price, or None if unavailable.

signal_date is the rebalance-target date — the bar the strategy actually used
to size positions. See generate_signals() in each strategy: they call
asof_index(history, asof) which returns the bar at asof itself when present,
falling back to T-1 only when the asof bar is unavailable in the cache.
"""


@dataclass(frozen=True)
class ReconRow:
    """One reconciled trade: intent joined with outcome plus derived metrics."""

    client_order_id: str
    strategy: str
    symbol: str
    side: str  # "buy" | "sell"
    submission_date: date
    submitted_qty: int
    filled_qty: int
    signal_price: float | None
    fill_price: float | None
    slippage_bps: float | None  # signed; positive = worse than signal
    fill_lag_seconds: float | None
    status: str  # filled | partial | rejected | missing | no_signal_price | no_fill_price | no_mid_price
    mid_price: float | None = None
    execution_cost_bps: float | None = None


@dataclass
class ReconciliationReport:
    since: date
    until: date
    modeled_slippage_bps: float
    rows: list[ReconRow] = field(default_factory=list)

    def aggregate_by_strategy(self) -> dict[str, dict[str, float | int | None]]:
        return self._aggregate(key=lambda r: r.strategy)

    def aggregate_by_symbol(self) -> dict[str, dict[str, float | int | None]]:
        return self._aggregate(key=lambda r: r.symbol)

    def _aggregate(
        self, key: Callable[[ReconRow], str]
    ) -> dict[str, dict[str, float | int | None]]:
        from statistics import mean

        out: dict[str, dict[str, float | int | None]] = {}
        groups: dict[str, list[ReconRow]] = {}
        for row in self.rows:
            groups.setdefault(key(row), []).append(row)

        for grp_key, rows in groups.items():
            filled = [r for r in rows if r.slippage_bps is not None]
            lags = [r.fill_lag_seconds for r in rows if r.fill_lag_seconds is not None]
            slippages: list[float] = [r.slippage_bps for r in filled if r.slippage_bps is not None]
            out[grp_key] = {
                "n_total": len(rows),
                "n_filled": len(filled),
                "n_partial": sum(1 for r in rows if r.status == "partial"),
                "n_rejected": sum(1 for r in rows if r.status == "rejected"),
                "n_missing": sum(1 for r in rows if r.status == "missing"),
                "mean_slippage_bps": mean(slippages) if slippages else None,
                "median_fill_lag_s": (
                    sorted(lags)[len(lags) // 2] if lags else None
                ),
            }
        return out


def _slippage_bps(side: str, signal_price: float, fill_price: float) -> float:
    if side == "buy":
        return (fill_price - signal_price) / signal_price * 1e4
    return (signal_price - fill_price) / signal_price * 1e4


def _execution_cost_bps(side: str, mid_price: float, fill_price: float) -> float:
    if side == "buy":
        return (fill_price - mid_price) / mid_price * 1e4
    return (mid_price - fill_price) / mid_price * 1e4


def reconcile(
    *,
    trades: pd.DataFrame,
    orders: Iterable[OrderRow],
    bar_fetcher: BarFetcher,
    mid_fetcher: MidFetcher | None = None,
    modeled_slippage_bps: float,
    since: date,
    until: date,
) -> ReconciliationReport:
    """Join trade intents to Alpaca order outcomes and compute per-row metrics."""
    orders_by_coid: dict[str, OrderRow] = {o.client_order_id: o for o in orders}
    rows: list[ReconRow] = []

    for _, t in trades.iterrows():
        coid = str(t["client_order_id"])
        order = orders_by_coid.get(coid)
        submission_date = (
            t["date"].date() if hasattr(t["date"], "date") else t["date"]
        )

        if order is None:
            rows.append(
                ReconRow(
                    client_order_id=coid,
                    strategy=str(t["strategy"]),
                    symbol=str(t["symbol"]),
                    side=str(t["side"]),
                    submission_date=submission_date,
                    submitted_qty=int(t["qty"]),
                    filled_qty=0,
                    signal_price=None,
                    fill_price=None,
                    mid_price=None,
                    slippage_bps=None,
                    execution_cost_bps=None,
                    fill_lag_seconds=None,
                    status="missing",
                )
            )
            continue

        signal_price = bar_fetcher(str(t["symbol"]), submission_date)
        mid_price = mid_fetcher(order) if mid_fetcher is not None else None
        fill_lag = None
        if order.filled_at is not None:
            fill_lag = (order.filled_at - order.submitted_at).total_seconds()

        execution_cost = None
        if order.status in {"canceled", "rejected", "expired"} or order.filled_qty == 0:
            status = "rejected"
            slippage = None
        elif signal_price is None or order.filled_avg_price is None:
            status = "no_signal_price" if signal_price is None else "no_fill_price"
            slippage = None
        else:
            slippage = _slippage_bps(order.side, signal_price, order.filled_avg_price)
            status = "filled" if order.filled_qty >= order.submitted_qty else "partial"
            if mid_fetcher is not None:
                if mid_price is None:
                    status = "no_mid_price"
                else:
                    execution_cost = _execution_cost_bps(
                        order.side, mid_price, order.filled_avg_price
                    )

        rows.append(
            ReconRow(
                client_order_id=coid,
                strategy=str(t["strategy"]),
                symbol=str(t["symbol"]),
                side=str(t["side"]),
                submission_date=submission_date,
                submitted_qty=int(t["qty"]),
                filled_qty=int(order.filled_qty),
                signal_price=signal_price,
                fill_price=order.filled_avg_price,
                mid_price=mid_price,
                slippage_bps=slippage,
                execution_cost_bps=execution_cost,
                fill_lag_seconds=fill_lag,
                status=status,
            )
        )

    return ReconciliationReport(
        since=since,
        until=until,
        modeled_slippage_bps=modeled_slippage_bps,
        rows=rows,
    )
