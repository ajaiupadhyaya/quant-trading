"""Event-driven backtest engine. Replays events in ts order; per event the loop
order is: (1) update the market view from the event, (2) fill any pending orders
now effective, (3) charge overnight financing if the session date advanced,
(4) let the strategy react (new orders are queued with a latency delay), (5)
record equity. No lookahead: an order placed at event E is queued with
effective_ts = E.ts + latency, so it can only fill at a later event."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import pandas as pd

from quant.backtest.financing import financing_charge
from quant.intraday.data.events import Bar, Event, QuoteBar, Trade
from quant.intraday.sim.fills import Fill, limit_fill, marketable_fill
from quant.intraday.sim.portfolio import Portfolio
from quant.intraday.sim.result import BacktestResult, CostBreakdown
from quant.intraday.strategy import IntradayStrategy, Order, OrderType


@dataclass(frozen=True)
class EngineConfig:
    starting_cash: float = 1_000_000.0
    latency: timedelta = timedelta(milliseconds=250)
    commission_per_share: float = 0.005
    annual_borrow_bps: float = 0.0
    annual_financing_bps: float = 0.0


@dataclass
class _PendingOrder:
    order: Order
    effective_ts: datetime


class _Context:
    """Concrete StrategyContext over the engine's already-seen market view."""

    def __init__(self, portfolio: Portfolio, nbbo_view: dict[str, QuoteBar]) -> None:
        self._pf = portfolio
        self._nbbo = nbbo_view
        self._now: datetime = datetime.min

    def position(self, symbol: str) -> int:
        return self._pf.position(symbol)

    def cash(self) -> float:
        return self._pf.cash

    def nbbo(self, symbol: str) -> QuoteBar | None:
        return self._nbbo.get(symbol)

    def now(self) -> datetime:
        return self._now


class BacktestEngine:
    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()

    def run(
        self,
        strategy: IntradayStrategy,
        events: Iterable[Event],
        *,
        adv_dollar: dict[str, float],
        impact_coef_bps: float,
    ) -> BacktestResult:
        cfg = self.config
        pf = Portfolio(cash=cfg.starting_cash)
        nbbo_view: dict[str, QuoteBar] = {}
        ctx = _Context(pf, nbbo_view)
        pending: list[_PendingOrder] = []
        fills: list[Fill] = []
        marks: dict[str, float] = {}
        equity_idx: list[datetime] = []
        equity_val: list[float] = []
        comm = imp = spr = fin = 0.0
        prev_date: date | None = None
        prev_marks: dict[str, float] = {}

        for event in events:
            now = event.ts
            ctx._now = now

            # (1) update market view from this event
            if isinstance(event, QuoteBar):
                nbbo_view[event.symbol] = event
                marks[event.symbol] = event.mid
            elif isinstance(event, Trade):
                marks[event.symbol] = event.price
            elif isinstance(event, Bar):
                marks[event.symbol] = event.close

            # (2) fill pending orders now effective
            still: list[_PendingOrder] = []
            for po in pending:
                if po.effective_ts > now:
                    still.append(po)
                    continue
                if po.order.type is OrderType.MARKET:
                    f = marketable_fill(
                        po.order,
                        nbbo_view.get(po.order.symbol),
                        now,
                        adv_dollar=adv_dollar.get(po.order.symbol, 0.0),
                        impact_coef_bps=impact_coef_bps,
                        commission_per_share=cfg.commission_per_share,
                    )
                    if f is None:
                        still.append(po)
                        continue
                else:
                    if not isinstance(event, Trade | QuoteBar):
                        still.append(po)
                        continue
                    f = limit_fill(
                        po.order, event, now, commission_per_share=cfg.commission_per_share
                    )
                    if f is None:
                        still.append(po)  # limit rests until it trades through
                        continue
                pf.apply_fill(f)
                fills.append(f)
                comm += f.commission
                imp += f.impact_cost
                spr += f.spread_cost
            pending[:] = still

            # (3) overnight financing when the session date advances
            ev_date = now.date()
            if prev_date is not None and ev_date > prev_date and pf.positions:
                charge = financing_charge(
                    positions=dict(pf.positions),
                    prior_close=prev_marks,
                    cash=pf.cash,
                    days_elapsed=(ev_date - prev_date).days,
                    annual_borrow_bps=cfg.annual_borrow_bps,
                    annual_financing_bps=cfg.annual_financing_bps,
                )
                pf.cash -= charge.total
                fin += charge.total

            # (4) strategy reacts; new orders queued with latency
            for order in strategy.on_event(event, ctx):
                pending.append(_PendingOrder(order=order, effective_ts=now + cfg.latency))

            # (5) record equity
            equity_idx.append(now)
            equity_val.append(pf.equity(marks))

            prev_date = ev_date
            prev_marks = dict(marks)

        curve = pd.Series(equity_val, index=pd.DatetimeIndex(equity_idx))
        return BacktestResult(
            equity_curve=curve,
            fills=fills,
            costs=CostBreakdown(commission=comm, impact=imp, spread=spr, financing=fin),
        )
