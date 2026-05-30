# Intraday Execution Simulator + Backtester Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build sub-project B — an event-driven backtester + realistic execution simulator that replays the data layer's events through an `IntradayStrategy`, models hybrid fills (marketable + limit-vs-1s-NBBO) with latency and the existing cost models, and emits returns the charter validation battery can judge.

**Architecture:** New package code under `quant/intraday/` (sibling to the built `quant/intraday/data/`). Pure fill + portfolio logic; an event-driven engine with a latency-delayed pending-order queue; a result object that resamples the intraday equity curve to daily returns and reuses `quant/backtest/metrics.py`. Defines the `IntradayStrategy.on_event` protocol shared with the future live engine (D).

**Tech Stack:** Python 3.12, pandas, pytest, `uv`. Reuses `quant/backtest/impact.py` (`market_impact_bps`), `quant/backtest/financing.py` (`financing_charge`), `quant/backtest/metrics.py` (`sharpe`/`max_drawdown`/`total_return`). Consumes `quant/intraday/data/events.py` (`Trade`/`QuoteBar`/`Bar`) and `MarketDataStore.replay()`.

**No subscription needed:** every test is fixture/synthetic. The engine accepts any iterable of events, so tests feed hand-built event lists (no `MarketDataStore`/network required).

---

## File Structure

```
quant/intraday/strategy.py        Side, OrderType, Order, StrategyContext (Protocol), IntradayStrategy (Protocol)
quant/intraday/sim/__init__.py    exports
quant/intraday/sim/fills.py       Fill, marketable_fill(), limit_crosses(), limit_fill()        [pure]
quant/intraday/sim/portfolio.py   Portfolio: apply_fill / market_value / equity (avg-cost P&L)   [pure]
quant/intraday/sim/result.py      CostBreakdown, BacktestResult (daily-resampled returns + metrics)
quant/intraday/sim/engine.py      BacktestEngine.run(strategy, events, ...) -> BacktestResult
tests/intraday/sim/               mirror, fixture-driven
```

Engine consumes an `Iterable[Event]` (so tests pass plain lists; production passes
`store.replay(...)`). Each file has one responsibility; `strategy.py` is imported by the engine and
by the future live engine (D).

---

## Task 1: Strategy protocol — orders & context

**Files:**
- Create: `quant/intraday/strategy.py`
- Test: `tests/intraday/__init__.py` (exists), `tests/intraday/sim/__init__.py`, `tests/intraday/sim/test_strategy.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/intraday/sim/test_strategy.py
import pytest

from quant.intraday.strategy import Order, OrderType, Side


def test_market_order_defaults():
    o = Order(symbol="AAPL", side=Side.BUY, qty=100)
    assert o.type is OrderType.MARKET and o.limit_price is None


def test_limit_order_requires_price():
    with pytest.raises(ValueError):
        Order(symbol="AAPL", side=Side.SELL, qty=10, type=OrderType.LIMIT)


def test_qty_must_be_positive():
    with pytest.raises(ValueError):
        Order(symbol="AAPL", side=Side.BUY, qty=0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/intraday/sim/test_strategy.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# quant/intraday/strategy.py
"""Event-driven intraday strategy interface. SEPARATE from the daily pull-based
quant.strategies.base.Strategy — intraday strategies react to an event stream.
The same on_event() drives the backtester (B) and the live engine (D), which is
the execution-side guarantee against train/serve skew."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Protocol, runtime_checkable

from quant.intraday.data.events import Event, QuoteBar


class Side(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"


@dataclass(frozen=True)
class Order:
    symbol: str
    side: Side
    qty: int
    type: OrderType = OrderType.MARKET
    limit_price: float | None = None

    def __post_init__(self) -> None:
        if self.qty <= 0:
            raise ValueError(f"Order qty must be positive, got {self.qty}")
        if self.type is OrderType.LIMIT and self.limit_price is None:
            raise ValueError("LIMIT order requires limit_price")
        if self.type is OrderType.MARKET and self.limit_price is not None:
            raise ValueError("MARKET order must not set limit_price")


@runtime_checkable
class StrategyContext(Protocol):
    """Read-only market+account view exposing ONLY data with ts <= the current event."""

    def position(self, symbol: str) -> int: ...
    def cash(self) -> float: ...
    def nbbo(self, symbol: str) -> QuoteBar | None: ...
    def now(self) -> datetime: ...


@runtime_checkable
class IntradayStrategy(Protocol):
    def on_event(self, event: Event, ctx: StrategyContext) -> list[Order]: ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/intraday/sim/test_strategy.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add quant/intraday/strategy.py tests/intraday/sim/
git commit -m "feat(intraday): event-driven IntradayStrategy protocol + Order"
```

---

## Task 2: Marketable fills (far touch + impact + commission)

**Files:**
- Create: `quant/intraday/sim/__init__.py`, `quant/intraday/sim/fills.py`
- Test: `tests/intraday/sim/test_fills_market.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/intraday/sim/test_fills_market.py
from datetime import datetime, timezone

from quant.intraday.data.events import QuoteBar
from quant.intraday.sim.fills import marketable_fill
from quant.intraday.strategy import Order, OrderType, Side


def _ts():
    return datetime(2023, 6, 1, 13, 30, tzinfo=timezone.utc)


def _nbbo():
    return QuoteBar(ts=_ts(), symbol="AAPL", bid=99.98, ask=100.02, bid_size=10, ask_size=10)


def test_market_buy_fills_at_ask_plus_costs():
    o = Order("AAPL", Side.BUY, 100)
    f = marketable_fill(o, _nbbo(), _ts(), adv_dollar=0.0, impact_coef_bps=0.0, commission_per_share=0.005)
    assert f is not None
    assert f.price == 100.02            # far touch (ask), zero impact
    assert f.commission == 0.5          # 100 * 0.005
    assert round(f.spread_cost, 4) == 2.0   # (ask - mid) * qty = 0.02 * 100
    assert f.impact_cost == 0.0


def test_market_sell_fills_at_bid():
    o = Order("AAPL", Side.SELL, 50)
    f = marketable_fill(o, _nbbo(), _ts(), adv_dollar=0.0, impact_coef_bps=0.0, commission_per_share=0.0)
    assert f.price == 99.98


def test_impact_raises_buy_price_with_size():
    o = Order("AAPL", Side.BUY, 100)
    # notional ~ 10002; adv 100000 -> participation ~0.10 -> sqrt ~0.316 -> 10bps*0.316 ~3.16bps
    f = marketable_fill(o, _nbbo(), _ts(), adv_dollar=100_000.0, impact_coef_bps=10.0, commission_per_share=0.0)
    assert f.price > 100.02             # impact pushes the buy fill above the ask
    assert f.impact_cost > 0.0


def test_no_nbbo_returns_none():
    o = Order("AAPL", Side.BUY, 100)
    assert marketable_fill(o, None, _ts(), adv_dollar=0.0, impact_coef_bps=0.0, commission_per_share=0.0) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/intraday/sim/test_fills_market.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# quant/intraday/sim/__init__.py
"""Intraday backtest + execution simulation."""
```

```python
# quant/intraday/sim/fills.py
"""Fill model: marketable (spread-crossing) + limit (fills only when the 1s NBBO
trades through). Pure functions; all costs charged here. The engine supplies
adv_dollar per symbol (so this stays decoupled from the daily bars frame)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from quant.backtest.impact import market_impact_bps
from quant.intraday.data.events import QuoteBar, Trade
from quant.intraday.strategy import Order, OrderType, Side


@dataclass(frozen=True)
class Fill:
    ts: datetime
    symbol: str
    side: Side
    qty: int
    price: float
    commission: float
    impact_cost: float
    spread_cost: float

    @property
    def signed_qty(self) -> int:
        return self.qty if self.side is Side.BUY else -self.qty


def marketable_fill(
    order: Order,
    nbbo: QuoteBar | None,
    ts: datetime,
    *,
    adv_dollar: float,
    impact_coef_bps: float,
    commission_per_share: float,
) -> Fill | None:
    """Cross the spread at the far touch; add square-root impact + commission."""
    if nbbo is None:
        return None
    ref = nbbo.ask if order.side is Side.BUY else nbbo.bid
    notional = ref * order.qty
    imp_bps = market_impact_bps(notional, adv_dollar, impact_coef_bps)
    slip = ref * (imp_bps / 1e4)
    price = ref + slip if order.side is Side.BUY else ref - slip
    spread_per_share = abs(ref - nbbo.mid)
    return Fill(
        ts=ts,
        symbol=order.symbol,
        side=order.side,
        qty=order.qty,
        price=price,
        commission=commission_per_share * order.qty,
        impact_cost=slip * order.qty,
        spread_cost=spread_per_share * order.qty,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/intraday/sim/test_fills_market.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add quant/intraday/sim/__init__.py quant/intraday/sim/fills.py tests/intraday/sim/test_fills_market.py
git commit -m "feat(intraday): marketable fill model (far touch + impact + commission)"
```

---

## Task 3: Limit fills (only when the NBBO trades through)

**Files:**
- Modify: `quant/intraday/sim/fills.py`
- Test: `tests/intraday/sim/test_fills_limit.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/intraday/sim/test_fills_limit.py
from datetime import datetime, timezone

from quant.intraday.data.events import QuoteBar, Trade
from quant.intraday.sim.fills import limit_crosses, limit_fill
from quant.intraday.strategy import Order, OrderType, Side


def _ts():
    return datetime(2023, 6, 1, 13, 30, tzinfo=timezone.utc)


def test_buy_limit_crosses_on_trade_through():
    o = Order("AAPL", Side.BUY, 10, type=OrderType.LIMIT, limit_price=100.0)
    assert limit_crosses(o, Trade(_ts(), "AAPL", price=99.99, size=5)) is True   # printed <= limit
    assert limit_crosses(o, Trade(_ts(), "AAPL", price=100.01, size=5)) is False  # above limit


def test_sell_limit_crosses_on_trade_through():
    o = Order("AAPL", Side.SELL, 10, type=OrderType.LIMIT, limit_price=100.0)
    assert limit_crosses(o, Trade(_ts(), "AAPL", price=100.01, size=5)) is True
    assert limit_crosses(o, Trade(_ts(), "AAPL", price=99.99, size=5)) is False


def test_buy_limit_crosses_on_quote_through():
    o = Order("AAPL", Side.BUY, 10, type=OrderType.LIMIT, limit_price=100.0)
    thru = QuoteBar(_ts(), "AAPL", bid=99.0, ask=99.95, bid_size=1, ask_size=1)  # ask <= limit
    assert limit_crosses(o, thru) is True


def test_limit_fill_at_limit_price_no_impact():
    o = Order("AAPL", Side.BUY, 10, type=OrderType.LIMIT, limit_price=100.0)
    f = limit_fill(o, Trade(_ts(), "AAPL", price=99.99, size=5), _ts(), commission_per_share=0.005)
    assert f is not None and f.price == 100.0 and f.impact_cost == 0.0 and f.spread_cost == 0.0
    assert f.commission == 0.05


def test_limit_no_cross_returns_none():
    o = Order("AAPL", Side.BUY, 10, type=OrderType.LIMIT, limit_price=100.0)
    assert limit_fill(o, Trade(_ts(), "AAPL", price=100.5, size=5), _ts(), commission_per_share=0.0) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/intraday/sim/test_fills_limit.py -v`
Expected: FAIL — `limit_crosses`/`limit_fill` not defined.

- [ ] **Step 3: Implement (append to `fills.py`)**

```python
def limit_crosses(order: Order, event: Trade | QuoteBar) -> bool:
    """True if this event shows the market trading THROUGH the limit (conservative;
    no passive-queue credit). Buy fills when price/ask <= limit; sell when
    price/bid >= limit."""
    if order.limit_price is None:
        return False
    if isinstance(event, Trade):
        ref = event.price
        return ref <= order.limit_price if order.side is Side.BUY else ref >= order.limit_price
    if isinstance(event, QuoteBar):
        if order.side is Side.BUY:
            return event.ask <= order.limit_price
        return event.bid >= order.limit_price
    return False


def limit_fill(
    order: Order,
    event: Trade | QuoteBar,
    ts: datetime,
    *,
    commission_per_share: float,
) -> Fill | None:
    """Fill a resting limit at the limit price iff the event trades through it."""
    if not limit_crosses(order, event):
        return None
    assert order.limit_price is not None
    return Fill(
        ts=ts,
        symbol=order.symbol,
        side=order.side,
        qty=order.qty,
        price=order.limit_price,
        commission=commission_per_share * order.qty,
        impact_cost=0.0,
        spread_cost=0.0,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/intraday/sim/test_fills_limit.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add quant/intraday/sim/fills.py tests/intraday/sim/test_fills_limit.py
git commit -m "feat(intraday): conservative limit fills (trade-through, no queue credit)"
```

---

## Task 4: Portfolio accounting (avg-cost P&L)

**Files:**
- Create: `quant/intraday/sim/portfolio.py`
- Test: `tests/intraday/sim/test_portfolio.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/intraday/sim/test_portfolio.py
from datetime import datetime, timezone

from quant.intraday.sim.fills import Fill
from quant.intraday.sim.portfolio import Portfolio
from quant.intraday.strategy import Side


def _f(side, qty, price, commission=0.0):
    return Fill(
        ts=datetime(2023, 6, 1, tzinfo=timezone.utc), symbol="AAPL", side=side, qty=qty,
        price=price, commission=commission, impact_cost=0.0, spread_cost=0.0,
    )


def test_buy_then_sell_round_trip_pnl():
    p = Portfolio(cash=100_000.0)
    p.apply_fill(_f(Side.BUY, 100, 100.0, commission=1.0))   # spend 10000 + 1 fee
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/intraday/sim/test_portfolio.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# quant/intraday/sim/portfolio.py
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
        else:
            # if direction flipped, the new average cost is this fill's price
            self.avg_cost[sym] = fill.price if (pos == 0 or (pos > 0) != (new_pos > 0)) else avg
        self.positions[sym] = new_pos
        if new_pos == 0:
            self.positions.pop(sym, None)

    def market_value(self, marks: dict[str, float]) -> float:
        return sum(qty * marks.get(sym, 0.0) for sym, qty in self.positions.items())

    def equity(self, marks: dict[str, float]) -> float:
        return self.cash + self.market_value(marks)
```

Note: the `realized_pnl` here folds commissions into realized P&L so that `cash` and
`realized_pnl + initial_cash + unrealized` reconcile; the round-trip test pins the exact numbers.
If the test fails on a sign, fix the accounting to satisfy the test (the test is the spec).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/intraday/sim/test_portfolio.py -v`
Expected: PASS (2 passed). If `realized_pnl` is off, reconcile against the test's expected 98.0.

- [ ] **Step 5: Commit**

```bash
git add quant/intraday/sim/portfolio.py tests/intraday/sim/test_portfolio.py
git commit -m "feat(intraday): portfolio accounting (avg-cost P&L, long/short)"
```

---

## Task 5: BacktestResult — daily-resampled returns + metrics

**Files:**
- Create: `quant/intraday/sim/result.py`
- Test: `tests/intraday/sim/test_result.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/intraday/sim/test_result.py
import pandas as pd

from quant.intraday.sim.result import BacktestResult, CostBreakdown


def _equity():
    idx = pd.to_datetime(
        ["2023-06-01T20:00:00Z", "2023-06-02T20:00:00Z", "2023-06-05T20:00:00Z"]
    )
    return pd.Series([100_000.0, 101_000.0, 100_500.0], index=idx)


def test_daily_returns_from_equity():
    r = BacktestResult(equity_curve=_equity(), fills=[], costs=CostBreakdown(0, 0, 0, 0)).daily_returns()
    assert len(r) == 2
    assert round(r.iloc[0], 5) == round(1_000 / 100_000, 5)


def test_sharpe_runs_on_daily_returns():
    res = BacktestResult(equity_curve=_equity(), fills=[], costs=CostBreakdown(1, 2, 3, 4))
    s = res.sharpe()
    assert isinstance(s, float)
    assert res.costs.total == 10.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/intraday/sim/test_result.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# quant/intraday/sim/result.py
"""Backtest output: intraday equity curve + daily-resampled returns so the
existing charter metrics (and, in sub-project C, DSR/PSR/bootstrap) operate on
daily returns exactly as the daily system does."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from quant.backtest.metrics import max_drawdown, sharpe, total_return
from quant.intraday.sim.fills import Fill


@dataclass(frozen=True)
class CostBreakdown:
    commission: float
    impact: float
    spread: float
    financing: float

    @property
    def total(self) -> float:
        return self.commission + self.impact + self.spread + self.financing


@dataclass(frozen=True)
class BacktestResult:
    equity_curve: pd.Series  # intraday marks (DatetimeIndex)
    fills: list[Fill]
    costs: CostBreakdown
    metadata: dict = field(default_factory=dict)

    def daily_returns(self) -> pd.Series:
        if self.equity_curve.empty:
            return pd.Series(dtype=float)
        daily = self.equity_curve.resample("1D").last().dropna()
        return daily.pct_change().dropna()

    def sharpe(self, periods_per_year: int = 252) -> float:
        return sharpe(self.daily_returns(), periods_per_year=periods_per_year)

    def max_drawdown(self) -> float:
        return max_drawdown(self.daily_returns())

    def total_return(self) -> float:
        return total_return(self.daily_returns())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/intraday/sim/test_result.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add quant/intraday/sim/result.py tests/intraday/sim/test_result.py
git commit -m "feat(intraday): BacktestResult (daily-resampled returns + metrics)"
```

---

## Task 6: Engine — market view, context, latency, marketable run loop

**Files:**
- Create: `quant/intraday/sim/engine.py`
- Test: `tests/intraday/sim/test_engine_latency.py`

- [ ] **Step 1: Write the failing test** (latency + no-lookahead are the invariants)

```python
# tests/intraday/sim/test_engine_latency.py
from datetime import datetime, timedelta, timezone

from quant.intraday.data.events import QuoteBar
from quant.intraday.sim.engine import BacktestEngine, EngineConfig
from quant.intraday.strategy import Order, Side


def _q(sec, bid, ask):
    return QuoteBar(
        ts=datetime(2023, 6, 1, 13, 30, sec, tzinfo=timezone.utc),
        symbol="AAPL", bid=bid, ask=ask, bid_size=10, ask_size=10,
    )


class BuyOnceStrategy:
    def __init__(self):
        self.fired = False

    def on_event(self, event, ctx):
        if not self.fired:
            self.fired = True
            return [Order("AAPL", Side.BUY, 100)]
        return []


def test_order_fills_at_post_latency_nbbo_not_signal_time():
    # quotes at :00 (ask 100.02) and :01 (ask 100.50); latency 250ms means the
    # order signalled at :00 fills against the :01 NBBO (the next event >= effective time).
    events = [_q(0, 99.98, 100.02), _q(1, 100.40, 100.50)]
    eng = BacktestEngine(EngineConfig(latency=timedelta(milliseconds=250), commission_per_share=0.0))
    res = eng.run(BuyOnceStrategy(), events, adv_dollar={"AAPL": 0.0}, impact_coef_bps=0.0)
    assert len(res.fills) == 1
    assert res.fills[0].price == 100.50   # filled at the :01 ask, NOT the :00 ask
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/intraday/sim/test_engine_latency.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# quant/intraday/sim/engine.py
"""Event-driven backtest engine. Replays events in ts order, calls the strategy,
queues orders with a latency delay, fills them against the prevailing NBBO at/after
their effective time, and accrues marks into an equity curve. No lookahead: the
strategy context exposes only events already seen."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pandas as pd

from quant.intraday.data.events import Bar, Event, QuoteBar, Trade
from quant.intraday.sim.fills import Fill, limit_fill, marketable_fill
from quant.intraday.sim.portfolio import Portfolio
from quant.intraday.sim.result import BacktestResult, CostBreakdown
from quant.intraday.strategy import IntradayStrategy, Order, OrderType, Side


@dataclass(frozen=True)
class EngineConfig:
    starting_cash: float = 1_000_000.0
    latency: timedelta = timedelta(milliseconds=250)
    commission_per_share: float = 0.005


@dataclass
class _PendingOrder:
    order: Order
    effective_ts: datetime


class _Context:
    """Concrete StrategyContext over the engine's already-seen market view."""

    def __init__(self, portfolio: Portfolio, nbbo_view: dict[str, QuoteBar]) -> None:
        self._pf = portfolio
        self._nbbo = nbbo_view
        self._now = datetime.min

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
        equity_points: list[tuple[datetime, float]] = []
        marks: dict[str, float] = {}
        comm = imp = spr = 0.0

        def _try_fill_pending(now: datetime, event: Event) -> None:
            nonlocal comm, imp, spr
            still: list[_PendingOrder] = []
            for po in pending:
                if po.effective_ts > now:
                    still.append(po)
                    continue
                if po.order.type is OrderType.MARKET:
                    f = marketable_fill(
                        po.order, nbbo_view.get(po.order.symbol), now,
                        adv_dollar=adv_dollar.get(po.order.symbol, 0.0),
                        impact_coef_bps=impact_coef_bps,
                        commission_per_share=cfg.commission_per_share,
                    )
                else:
                    f = limit_fill(po.order, event, now, commission_per_share=cfg.commission_per_share) \
                        if isinstance(event, (Trade, QuoteBar)) else None
                    if f is None:
                        still.append(po)  # limit rests until it trades through
                        continue
                if f is None:
                    still.append(po)
                    continue
                pf.apply_fill(f)
                fills.append(f)
                comm += f.commission
                imp += f.impact_cost
                spr += f.spread_cost
            pending[:] = still

        for event in events:
            now = event.ts
            ctx._now = now
            # 1) fill any pending orders now effective (against prevailing NBBO / this event)
            _try_fill_pending(now, event)
            # 2) update the market view AFTER fills (fills see the prevailing, not this event's, NBBO)
            if isinstance(event, QuoteBar):
                nbbo_view[event.symbol] = event
                marks[event.symbol] = event.mid
            elif isinstance(event, Trade):
                marks[event.symbol] = event.price
            elif isinstance(event, Bar):
                marks[event.symbol] = event.close
            # 3) strategy reacts to this event
            for order in strategy.on_event(event, ctx):
                pending.append(_PendingOrder(order=order, effective_ts=now + cfg.latency))
            # 4) record equity
            equity_points.append((now, pf.equity(marks)))

        idx = pd.DatetimeIndex([t for t, _ in equity_points])
        curve = pd.Series([e for _, e in equity_points], index=idx)
        return BacktestResult(
            equity_curve=curve,
            fills=fills,
            costs=CostBreakdown(commission=comm, impact=imp, spread=spr, financing=0.0),
        )
```

Note on the latency test: with quotes at :00 and :01 and 250 ms latency, the order signalled while
processing the :00 event has `effective_ts = :00.250`. It is not filled during the :00 step (the
`_try_fill_pending` at :00 runs before the order is appended). At the :01 event, `_try_fill_pending`
runs with `now = :01 >= :00.250`, and the NBBO view still holds the :00 quote *until step 2 updates
it* — but the marketable fill reads `nbbo_view` which is updated to :01 only at step 2, AFTER the
fill attempt. So the fill uses the **:00** NBBO. **The test expects the :01 ask (100.50).** Reconcile
by moving the market-view update (step 2) to run BEFORE `_try_fill_pending` so a post-latency order
fills against the current event's NBBO. Adjust the loop order to: (1) update market view, (2) fill
pending, (3) strategy, (4) record — and re-run. The test pins the intended semantics: an order fills
against the NBBO prevailing at/after its effective time, i.e. the :01 quote.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/intraday/sim/test_engine_latency.py -v`
Expected: PASS (1 passed) after ordering market-view update before pending-fill as the note describes.

- [ ] **Step 5: Add a no-lookahead test + commit**

```python
# append to tests/intraday/sim/test_engine_latency.py
class PeekStrategy:
    """Records the best ask it can see at each event; must never see a future quote."""
    def __init__(self):
        self.seen_asks = []
    def on_event(self, event, ctx):
        q = ctx.nbbo("AAPL")
        self.seen_asks.append(None if q is None else q.ask)
        return []


def test_context_never_sees_future_quote():
    events = [_q(0, 99.98, 100.02), _q(1, 100.40, 100.50)]
    strat = PeekStrategy()
    BacktestEngine(EngineConfig()).run(strat, events, adv_dollar={"AAPL": 0.0}, impact_coef_bps=0.0)
    # at the first event the only visible ask is the first event's (100.02); never 100.50 early
    assert strat.seen_asks[0] == 100.02
    assert strat.seen_asks[1] == 100.50
```

Run: `uv run pytest tests/intraday/sim/test_engine_latency.py -v` → 2 passed.

```bash
git add quant/intraday/sim/engine.py tests/intraday/sim/test_engine_latency.py
git commit -m "feat(intraday): event-driven engine with latency queue + no-lookahead"
```

---

## Task 7: Determinism + cost-charged + end-to-end fixture

**Files:**
- Test: `tests/intraday/sim/test_engine_e2e.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/intraday/sim/test_engine_e2e.py
from datetime import datetime, timedelta, timezone

from quant.intraday.data.events import QuoteBar
from quant.intraday.sim.engine import BacktestEngine, EngineConfig
from quant.intraday.strategy import Order, Side


def _q(day, sec, bid, ask):
    return QuoteBar(
        ts=datetime(2023, 6, day, 20, 0, sec, tzinfo=timezone.utc),
        symbol="AAPL", bid=bid, ask=ask, bid_size=100, ask_size=100,
    )


class BuyDay1SellDay2:
    def on_event(self, event, ctx):
        if event.ts.day == 1 and ctx.position("AAPL") == 0:
            return [Order("AAPL", Side.BUY, 100)]
        if event.ts.day == 2 and ctx.position("AAPL") > 0:
            return [Order("AAPL", Side.SELL, 100)]
        return []


def _events():
    # two quotes per day so the post-latency fill has a NBBO to hit
    return [_q(1, 0, 99.98, 100.02), _q(1, 1, 99.98, 100.02),
            _q(2, 0, 100.98, 101.02), _q(2, 1, 100.98, 101.02)]


def test_costs_are_always_charged_and_result_is_deterministic():
    cfg = EngineConfig(latency=timedelta(milliseconds=250), commission_per_share=0.01)
    r1 = BacktestEngine(cfg).run(BuyDay1SellDay2(), _events(), adv_dollar={"AAPL": 1e9}, impact_coef_bps=10.0)
    r2 = BacktestEngine(cfg).run(BuyDay1SellDay2(), _events(), adv_dollar={"AAPL": 1e9}, impact_coef_bps=10.0)
    assert len(r1.fills) == 2
    assert r1.costs.commission == 2.0          # 2 fills * 100 sh * 0.01
    assert r1.costs.spread > 0.0               # crossed the spread both ways
    assert r1.equity_curve.equals(r2.equity_curve)   # deterministic
    # bought ~100.02, sold ~101.02 -> gross +~100 minus costs; net still positive here
    assert r1.daily_returns().abs().sum() > 0.0
```

- [ ] **Step 2: Run test to verify it fails, then passes**

Run: `uv run pytest tests/intraday/sim/test_engine_e2e.py -v`
Expected: this exercises only already-built code; it should PASS once the Task 6 loop order is correct. If `len(fills) != 2`, check that the day-2 sell sees a post-latency NBBO (two quotes per day provide it). Fix nothing in source unless a real bug surfaces; the test pins intended behavior.

- [ ] **Step 3: Commit**

```bash
git add tests/intraday/sim/test_engine_e2e.py
git commit -m "test(intraday): engine determinism + costs-always-charged e2e"
```

---

## Task 8: Overnight financing on carried positions

**Files:**
- Modify: `quant/intraday/sim/engine.py` (charge financing when the session date advances)
- Test: `tests/intraday/sim/test_engine_financing.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/intraday/sim/test_engine_financing.py
from datetime import datetime, timedelta, timezone

from quant.intraday.data.events import QuoteBar
from quant.intraday.sim.engine import BacktestEngine, EngineConfig
from quant.intraday.strategy import Order, Side


def _q(day, sec, bid, ask):
    return QuoteBar(ts=datetime(2023, 6, day, 20, 0, sec, tzinfo=timezone.utc),
                    symbol="AAPL", bid=bid, ask=ask, bid_size=100, ask_size=100)


class ShortAndHold:
    def on_event(self, event, ctx):
        if event.ts.day == 1 and ctx.position("AAPL") == 0:
            return [Order("AAPL", Side.SELL, 100)]   # open a short, carry it overnight
        return []


def test_overnight_short_incurs_borrow_financing():
    events = [_q(1, 0, 99.98, 100.02), _q(1, 1, 99.98, 100.02),
              _q(2, 0, 99.98, 100.02), _q(2, 1, 99.98, 100.02)]
    cfg = EngineConfig(latency=timedelta(milliseconds=250), commission_per_share=0.0,
                       annual_borrow_bps=50.0)
    res = BacktestEngine(cfg).run(ShortAndHold(), events, adv_dollar={"AAPL": 1e12}, impact_coef_bps=0.0)
    assert res.costs.financing > 0.0   # one overnight borrow charge on ~100*~100 short notional
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/intraday/sim/test_engine_financing.py -v`
Expected: FAIL — `EngineConfig` has no `annual_borrow_bps`; financing stays 0.

- [ ] **Step 3: Implement**

Add to `EngineConfig`: `annual_borrow_bps: float = 0.0` and `annual_financing_bps: float = 0.0`.

In `engine.py`, import `from quant.backtest.financing import financing_charge`. Track the prior
event's calendar date and the prior marks; when an event's date is later than the previous event's
date and there are carried positions, charge financing for the elapsed days:

```python
# inside run(), before the main loop:
fin = 0.0
prev_date = None
prev_marks: dict[str, float] = {}
# ... inside the loop, right after step 1 (fill pending) and step 2 (update marks),
# BEFORE recording equity:
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
prev_date = ev_date
prev_marks = dict(marks)
```

Then include `financing=fin` in the `CostBreakdown`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/intraday/sim/test_engine_financing.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add quant/intraday/sim/engine.py tests/intraday/sim/test_engine_financing.py
git commit -m "feat(intraday): overnight borrow/financing on carried positions"
```

---

## Task 9: Package exports + full-suite green

**Files:**
- Modify: `quant/intraday/sim/__init__.py`
- Test: `tests/intraday/sim/test_exports.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/intraday/sim/test_exports.py
def test_sim_exports():
    from quant.intraday.sim import BacktestEngine, BacktestResult, EngineConfig, Portfolio
    from quant.intraday.strategy import IntradayStrategy, Order, OrderType, Side
    assert all([BacktestEngine, BacktestResult, EngineConfig, Portfolio,
                IntradayStrategy, Order, OrderType, Side])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/intraday/sim/test_exports.py -v`
Expected: FAIL — names not exported from `quant.intraday.sim`.

- [ ] **Step 3: Implement**

```python
# quant/intraday/sim/__init__.py
"""Intraday backtest + execution simulation."""

from quant.intraday.sim.engine import BacktestEngine, EngineConfig
from quant.intraday.sim.fills import Fill, limit_fill, marketable_fill
from quant.intraday.sim.portfolio import Portfolio
from quant.intraday.sim.result import BacktestResult, CostBreakdown

__all__ = [
    "BacktestEngine", "EngineConfig", "Fill", "limit_fill", "marketable_fill",
    "Portfolio", "BacktestResult", "CostBreakdown",
]
```

- [ ] **Step 4: Run the test + full gates**

Run: `uv run pytest tests/intraday/sim/test_exports.py -v` → PASS.

Run: `uv run pytest -m "not alpaca" -q && uv run ruff check quant/intraday tests/intraday && uv run ruff format quant/intraday tests/intraday && uv run mypy quant/intraday`
Expected: full suite green, lint/format clean, mypy clean. Fix any issues before committing.

- [ ] **Step 5: Commit**

```bash
git add quant/intraday/sim/__init__.py tests/intraday/sim/test_exports.py
git commit -m "feat(intraday): sim package exports; execution simulator complete"
```

---

## Self-Review

**1. Spec coverage:**
- IntradayStrategy protocol shared with D → Task 1. ✓
- Hybrid fills (marketable far-touch + impact + commission; limit trade-through, no queue credit) → Tasks 2, 3. ✓
- Reuse impact (`market_impact_bps`) → Task 2; financing (`financing_charge`) → Task 8; metrics → Task 5. ✓
- Portfolio P&L / mark-to-mid → Task 4. ✓
- Event loop + latency pending-queue (default 250 ms) → Task 6. ✓
- No-lookahead → Task 6 (`test_context_never_sees_future_quote`). ✓
- Determinism + costs-always-charged → Task 7. ✓
- Daily-resampled returns feeding charter metrics → Task 5. ✓
- BacktestResult cost breakdown (commission/impact/spread/financing) → Tasks 5, 6, 8. ✓

**Gap flagged (deferred, not silent):** the spec's optional `on_start`/`on_end` strategy hooks are
omitted (YAGNI for B; add in D when the live loop needs lifecycle hooks). Limit-order *cancel* is
not modelled (limits simply rest until they trade through or the run ends) — also a D concern, noted
in the spec.

**2. Placeholder scan:** every code step has runnable code. The Task 6 implementation note is an
explicit reconciliation instruction (loop ordering), not a placeholder — it tells the engineer the
exact ordering the test pins. Re-state for clarity: **the engine loop order must be (1) update market
view from the event, (2) fill pending orders effective ≤ now, (3) call strategy, (4) record equity** —
write it that way; the in-code comment showing the alternative is annotated as the thing to fix.

**3. Type consistency:** `Side`/`OrderType`/`Order` (Task 1) are used identically in fills (2,3),
portfolio (4), engine (6). `Fill` fields (ts, symbol, side, qty, price, commission, impact_cost,
spread_cost, signed_qty) are consistent across fills/portfolio/engine. `EngineConfig` gains
`annual_borrow_bps`/`annual_financing_bps` in Task 8 (additive). `BacktestResult`(equity_curve,
fills, costs) and `CostBreakdown`(commission, impact, spread, financing) are consistent across Tasks
5/6/8.

**Correction applied inline:** Task 6's loop ordering is stated authoritatively in the self-review
(update view → fill → strategy → record) so the implementer doesn't follow the deliberately-wrong
ordering shown in the annotated code comment.
