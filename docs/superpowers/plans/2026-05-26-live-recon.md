# Live Reconciliation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `scripts/reconcile_live.py` + supporting modules to compare actual Alpaca paper fills against the project's backtest cost model (5 bps slippage), writing a dated Markdown report to `docs/live-recon/YYYY-MM-DD.md`.

**Architecture:** Three units. `quant/live/recon.py` (pure logic — takes DataFrames + a bar fetcher, returns a `ReconciliationReport`). `quant/live/recon_render.py` (pure Markdown formatter). `scripts/reconcile_live.py` (thin I/O orchestrator). Plus one prerequisite addition: `AlpacaClient.list_orders()` doesn't exist yet and must be added before the orchestrator can call it.

**Tech Stack:** Python 3.12, `uv`, `pandas`, `pytest`, `ruff`, `mypy --strict`, `alpaca-py` SDK, existing `quant/data/bars.py` for signal-price lookup, existing `quant/util/calendar.py` (NYSE) for prior-trading-day computation.

**Spec:** `docs/superpowers/specs/2026-05-26-live-recon-design.md`

---

## File Structure

**New files:**
- `quant/live/recon.py` (~150 LOC) — `ReconRow`, `ReconciliationReport`, `reconcile()` function
- `quant/live/recon_render.py` (~100 LOC) — `render_markdown(report) -> str`
- `scripts/reconcile_live.py` (~60 LOC) — CLI orchestrator
- `tests/live/test_recon.py` — unit tests for pure logic
- `tests/live/test_recon_render.py` — snapshot tests for renderer
- `docs/live-recon/.gitkeep` — directory marker
- `docs/live-recon/2026-05-26.md` — first real report (smoke output)

**Modified files:**
- `quant/execution/alpaca.py` — add `OrderRow` dataclass + `AlpacaClient.list_orders()` method

---

## Task 1: Add `list_orders()` to AlpacaClient

The spec assumes `AlpacaClient.list_orders()` exists; it doesn't. Add it first so the rest of the plan can call it.

**Files:**
- Modify: `quant/execution/alpaca.py`
- Modify: `tests/live/` — add `tests/live/test_alpaca_list_orders.py`

- [ ] **Step 1: Write the failing test**

Create `tests/live/test_alpaca_list_orders.py`:

```python
"""Tests for AlpacaClient.list_orders()."""

from __future__ import annotations

from datetime import date, datetime, timezone
from unittest.mock import MagicMock

from quant.execution.alpaca import AlpacaClient, OrderRow


def _fake_order(
    *,
    client_order_id: str = "trend-20260526-SPY-deadbeef",
    symbol: str = "SPY",
    side: str = "buy",
    qty: str = "69",
    filled_qty: str = "69",
    filled_avg_price: str | None = "500.12",
    submitted_at: datetime = datetime(2026, 5, 26, 19, 55, tzinfo=timezone.utc),
    filled_at: datetime | None = datetime(2026, 5, 26, 19, 55, 4, tzinfo=timezone.utc),
    status: str = "filled",
) -> MagicMock:
    o = MagicMock()
    o.client_order_id = client_order_id
    o.symbol = symbol
    o.side = MagicMock(value=side)
    o.qty = qty
    o.filled_qty = filled_qty
    o.filled_avg_price = filled_avg_price
    o.submitted_at = submitted_at
    o.filled_at = filled_at
    o.status = MagicMock(value=status)
    return o


def test_list_orders_returns_typed_rows() -> None:
    client = AlpacaClient.__new__(AlpacaClient)
    client._trading = MagicMock()  # type: ignore[attr-defined]
    client._trading.get_orders.return_value = [_fake_order()]

    rows = client.list_orders(since=date(2026, 5, 26), until=date(2026, 5, 26))

    assert len(rows) == 1
    row = rows[0]
    assert isinstance(row, OrderRow)
    assert row.client_order_id == "trend-20260526-SPY-deadbeef"
    assert row.symbol == "SPY"
    assert row.side == "buy"
    assert row.submitted_qty == 69
    assert row.filled_qty == 69
    assert row.filled_avg_price == 500.12
    assert row.status == "filled"


def test_list_orders_handles_unfilled() -> None:
    client = AlpacaClient.__new__(AlpacaClient)
    client._trading = MagicMock()  # type: ignore[attr-defined]
    client._trading.get_orders.return_value = [
        _fake_order(filled_qty="0", filled_avg_price=None, filled_at=None, status="canceled")
    ]

    rows = client.list_orders(since=date(2026, 5, 26), until=date(2026, 5, 26))

    assert rows[0].filled_qty == 0
    assert rows[0].filled_avg_price is None
    assert rows[0].filled_at is None
    assert rows[0].status == "canceled"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/live/test_alpaca_list_orders.py -v`
Expected: FAIL with `ImportError: cannot import name 'OrderRow'` or `AttributeError: ... has no attribute 'list_orders'`.

- [ ] **Step 3: Add `OrderRow` and `list_orders()`**

Edit `quant/execution/alpaca.py`. Add the import near the others:

```python
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest
```

(replacing the existing single-symbol `MarketOrderRequest` import line)

Add `from datetime import date, datetime` (replacing the existing `from datetime import date`).

Add the `OrderRow` dataclass right after the existing `PositionRow` dataclass (around line 44):

```python
@dataclass(frozen=True)
class OrderRow:
    """A single Alpaca order with fill outcome, in plain Python types."""

    client_order_id: str
    symbol: str
    side: str  # "buy" | "sell"
    submitted_qty: int
    filled_qty: int
    filled_avg_price: float | None
    submitted_at: datetime
    filled_at: datetime | None
    status: str  # alpaca-py OrderStatus.value, e.g. "filled" | "canceled" | "rejected"
```

Add the `list_orders` method to `AlpacaClient` (place it between `positions` and `submit_order`):

```python
    def list_orders(
        self,
        *,
        since: date,
        until: date,
        limit: int = 500,
    ) -> list[OrderRow]:
        """Fetch orders submitted on [since, until] inclusive, newest first."""
        req = GetOrdersRequest(
            status="all",  # type: ignore[arg-type]
            after=datetime.combine(since, datetime.min.time()),
            until=datetime.combine(until, datetime.max.time()),
            limit=limit,
        )
        orders = self._trading.get_orders(filter=req)
        rows: list[OrderRow] = []
        for o in orders:
            filled_avg = o.filled_avg_price
            rows.append(
                OrderRow(
                    client_order_id=str(o.client_order_id),
                    symbol=str(o.symbol),
                    side=str(o.side.value),  # type: ignore[union-attr]
                    submitted_qty=_i(o.qty),
                    filled_qty=_i(o.filled_qty),
                    filled_avg_price=_f(filled_avg) if filled_avg is not None else None,
                    submitted_at=o.submitted_at,  # type: ignore[arg-type]
                    filled_at=o.filled_at,  # type: ignore[arg-type]
                    status=str(o.status.value),  # type: ignore[union-attr]
                )
            )
        return rows
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/live/test_alpaca_list_orders.py -v`
Expected: 2 passed.

- [ ] **Step 5: Lint & type-check**

Run: `uv run ruff check quant/execution/alpaca.py tests/live/test_alpaca_list_orders.py && uv run mypy --strict quant/execution/alpaca.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add quant/execution/alpaca.py tests/live/test_alpaca_list_orders.py
git commit -m "feat(alpaca): add OrderRow + AlpacaClient.list_orders() for live recon

Prerequisite for scripts/reconcile_live.py — fetches the fill-outcome
side of the reconciliation join (the project's own trades.parquet only
records intent). Returns typed OrderRow rows keyed on client_order_id."
```

---

## Task 2: Define recon data structures + happy-path logic

Create `quant/live/recon.py` with the `ReconRow` and `ReconciliationReport` dataclasses and the `reconcile()` function. Cover the clean 1:1 match case first.

**Files:**
- Create: `quant/live/recon.py`
- Create: `tests/live/test_recon.py`

- [ ] **Step 1: Write the failing test**

Create `tests/live/test_recon.py`:

```python
"""Tests for quant/live/recon.py — pure logic, no I/O."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Callable

import pandas as pd

from quant.execution.alpaca import OrderRow
from quant.live.recon import ReconciliationReport, ReconRow, reconcile

UTC = timezone.utc


def _trade(
    *,
    coid: str = "trend-20260526-SPY-deadbeef",
    strategy: str = "trend",
    symbol: str = "SPY",
    side: str = "buy",
    qty: int = 100,
    dt: date = date(2026, 5, 26),
) -> dict[str, object]:
    return {
        "date": dt,
        "strategy": strategy,
        "symbol": symbol,
        "side": side,
        "qty": qty,
        "client_order_id": coid,
        "dry_run": False,
    }


def _order(
    *,
    coid: str = "trend-20260526-SPY-deadbeef",
    symbol: str = "SPY",
    side: str = "buy",
    submitted_qty: int = 100,
    filled_qty: int = 100,
    filled_avg_price: float | None = 500.12,
    submitted_at: datetime = datetime(2026, 5, 26, 19, 55, tzinfo=UTC),
    filled_at: datetime | None = datetime(2026, 5, 26, 19, 55, 4, tzinfo=UTC),
    status: str = "filled",
) -> OrderRow:
    return OrderRow(
        client_order_id=coid,
        symbol=symbol,
        side=side,
        submitted_qty=submitted_qty,
        filled_qty=filled_qty,
        filled_avg_price=filled_avg_price,
        submitted_at=submitted_at,
        filled_at=filled_at,
        status=status,
    )


def _bar_fetcher_for(prices: dict[tuple[str, date], float]) -> Callable[[str, date], float | None]:
    """Return a fake bar fetcher that returns the close price for (symbol, prior_trading_day)."""

    def fetch(symbol: str, prior_trading_day: date) -> float | None:
        return prices.get((symbol, prior_trading_day))

    return fetch


def test_reconcile_clean_one_to_one_buy() -> None:
    trades = pd.DataFrame([_trade()])
    orders = [_order()]
    bars = _bar_fetcher_for({("SPY", date(2026, 5, 22)): 499.87})  # prior trading day

    report = reconcile(
        trades=trades,
        orders=orders,
        bar_fetcher=bars,
        modeled_slippage_bps=5.0,
        since=date(2026, 5, 26),
        until=date(2026, 5, 26),
    )

    assert isinstance(report, ReconciliationReport)
    assert report.modeled_slippage_bps == 5.0
    assert len(report.rows) == 1
    row = report.rows[0]
    assert isinstance(row, ReconRow)
    assert row.status == "filled"
    assert row.signal_price == 499.87
    assert row.fill_price == 500.12
    # Buy: (500.12 - 499.87) / 499.87 * 1e4 ≈ 5.0014 bps
    assert row.slippage_bps is not None
    assert abs(row.slippage_bps - 5.001) < 0.01
    assert row.fill_lag_seconds == 4.0


def test_reconcile_clean_one_to_one_sell_signed_correctly() -> None:
    trades = pd.DataFrame([_trade(side="sell")])
    orders = [_order(side="sell", filled_avg_price=499.62)]  # received less than 499.87
    bars = _bar_fetcher_for({("SPY", date(2026, 5, 22)): 499.87})

    report = reconcile(
        trades=trades,
        orders=orders,
        bar_fetcher=bars,
        modeled_slippage_bps=5.0,
        since=date(2026, 5, 26),
        until=date(2026, 5, 26),
    )

    row = report.rows[0]
    # Sell: (499.87 - 499.62) / 499.87 * 1e4 ≈ 5.001 bps (positive = received less)
    assert row.slippage_bps is not None
    assert abs(row.slippage_bps - 5.001) < 0.01
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/live/test_recon.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'quant.live.recon'`.

- [ ] **Step 3: Implement `quant/live/recon.py`**

Create `quant/live/recon.py`:

```python
"""Pure reconciliation logic: join local trade intents against Alpaca fill outcomes.

Inputs come in as already-loaded DataFrames + iterables — no Alpaca calls, no
file I/O, no bar fetching. The orchestrator in scripts/reconcile_live.py wires
those in. Keep this module side-effect-free so tests can stay synchronous and
network-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Iterable

import pandas as pd

from quant.execution.alpaca import OrderRow
from quant.util.calendar import prior_trading_day

BarFetcher = Callable[[str, date], float | None]
"""(symbol, prior_trading_day) -> close price, or None if unavailable."""


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
    status: str  # filled | partial | rejected | missing | no_signal_price


@dataclass
class ReconciliationReport:
    since: date
    until: date
    modeled_slippage_bps: float
    rows: list[ReconRow] = field(default_factory=list)


def _slippage_bps(side: str, signal_price: float, fill_price: float) -> float:
    if side == "buy":
        return (fill_price - signal_price) / signal_price * 1e4
    return (signal_price - fill_price) / signal_price * 1e4


def reconcile(
    *,
    trades: pd.DataFrame,
    orders: Iterable[OrderRow],
    bar_fetcher: BarFetcher,
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
                    slippage_bps=None,
                    fill_lag_seconds=None,
                    status="missing",
                )
            )
            continue

        signal_price = bar_fetcher(str(t["symbol"]), prior_trading_day(submission_date))
        fill_lag = None
        if order.filled_at is not None:
            fill_lag = (order.filled_at - order.submitted_at).total_seconds()

        if order.status in {"canceled", "rejected", "expired"}:
            status = "rejected"
            slippage = None
        elif order.filled_qty == 0:
            status = "rejected"
            slippage = None
        elif signal_price is None or order.filled_avg_price is None:
            status = "no_signal_price" if signal_price is None else "rejected"
            slippage = None
        else:
            slippage = _slippage_bps(order.side, signal_price, order.filled_avg_price)
            status = "filled" if order.filled_qty >= order.submitted_qty else "partial"

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
                slippage_bps=slippage,
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
```

- [ ] **Step 4: Verify `quant/util/calendar.prior_trading_day` exists**

Run: `uv run python -c "from quant.util.calendar import prior_trading_day; print(prior_trading_day.__doc__ or 'ok')"`

If this errors (function not found): add a small helper to `quant/util/calendar.py` before continuing. Use the existing NYSE calendar object in that module and define:

```python
def prior_trading_day(asof: date) -> date:
    """Return the most recent NYSE trading day strictly before ``asof``."""
    cal = _nyse_calendar()  # reuse whatever the module already exposes
    sched = cal.schedule(start_date=asof - timedelta(days=10), end_date=asof - timedelta(days=1))
    return sched.index[-1].date()
```

Adjust the import (`from datetime import timedelta`) and the `_nyse_calendar()` reference to whatever the file already uses. Add a one-line test in `tests/util/test_calendar.py` if a test file exists; otherwise skip.

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/live/test_recon.py -v`
Expected: 2 passed.

- [ ] **Step 6: Lint & type-check**

Run: `uv run ruff check quant/live/recon.py tests/live/test_recon.py && uv run mypy --strict quant/live/recon.py`
Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add quant/live/recon.py tests/live/test_recon.py quant/util/calendar.py tests/util/test_calendar.py 2>/dev/null; git add quant/live/recon.py tests/live/test_recon.py
git commit -m "feat(live): add reconcile() — pure logic for joining trade intent to fill outcome

Defines ReconRow + ReconciliationReport dataclasses and the reconcile()
function. Covers happy-path clean 1:1 match for both buys and sells with
signed slippage_bps. No I/O — bar lookups go through an injected fetcher."
```

---

## Task 3: Add fidelity scenarios (missing / rejected / partial / no_signal_price)

Extend the test suite to cover the four failure modes from spec §4 (other than "trades.parquet empty" and "Alpaca API error" which are orchestrator concerns).

**Files:**
- Modify: `tests/live/test_recon.py`

- [ ] **Step 1: Add failing tests for the four fidelity scenarios**

Append to `tests/live/test_recon.py`:

```python
def test_reconcile_missing_order() -> None:
    trades = pd.DataFrame([_trade()])
    orders: list[OrderRow] = []  # nothing came back from Alpaca
    bars = _bar_fetcher_for({})

    report = reconcile(
        trades=trades, orders=orders, bar_fetcher=bars,
        modeled_slippage_bps=5.0,
        since=date(2026, 5, 26), until=date(2026, 5, 26),
    )

    row = report.rows[0]
    assert row.status == "missing"
    assert row.filled_qty == 0
    assert row.signal_price is None
    assert row.slippage_bps is None


def test_reconcile_rejected_order() -> None:
    trades = pd.DataFrame([_trade()])
    orders = [_order(filled_qty=0, filled_avg_price=None, filled_at=None, status="rejected")]
    bars = _bar_fetcher_for({("SPY", date(2026, 5, 22)): 499.87})

    report = reconcile(
        trades=trades, orders=orders, bar_fetcher=bars,
        modeled_slippage_bps=5.0,
        since=date(2026, 5, 26), until=date(2026, 5, 26),
    )

    row = report.rows[0]
    assert row.status == "rejected"
    assert row.slippage_bps is None


def test_reconcile_partial_fill_computes_slippage_on_filled_portion() -> None:
    trades = pd.DataFrame([_trade(qty=100)])
    orders = [_order(submitted_qty=100, filled_qty=60, filled_avg_price=500.12, status="partially_filled")]
    bars = _bar_fetcher_for({("SPY", date(2026, 5, 22)): 499.87})

    report = reconcile(
        trades=trades, orders=orders, bar_fetcher=bars,
        modeled_slippage_bps=5.0,
        since=date(2026, 5, 26), until=date(2026, 5, 26),
    )

    row = report.rows[0]
    assert row.status == "partial"
    assert row.filled_qty == 60
    assert row.slippage_bps is not None
    assert abs(row.slippage_bps - 5.001) < 0.01


def test_reconcile_no_signal_price_marks_row() -> None:
    trades = pd.DataFrame([_trade()])
    orders = [_order()]
    bars = _bar_fetcher_for({})  # bar fetch returns None

    report = reconcile(
        trades=trades, orders=orders, bar_fetcher=bars,
        modeled_slippage_bps=5.0,
        since=date(2026, 5, 26), until=date(2026, 5, 26),
    )

    row = report.rows[0]
    assert row.status == "no_signal_price"
    assert row.signal_price is None
    assert row.fill_price == 500.12  # still recorded
    assert row.slippage_bps is None
```

- [ ] **Step 2: Run tests to verify they pass**

The implementation from Task 2 should already handle all four cases. Run: `uv run pytest tests/live/test_recon.py -v`
Expected: 6 passed.

If any test fails, fix `reconcile()` in `quant/live/recon.py` to handle the case. The classification order in the existing impl is: missing → rejected (canceled/rejected/expired) → rejected (filled_qty==0) → no_signal_price → partial/filled. Audit that path against the failure.

- [ ] **Step 3: Lint & type-check**

Run: `uv run ruff check tests/live/test_recon.py && uv run mypy --strict quant/live/recon.py`
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add tests/live/test_recon.py quant/live/recon.py
git commit -m "test(recon): cover missing/rejected/partial/no_signal_price fidelity cases"
```

---

## Task 4: Add per-strategy / per-symbol aggregation helpers

The renderer in Task 5 needs aggregate stats. Add them as methods on `ReconciliationReport` so they're testable independently of the Markdown layer.

**Files:**
- Modify: `quant/live/recon.py`
- Modify: `tests/live/test_recon.py`

- [ ] **Step 1: Write failing test for aggregations**

Append to `tests/live/test_recon.py`:

```python
def test_report_aggregate_by_strategy() -> None:
    trades = pd.DataFrame([
        _trade(coid="trend-20260526-SPY-a", strategy="trend", symbol="SPY", qty=100),
        _trade(coid="trend-20260526-DBC-b", strategy="trend", symbol="DBC", qty=50),
        _trade(coid="momentum-20260526-EFA-c", strategy="momentum", symbol="EFA", qty=80),
    ])
    orders = [
        _order(coid="trend-20260526-SPY-a", symbol="SPY", submitted_qty=100, filled_qty=100, filled_avg_price=500.12),
        _order(coid="trend-20260526-DBC-b", symbol="DBC", submitted_qty=50, filled_qty=50, filled_avg_price=25.05),
        _order(coid="momentum-20260526-EFA-c", symbol="EFA", submitted_qty=80, filled_qty=80, filled_avg_price=80.20),
    ]
    bars = _bar_fetcher_for({
        ("SPY", date(2026, 5, 22)): 499.87,
        ("DBC", date(2026, 5, 22)): 25.00,
        ("EFA", date(2026, 5, 22)): 80.00,
    })

    report = reconcile(
        trades=trades, orders=orders, bar_fetcher=bars,
        modeled_slippage_bps=5.0,
        since=date(2026, 5, 26), until=date(2026, 5, 26),
    )

    by_strategy = report.aggregate_by_strategy()
    assert set(by_strategy.keys()) == {"trend", "momentum"}
    assert by_strategy["trend"]["n_filled"] == 2
    assert by_strategy["momentum"]["n_filled"] == 1
    # trend mean slippage = mean of ~5 bps and 20 bps = ~12.5
    assert 10 < by_strategy["trend"]["mean_slippage_bps"] < 15


def test_report_aggregate_by_symbol() -> None:
    trades = pd.DataFrame([
        _trade(coid="trend-20260526-SPY-a", symbol="SPY", qty=100),
        _trade(coid="trend-20260526-SPY-b", symbol="SPY", qty=50),
    ])
    orders = [
        _order(coid="trend-20260526-SPY-a", symbol="SPY", submitted_qty=100, filled_qty=100, filled_avg_price=500.12),
        _order(coid="trend-20260526-SPY-b", symbol="SPY", submitted_qty=50, filled_qty=50, filled_avg_price=500.50),
    ]
    bars = _bar_fetcher_for({("SPY", date(2026, 5, 22)): 499.87})

    report = reconcile(
        trades=trades, orders=orders, bar_fetcher=bars,
        modeled_slippage_bps=5.0,
        since=date(2026, 5, 26), until=date(2026, 5, 26),
    )

    by_symbol = report.aggregate_by_symbol()
    assert set(by_symbol.keys()) == {"SPY"}
    assert by_symbol["SPY"]["n_filled"] == 2
    assert by_symbol["SPY"]["mean_slippage_bps"] is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/live/test_recon.py -v -k aggregate`
Expected: FAIL with `AttributeError: 'ReconciliationReport' object has no attribute 'aggregate_by_strategy'`.

- [ ] **Step 3: Add aggregation methods to `ReconciliationReport`**

Replace the `ReconciliationReport` dataclass in `quant/live/recon.py` with:

```python
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
            out[grp_key] = {
                "n_total": len(rows),
                "n_filled": len(filled),
                "n_partial": sum(1 for r in rows if r.status == "partial"),
                "n_rejected": sum(1 for r in rows if r.status == "rejected"),
                "n_missing": sum(1 for r in rows if r.status == "missing"),
                "mean_slippage_bps": mean(r.slippage_bps for r in filled) if filled else None,
                "median_fill_lag_s": (
                    sorted(lags)[len(lags) // 2] if lags else None
                ),
            }
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/live/test_recon.py -v`
Expected: 8 passed.

- [ ] **Step 5: Lint & type-check**

Run: `uv run ruff check quant/live/recon.py && uv run mypy --strict quant/live/recon.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add quant/live/recon.py tests/live/test_recon.py
git commit -m "feat(recon): add per-strategy/per-symbol aggregation methods"
```

---

## Task 5: Markdown renderer with snapshot tests

`quant/live/recon_render.py` formats a `ReconciliationReport` into the report Markdown. Snapshot tests guard against accidental format changes.

**Files:**
- Create: `quant/live/recon_render.py`
- Create: `tests/live/test_recon_render.py`

- [ ] **Step 1: Write the failing test**

Create `tests/live/test_recon_render.py`:

```python
"""Snapshot tests for the Markdown renderer."""

from __future__ import annotations

from datetime import date

from quant.live.recon import ReconciliationReport, ReconRow
from quant.live.recon_render import render_markdown


def _fixture_report() -> ReconciliationReport:
    return ReconciliationReport(
        since=date(2026, 5, 26),
        until=date(2026, 5, 26),
        modeled_slippage_bps=5.0,
        rows=[
            ReconRow(
                client_order_id="trend-20260526-SPY-a",
                strategy="trend", symbol="SPY", side="buy",
                submission_date=date(2026, 5, 26),
                submitted_qty=100, filled_qty=100,
                signal_price=499.87, fill_price=500.12,
                slippage_bps=5.001, fill_lag_seconds=4.0,
                status="filled",
            ),
            ReconRow(
                client_order_id="trend-20260526-DBC-b",
                strategy="trend", symbol="DBC", side="buy",
                submission_date=date(2026, 5, 26),
                submitted_qty=50, filled_qty=0,
                signal_price=25.00, fill_price=None,
                slippage_bps=None, fill_lag_seconds=None,
                status="rejected",
            ),
        ],
    )


def test_render_markdown_contains_required_sections() -> None:
    md = render_markdown(_fixture_report())

    assert "# Live Reconciliation 2026-05-26" in md
    assert "## Summary" in md
    assert "## Slippage (filled orders)" in md
    assert "## Timing" in md
    assert "## Fidelity" in md
    assert "## Per-symbol breakdown" in md
    # modeled benchmark surfaces
    assert "5.0 bps" in md or "5.00 bps" in md
    # both rows present
    assert "SPY" in md and "DBC" in md
    # rejected row appears in fidelity section
    assert "rejected" in md


def test_render_markdown_empty_report_still_valid() -> None:
    empty = ReconciliationReport(
        since=date(2026, 5, 26),
        until=date(2026, 5, 26),
        modeled_slippage_bps=5.0,
        rows=[],
    )
    md = render_markdown(empty)
    assert "# Live Reconciliation 2026-05-26" in md
    assert "no trades to reconcile" in md.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/live/test_recon_render.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'quant.live.recon_render'`.

- [ ] **Step 3: Implement `quant/live/recon_render.py`**

Create `quant/live/recon_render.py`:

```python
"""Markdown renderer for ReconciliationReport. Pure formatting, no I/O."""

from __future__ import annotations

from datetime import date
from io import StringIO

from quant.live.recon import ReconciliationReport, ReconRow


def render_markdown(report: ReconciliationReport) -> str:
    buf = StringIO()
    _write_header(buf, report)

    if not report.rows:
        buf.write("\n_no trades to reconcile in this window._\n")
        return buf.getvalue()

    _write_summary(buf, report)
    _write_slippage_section(buf, report)
    _write_timing_section(buf, report)
    _write_fidelity_section(buf, report)
    _write_per_symbol_section(buf, report)
    return buf.getvalue()


def _write_header(buf: StringIO, report: ReconciliationReport) -> None:
    title_date = report.until.isoformat()
    buf.write(f"# Live Reconciliation {title_date}\n\n")
    buf.write(f"**Window:** {report.since.isoformat()} → {report.until.isoformat()}  \n")
    buf.write(f"**Modeled slippage benchmark:** {report.modeled_slippage_bps:g} bps  \n")
    buf.write(f"**Total orders in window:** {len(report.rows)}  \n")


def _write_summary(buf: StringIO, report: ReconciliationReport) -> None:
    n_filled = sum(1 for r in report.rows if r.status == "filled")
    n_partial = sum(1 for r in report.rows if r.status == "partial")
    n_rejected = sum(1 for r in report.rows if r.status == "rejected")
    n_missing = sum(1 for r in report.rows if r.status == "missing")
    n_no_signal = sum(1 for r in report.rows if r.status == "no_signal_price")

    buf.write("\n## Summary\n\n")
    buf.write("| status | count |\n|---|---:|\n")
    buf.write(f"| filled | {n_filled} |\n")
    buf.write(f"| partial | {n_partial} |\n")
    buf.write(f"| rejected | {n_rejected} |\n")
    buf.write(f"| missing | {n_missing} |\n")
    buf.write(f"| no_signal_price | {n_no_signal} |\n")


def _write_slippage_section(buf: StringIO, report: ReconciliationReport) -> None:
    buf.write("\n## Slippage (filled orders)\n\n")
    by_strat = report.aggregate_by_strategy()
    buf.write("| strategy | n | mean slippage (bps) | vs modeled |\n|---|---:|---:|---:|\n")
    for strat, stats in sorted(by_strat.items()):
        mean_slip = stats["mean_slippage_bps"]
        if mean_slip is None:
            buf.write(f"| {strat} | 0 | — | — |\n")
            continue
        delta = float(mean_slip) - report.modeled_slippage_bps
        buf.write(f"| {strat} | {stats['n_filled']} | {float(mean_slip):.2f} | {delta:+.2f} |\n")


def _write_timing_section(buf: StringIO, report: ReconciliationReport) -> None:
    buf.write("\n## Timing\n\n")
    by_strat = report.aggregate_by_strategy()
    buf.write("| strategy | median fill lag (s) |\n|---|---:|\n")
    for strat, stats in sorted(by_strat.items()):
        lag = stats["median_fill_lag_s"]
        buf.write(f"| {strat} | {'—' if lag is None else f'{float(lag):.1f}'} |\n")


def _write_fidelity_section(buf: StringIO, report: ReconciliationReport) -> None:
    buf.write("\n## Fidelity\n\n")
    flagged = [r for r in report.rows if r.status in {"partial", "rejected", "missing", "no_signal_price"}]
    if not flagged:
        buf.write("_all orders filled cleanly._\n")
        return
    buf.write("| coid | symbol | side | submitted | filled | status |\n|---|---|---|---:|---:|---|\n")
    for r in flagged:
        buf.write(
            f"| `{r.client_order_id}` | {r.symbol} | {r.side} | "
            f"{r.submitted_qty} | {r.filled_qty} | {r.status} |\n"
        )


def _write_per_symbol_section(buf: StringIO, report: ReconciliationReport) -> None:
    buf.write("\n## Per-symbol breakdown\n\n")
    by_sym = report.aggregate_by_symbol()
    buf.write("| symbol | n filled | mean slippage (bps) | median lag (s) |\n|---|---:|---:|---:|\n")
    for sym, stats in sorted(by_sym.items()):
        slip = stats["mean_slippage_bps"]
        lag = stats["median_fill_lag_s"]
        slip_s = "—" if slip is None else f"{float(slip):.2f}"
        lag_s = "—" if lag is None else f"{float(lag):.1f}"
        buf.write(f"| {sym} | {stats['n_filled']} | {slip_s} | {lag_s} |\n")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/live/test_recon_render.py -v`
Expected: 2 passed.

- [ ] **Step 5: Lint & type-check**

Run: `uv run ruff check quant/live/recon_render.py tests/live/test_recon_render.py && uv run mypy --strict quant/live/recon_render.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add quant/live/recon_render.py tests/live/test_recon_render.py
git commit -m "feat(recon): add Markdown renderer with sectioned report layout"
```

---

## Task 6: Orchestrator script

Wire it together. `scripts/reconcile_live.py` loads `trades.parquet`, calls `AlpacaClient.list_orders()`, fetches bars, hands off to `reconcile()` then `render_markdown()`, writes to `docs/live-recon/YYYY-MM-DD.md`.

**Files:**
- Create: `scripts/reconcile_live.py`
- Create: `docs/live-recon/.gitkeep`

- [ ] **Step 1: Add the directory marker**

```bash
mkdir -p docs/live-recon
touch docs/live-recon/.gitkeep
```

- [ ] **Step 2: Implement the script**

Create `scripts/reconcile_live.py`:

```python
"""On-demand live reconciliation runner.

Joins ``data/live/trades.parquet`` (intent) with Alpaca's order history
(outcome), computes per-fill slippage / timing / fidelity, and writes a
dated Markdown report into ``docs/live-recon/YYYY-MM-DD.md``.

The report file is written to disk only; commit it by hand after review.

Usage::

    uv run python scripts/reconcile_live.py
    uv run python scripts/reconcile_live.py --since 2026-05-22 --until 2026-05-26
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from quant.backtest.engine import BacktestConfig
from quant.data.bars import BarRequest, get_bars
from quant.execution.alpaca import AlpacaClient
from quant.live.recon import ReconciliationReport, reconcile
from quant.live.recon_render import render_markdown
from quant.util.logging import logger


REPO_ROOT = Path(__file__).resolve().parents[1]
TRADES_PATH = REPO_ROOT / "data" / "live" / "trades.parquet"
REPORT_DIR = REPO_ROOT / "docs" / "live-recon"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconcile live Alpaca fills vs backtest model.")
    parser.add_argument("--since", type=date.fromisoformat, default=None,
                        help="Start date (inclusive). Default: 7 days before --until.")
    parser.add_argument("--until", type=date.fromisoformat, default=date.today(),
                        help="End date (inclusive). Default: today.")
    return parser.parse_args()


def _make_bar_fetcher() -> "callable[[str, date], float | None]":  # type: ignore[name-defined]
    cache: dict[tuple[str, date], float | None] = {}

    def fetch(symbol: str, asof: date) -> float | None:
        key = (symbol, asof)
        if key in cache:
            return cache[key]
        try:
            df = get_bars(BarRequest(symbols=[symbol], start=asof, end=asof))
            if df.empty:
                cache[key] = None
                return None
            close = float(df.loc[df["symbol"] == symbol, "close"].iloc[-1])
            cache[key] = close
            return close
        except Exception as exc:
            logger.warning("bar fetch failed for {} @ {}: {}", symbol, asof, exc)
            cache[key] = None
            return None

    return fetch


def main() -> int:
    args = _parse_args()
    until: date = args.until
    since: date = args.since or (until - timedelta(days=7))

    if not TRADES_PATH.exists():
        print(f"No trades.parquet at {TRADES_PATH}; nothing to reconcile.", file=sys.stderr)
        return 0

    trades_all = pd.read_parquet(TRADES_PATH)
    trades_all["date"] = pd.to_datetime(trades_all["date"]).dt.date
    mask = (trades_all["date"] >= since) & (trades_all["date"] <= until)
    trades = trades_all.loc[mask].copy()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / f"{until.isoformat()}.md"

    if trades.empty:
        report = ReconciliationReport(
            since=since, until=until,
            modeled_slippage_bps=BacktestConfig().slippage_bps,
            rows=[],
        )
        report_path.write_text(render_markdown(report))
        print(f"Wrote empty report: {report_path}")
        return 0

    try:
        client = AlpacaClient()
        orders = client.list_orders(since=since, until=until)
    except Exception as exc:
        print(f"Alpaca API error: {exc}", file=sys.stderr)
        return 1

    report = reconcile(
        trades=trades,
        orders=orders,
        bar_fetcher=_make_bar_fetcher(),
        modeled_slippage_bps=BacktestConfig().slippage_bps,
        since=since,
        until=until,
    )

    report_path.write_text(render_markdown(report))
    print(f"Wrote {report_path} — {len(report.rows)} reconciled rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Lint & type-check**

Run: `uv run ruff check scripts/reconcile_live.py && uv run mypy --strict scripts/reconcile_live.py`
Expected: no errors.

- [ ] **Step 4: Smoke run against the actual paper account**

Run: `uv run python scripts/reconcile_live.py --since 2026-05-22 --until 2026-05-26`
Expected: prints `Wrote docs/live-recon/2026-05-26.md — N reconciled rows.` where N matches `pd.read_parquet('data/live/trades.parquet').shape[0]` (currently 57).

Open the report and skim it: header present, summary table populated, slippage section shows per-strategy means, fidelity section either says "all orders filled cleanly" or lists flagged rows. If the layout looks broken, fix the renderer and re-run (overwriting is fine — deterministic).

- [ ] **Step 5: Run full test suite to confirm no regressions**

Run: `uv run pytest -q`
Expected: all prior tests still pass; new live/recon tests pass.

- [ ] **Step 6: Commit script + first report**

```bash
git add scripts/reconcile_live.py docs/live-recon/.gitkeep docs/live-recon/2026-05-26.md
git commit -m "feat(scripts): add reconcile_live.py — on-demand live-vs-backtest report

Loads data/live/trades.parquet, joins Alpaca order outcomes via
client_order_id, fetches signal-day bar closes for slippage comparison,
writes a sectioned Markdown report to docs/live-recon/YYYY-MM-DD.md.
First report covers the 5/22–5/26 paper trading window."
```

---

## Self-Review (post-write)

**Spec coverage:**
- §1 purpose & boundaries → Tasks 2/5/6 (recon module + renderer + script)
- §2 architecture (3-unit split) → Tasks 2 (recon), 5 (renderer), 6 (script) — matches exactly
- §3 data flow (join on coid + prior-trading-day signal price) → Task 2 Step 3 (impl uses `prior_trading_day` + `client_order_id` join)
- §3 slippage definition (signed by side) → Task 2 tests cover buy and sell signedness
- §4 error handling (missing/rejected/partial/no_signal/empty/Alpaca-error) → Task 3 tests cover the per-row cases; Task 6 script handles empty trades + Alpaca error
- §5 testing (~15 tests, no E2E) → 2 (Task 1) + 6 (Tasks 2+3) + 2 (Task 4) + 2 (Task 5) = 12 new tests. Slightly under 15, but covers every scenario the spec lists; adding filler tests just to hit a number violates YAGNI.
- §6 file layout → matches exactly
- §6 stateless CLI defaults (--until=today, --since=until-7d) → Task 6 `_parse_args` + `main` defaults match
- §6 commit policy (no auto-commit) → script writes to disk only; final commit in Task 6 is a one-time bootstrap (script itself + first report), reviewed by hand. Future runs leave the file un-staged.
- §6 SDK not MCP → Task 6 uses `AlpacaClient`, no MCP calls
- Prerequisite gap: `AlpacaClient.list_orders()` doesn't exist — caught and added as Task 1

**Placeholder scan:** No "TBD", "TODO", "implement later." Task 2 Step 4 conditionally adds a `prior_trading_day` helper if missing — that's an inline action, not a placeholder.

**Type consistency:**
- `OrderRow` (added Task 1) used in Task 2 test imports — names match
- `ReconRow` / `ReconciliationReport` defined Task 2, extended Task 4, consumed Task 5 — field names match across (`client_order_id`, `strategy`, `symbol`, `side`, `submission_date`, `submitted_qty`, `filled_qty`, `signal_price`, `fill_price`, `slippage_bps`, `fill_lag_seconds`, `status`)
- `aggregate_by_strategy` / `aggregate_by_symbol` method names match between Task 4 impl and Task 5 renderer calls
- `BarFetcher` callable signature (`(symbol, prior_trading_day) -> float | None`) matches between recon module, test fixtures, and orchestrator `_make_bar_fetcher`
- Status string set: `{filled, partial, rejected, missing, no_signal_price}` — same set used in recon logic, fidelity tests, renderer summary table

No inconsistencies found. Plan is internally complete.
