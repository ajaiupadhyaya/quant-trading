# Intraday Live Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the "spine" of the intraday/60s track — a continuous tick loop that trades a ring-fenced ETF sleeve (QQQ/IWM/DIA) on live Alpaca paper, additive to and isolated from the untouched daily system, with tight guardrails and an intraday mean-reversion proof-of-life strategy.

**Architecture:** A new `quant/intraday/live/` subpackage. A `loop` orchestrates a 60s lifecycle: session-check → guardrails-first → pull live quotes (`feed`) → strategy targets → sizing/caps (`guardrails`) → reconcile vs an internal `sleeve` ledger → submit unique-COID orders → journal. Reuses the existing `IntradayStrategy`/`Order` protocol, `QuoteBar` events, `AlpacaClient`, and the governance halt pattern. The daily system's deterministic-COID `submit_order` is NOT reused (it would reject repeated same-symbol orders); a new `submit_simple_order` with caller-supplied COID is added.

**Tech Stack:** Python 3.11+, `uv` for all commands, Click CLI, `alpaca-py` (`TradingClient`, `StockHistoricalDataClient`), pandas/parquet, pytest + hypothesis, loguru, launchd (M4).

**Conventions:** Run everything with `uv run`. Sleeve artifacts live under `data_dir/intraday/live/` (already covered by the `data/intraday/**` gitignore — they must never be committed). Commit after every task. Type hints + docstrings per the Charter. `uv run ruff check . && uv run mypy quant` must stay clean.

**Spec:** `docs/superpowers/specs/2026-06-07-intraday-live-loop-design.md`

---

### Task 1: Sleeve configuration

**Files:**
- Create: `quant/intraday/live/__init__.py`
- Create: `quant/intraday/live/config.py`
- Test: `tests/intraday/live/test_config.py`

- [ ] **Step 1: Create the package marker**

Create `quant/intraday/live/__init__.py` (empty file).

- [ ] **Step 2: Write the failing test**

Create `tests/intraday/live/__init__.py` (empty) and `tests/intraday/live/test_config.py`:

```python
import pytest

from quant.intraday.live.config import SleeveConfig


def test_defaults_are_tight_and_safe():
    c = SleeveConfig()
    assert c.universe == ("QQQ", "IWM", "DIA")
    assert c.notional_cap_pct == 0.10
    assert c.notional_cap_abs == 10_000.0
    assert c.per_trade_cap == 2_000.0
    assert c.max_round_trips == 20
    assert c.daily_loss_halt_pct == 0.015
    assert c.flat_by_close_minutes == 15
    assert c.tick_seconds == 60


def test_rejects_nonpositive_caps():
    with pytest.raises(ValueError):
        SleeveConfig(per_trade_cap=0.0)
    with pytest.raises(ValueError):
        SleeveConfig(notional_cap_abs=-1.0)


def test_sleeve_allocation_is_min_of_pct_and_abs():
    c = SleeveConfig(notional_cap_pct=0.10, notional_cap_abs=10_000.0)
    assert c.sleeve_allocation(equity=50_000.0) == 5_000.0   # pct binds
    assert c.sleeve_allocation(equity=200_000.0) == 10_000.0  # abs cap binds
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/intraday/live/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'quant.intraday.live.config'`

- [ ] **Step 4: Write minimal implementation**

Create `quant/intraday/live/config.py`:

```python
"""Configuration for the intraday live sleeve. All thresholds live here (no magic
numbers per the Charter); the 'tight & safe' profile is the default."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SleeveConfig:
    universe: tuple[str, ...] = ("QQQ", "IWM", "DIA")
    notional_cap_pct: float = 0.10        # fraction of paper equity
    notional_cap_abs: float = 10_000.0    # hard $ cap on sleeve notional
    per_trade_cap: float = 2_000.0        # max $ notional per single order
    max_round_trips: int = 20             # max opens per day
    daily_loss_halt_pct: float = 0.015    # of sleeve allocation -> auto-flatten+halt
    flat_by_close_minutes: int = 15       # flatten this many min before close
    tick_seconds: int = 60
    mean_reversion_lookback: int = 30     # ticks for the rolling mean/vol
    entry_z: float = 2.0                  # |z| beyond this -> fade
    exit_z: float = 0.5                   # revert inside this -> exit

    def __post_init__(self) -> None:
        for name in ("notional_cap_pct", "notional_cap_abs", "per_trade_cap"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if self.max_round_trips <= 0:
            raise ValueError("max_round_trips must be positive")

    def sleeve_allocation(self, equity: float) -> float:
        """Dollar capital the sleeve may deploy: min(pct of equity, absolute cap)."""
        return min(equity * self.notional_cap_pct, self.notional_cap_abs)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/intraday/live/test_config.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add quant/intraday/live/__init__.py quant/intraday/live/config.py tests/intraday/live/
git commit -m "feat(intraday-live): sleeve config with tight-and-safe defaults"
```

---

### Task 2: Sleeve ledger (internal position + P&L tracking)

**Files:**
- Create: `quant/intraday/live/sleeve.py`
- Test: `tests/intraday/live/test_sleeve.py`

The ledger tracks the sleeve's OWN positions and realized/unrealized P&L from its OWN fills — never the Alpaca aggregate. Average-cost accounting, supports long and short.

- [ ] **Step 1: Write the failing test**

Create `tests/intraday/live/test_sleeve.py`:

```python
from quant.intraday.live.sleeve import Fill, SleeveLedger


def test_long_round_trip_realized_pnl():
    led = SleeveLedger()
    led.record(Fill(symbol="QQQ", qty=10, price=100.0))   # buy 10 @100
    assert led.position("QQQ") == 10
    led.record(Fill(symbol="QQQ", qty=-10, price=105.0))  # sell 10 @105
    assert led.position("QQQ") == 0
    assert led.realized_pnl == 50.0   # (105-100)*10


def test_short_round_trip_realized_pnl():
    led = SleeveLedger()
    led.record(Fill(symbol="IWM", qty=-5, price=200.0))   # short 5 @200
    led.record(Fill(symbol="IWM", qty=5, price=190.0))    # cover 5 @190
    assert led.position("IWM") == 0
    assert led.realized_pnl == 50.0   # (200-190)*5


def test_unrealized_and_sleeve_value_from_marks():
    led = SleeveLedger()
    led.record(Fill(symbol="DIA", qty=4, price=300.0))
    marks = {"DIA": 310.0}
    assert led.unrealized_pnl(marks) == 40.0
    # gross notional exposure at marks
    assert led.gross_notional(marks) == 4 * 310.0


def test_day_pnl_is_realized_plus_unrealized():
    led = SleeveLedger()
    led.record(Fill(symbol="QQQ", qty=10, price=100.0))
    led.record(Fill(symbol="QQQ", qty=-4, price=110.0))   # realize (110-100)*4=40
    marks = {"QQQ": 120.0}                                 # 6 left, unreal (120-100)*6=120
    assert led.realized_pnl == 40.0
    assert led.unrealized_pnl(marks) == 120.0
    assert led.day_pnl(marks) == 160.0


def test_round_trips_counts_opens_only():
    led = SleeveLedger()
    led.record(Fill(symbol="QQQ", qty=10, price=100.0))   # open
    led.record(Fill(symbol="QQQ", qty=-10, price=101.0))  # close (not a new open)
    led.record(Fill(symbol="IWM", qty=-3, price=50.0))    # open short
    assert led.round_trips == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/intraday/live/test_sleeve.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'quant.intraday.live.sleeve'`

- [ ] **Step 3: Write minimal implementation**

Create `quant/intraday/live/sleeve.py`:

```python
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
        return {s: l.qty for s, l in self._lots.items() if l.qty != 0}

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/intraday/live/test_sleeve.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add quant/intraday/live/sleeve.py tests/intraday/live/test_sleeve.py
git commit -m "feat(intraday-live): sleeve ledger (avg-cost P&L, long/short, round-trip count)"
```

---

### Task 3: Sleeve-local halt artifact

**Files:**
- Create: `quant/intraday/live/halt.py`
- Test: `tests/intraday/live/test_halt.py`

A sleeve-scoped halt, separate from the global `quant/governance/halt.py`. Fail-closed (a corrupt artifact reads as halted). The loop checks BOTH this and the global halt; a sleeve halt never touches the global one.

- [ ] **Step 1: Write the failing test**

Create `tests/intraday/live/test_halt.py`:

```python
from datetime import UTC, datetime

from quant.intraday.live.halt import clear_sleeve_halt, load_sleeve_halt, set_sleeve_halt


def test_default_is_not_halted(tmp_path):
    st = load_sleeve_halt(tmp_path)
    assert st.active is False


def test_set_then_load_is_active(tmp_path):
    set_sleeve_halt(tmp_path, reason="daily loss breach",
                    created_at=datetime(2026, 6, 7, tzinfo=UTC))
    st = load_sleeve_halt(tmp_path)
    assert st.active is True
    assert "daily loss" in st.reason


def test_clear_then_load_is_inactive(tmp_path):
    set_sleeve_halt(tmp_path, reason="x")
    clear_sleeve_halt(tmp_path, reason="manual resume")
    assert load_sleeve_halt(tmp_path).active is False


def test_corrupt_artifact_fails_closed(tmp_path):
    path = tmp_path / "intraday" / "live" / "sleeve_halt.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json")
    assert load_sleeve_halt(tmp_path).active is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/intraday/live/test_halt.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'quant.intraday.live.halt'`

- [ ] **Step 3: Write minimal implementation**

Create `quant/intraday/live/halt.py`:

```python
"""Sleeve-scoped halt artifact. Distinct from quant.governance.halt (global). Stops
ONLY the intraday loop; the daily system is unaffected. Fail-closed on corruption."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class SleeveHaltState:
    active: bool
    reason: str
    updated_at: datetime


def sleeve_halt_path(data_dir: Path) -> Path:
    return data_dir / "intraday" / "live" / "sleeve_halt.json"


def load_sleeve_halt(data_dir: Path) -> SleeveHaltState:
    path = sleeve_halt_path(data_dir)
    if not path.exists():
        return SleeveHaltState(False, "not halted", datetime.fromtimestamp(0, UTC))
    try:
        obj = json.loads(path.read_text())
        if not isinstance(obj, dict):
            raise ValueError("sleeve halt artifact is not a JSON object")
        return SleeveHaltState(
            active=bool(obj["active"]),
            reason=str(obj["reason"]),
            updated_at=datetime.fromisoformat(obj["updated_at"]),
        )
    except (ValueError, KeyError, OSError) as exc:
        # Fail closed: an unreadable halt artifact must read as HALTED.
        return SleeveHaltState(True, f"corrupt sleeve halt artifact: {exc}", datetime.now(UTC))


def _write(data_dir: Path, state: SleeveHaltState) -> None:
    path = sleeve_halt_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "active": state.active,
        "reason": state.reason,
        "updated_at": state.updated_at.isoformat(),
    }))


def set_sleeve_halt(data_dir: Path, *, reason: str, created_at: datetime | None = None) -> SleeveHaltState:
    st = SleeveHaltState(True, reason, created_at or datetime.now(UTC))
    _write(data_dir, st)
    return st


def clear_sleeve_halt(data_dir: Path, *, reason: str) -> SleeveHaltState:
    st = SleeveHaltState(False, reason, datetime.now(UTC))
    _write(data_dir, st)
    return st
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/intraday/live/test_halt.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add quant/intraday/live/halt.py tests/intraday/live/test_halt.py
git commit -m "feat(intraday-live): sleeve-local halt artifact (fail-closed)"
```

---

### Task 4: Guardrails (pure decision functions)

**Files:**
- Create: `quant/intraday/live/guardrails.py`
- Test: `tests/intraday/live/test_guardrails.py`

Pure functions the loop composes: clamp an order to the per-trade and remaining-sleeve caps, detect trade-budget exhaustion, detect daily-loss breach, and detect the flat-by-close window.

- [ ] **Step 1: Write the failing test**

Create `tests/intraday/live/test_guardrails.py`:

```python
from datetime import UTC, datetime

from quant.intraday.live.config import SleeveConfig
from quant.intraday.live.guardrails import (
    clamp_qty_to_caps,
    daily_loss_breached,
    in_flat_window,
    trade_budget_exhausted,
)


def test_clamp_to_per_trade_cap():
    c = SleeveConfig(per_trade_cap=2_000.0)
    # desired 100 shares @ $100 = $10k; per-trade cap $2k -> 20 shares
    qty = clamp_qty_to_caps(desired_qty=100, price=100.0, gross_notional=0.0,
                            sleeve_allocation=10_000.0, config=c)
    assert qty == 20


def test_clamp_to_remaining_sleeve_room():
    c = SleeveConfig(per_trade_cap=5_000.0)
    # sleeve room left = 10k - 9k = 1k -> at $100 only 10 shares fit
    qty = clamp_qty_to_caps(desired_qty=40, price=100.0, gross_notional=9_000.0,
                            sleeve_allocation=10_000.0, config=c)
    assert qty == 10


def test_clamp_never_negative():
    c = SleeveConfig()
    qty = clamp_qty_to_caps(desired_qty=10, price=100.0, gross_notional=10_000.0,
                            sleeve_allocation=10_000.0, config=c)
    assert qty == 0


def test_trade_budget_exhausted():
    c = SleeveConfig(max_round_trips=20)
    assert trade_budget_exhausted(round_trips=20, config=c) is True
    assert trade_budget_exhausted(round_trips=19, config=c) is False


def test_daily_loss_breached():
    c = SleeveConfig(daily_loss_halt_pct=0.015)
    # allocation 10k, 1.5% = $150 loss threshold
    assert daily_loss_breached(day_pnl=-150.0, sleeve_allocation=10_000.0, config=c) is True
    assert daily_loss_breached(day_pnl=-149.0, sleeve_allocation=10_000.0, config=c) is False
    assert daily_loss_breached(day_pnl=500.0, sleeve_allocation=10_000.0, config=c) is False


def test_in_flat_window():
    c = SleeveConfig(flat_by_close_minutes=15)
    close = datetime(2026, 6, 8, 20, 0, tzinfo=UTC)  # 16:00 ET == 20:00 UTC
    assert in_flat_window(datetime(2026, 6, 8, 19, 46, tzinfo=UTC), close, c) is True
    assert in_flat_window(datetime(2026, 6, 8, 19, 44, tzinfo=UTC), close, c) is False


@given(
    desired=st.integers(min_value=0, max_value=100_000),
    price=st.floats(min_value=1.0, max_value=10_000.0, allow_nan=False),
    gross=st.floats(min_value=0.0, max_value=10_000.0, allow_nan=False),
)
def test_property_clamp_never_exceeds_caps(desired, price, gross):
    """Spec §7 invariant: a clamped order NEVER breaches the per-trade cap nor the
    remaining sleeve room, for ANY input."""
    c = SleeveConfig()
    qty = clamp_qty_to_caps(desired_qty=desired, price=price, gross_notional=gross,
                            sleeve_allocation=10_000.0, config=c)
    assert qty >= 0
    assert qty * price <= c.per_trade_cap + price          # within one share of cap
    assert qty * price <= max(0.0, 10_000.0 - gross) + price
    assert qty <= desired
```

Add these imports at the top of the test file:

```python
from hypothesis import given
from hypothesis import strategies as st
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/intraday/live/test_guardrails.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'quant.intraday.live.guardrails'`

- [ ] **Step 3: Write minimal implementation**

Create `quant/intraday/live/guardrails.py`:

```python
"""Pure guardrail decision functions. No I/O, no broker calls — the loop composes
these. Every function is independently testable."""

from __future__ import annotations

from datetime import datetime, timedelta

from quant.intraday.live.config import SleeveConfig


def clamp_qty_to_caps(
    *, desired_qty: int, price: float, gross_notional: float,
    sleeve_allocation: float, config: SleeveConfig,
) -> int:
    """Clamp |desired_qty| to BOTH the per-trade cap and the remaining sleeve room.
    Returns a non-negative share count (sign handled by the caller)."""
    if price <= 0 or desired_qty <= 0:
        return 0
    per_trade_shares = int(config.per_trade_cap // price)
    room_dollars = max(0.0, sleeve_allocation - gross_notional)
    room_shares = int(room_dollars // price)
    return max(0, min(desired_qty, per_trade_shares, room_shares))


def trade_budget_exhausted(*, round_trips: int, config: SleeveConfig) -> bool:
    return round_trips >= config.max_round_trips


def daily_loss_breached(*, day_pnl: float, sleeve_allocation: float, config: SleeveConfig) -> bool:
    threshold = -abs(config.daily_loss_halt_pct) * sleeve_allocation
    return day_pnl <= threshold


def in_flat_window(now: datetime, session_close: datetime, config: SleeveConfig) -> bool:
    return now >= session_close - timedelta(minutes=config.flat_by_close_minutes)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/intraday/live/test_guardrails.py -v`
Expected: PASS (7 tests, incl. the hypothesis property test)

- [ ] **Step 5: Commit**

```bash
git add quant/intraday/live/guardrails.py tests/intraday/live/test_guardrails.py
git commit -m "feat(intraday-live): pure guardrail functions (caps, budget, loss-halt, flat-window)"
```

---

### Task 5: Unique-COID order submission on AlpacaClient

**Files:**
- Modify: `quant/execution/alpaca.py` (add a method to `AlpacaClient`, ~after line 211)
- Test: `tests/execution/test_alpaca_simple_order.py`

The daily `submit_order` builds a deterministic COID per `(slug, symbol, date)` for idempotency — unusable for a loop placing many same-symbol orders per day. Add `submit_simple_order` that accepts a caller-supplied COID.

- [ ] **Step 1: Write the failing test**

Create `tests/execution/test_alpaca_simple_order.py`:

```python
from unittest.mock import MagicMock

from quant.execution.alpaca import AlpacaClient


def _client_with_fake_trading():
    c = AlpacaClient.__new__(AlpacaClient)  # bypass __init__ (no network)
    c._trading = MagicMock()
    return c


def test_submit_simple_order_market_uses_supplied_coid():
    c = _client_with_fake_trading()
    coid = c.submit_simple_order(symbol="QQQ", side="buy", qty=5,
                                 client_order_id="sleeve:QQQ:123:0")
    assert coid == "sleeve:QQQ:123:0"
    req = c._trading.submit_order.call_args.kwargs["order_data"] \
        if "order_data" in c._trading.submit_order.call_args.kwargs \
        else c._trading.submit_order.call_args.args[0]
    assert req.client_order_id == "sleeve:QQQ:123:0"
    assert req.qty == 5


def test_submit_simple_order_dry_run_skips_broker():
    c = _client_with_fake_trading()
    coid = c.submit_simple_order(symbol="IWM", side="sell", qty=3,
                                 client_order_id="sleeve:IWM:9:1", dry_run=True)
    assert coid == "sleeve:IWM:9:1"
    c._trading.submit_order.assert_not_called()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/execution/test_alpaca_simple_order.py -v`
Expected: FAIL — `AttributeError: 'AlpacaClient' object has no attribute 'submit_simple_order'`

- [ ] **Step 3: Write minimal implementation**

In `quant/execution/alpaca.py`, confirm these imports exist near the top (they are already used by `submit_order`): `MarketOrderRequest`, `LimitOrderRequest`, `AlpacaSide` (the alpaca-py `OrderSide`), and `TimeInForce`. If `TimeInForce` is not already imported, add `from alpaca.trading.enums import TimeInForce`. Then add this method to `AlpacaClient` (after `submit_order`, ~line 211):

```python
    def submit_simple_order(
        self,
        *,
        symbol: str,
        side: str,                       # "buy" | "sell"
        qty: int,
        client_order_id: str,
        order_type: str = "market",      # "market" | "limit"
        limit_price: float | None = None,
        dry_run: bool = False,
    ) -> str:
        """Submit a single intraday order with a CALLER-SUPPLIED client_order_id.

        Unlike submit_order (deterministic per-day COID for idempotent daily
        rebalances), the intraday loop places many same-symbol orders per day, so
        the COID must be unique per order — the caller owns uniqueness.
        Time-in-force is DAY. Returns the client_order_id.
        """
        if qty <= 0:
            raise ValueError(f"qty must be positive, got {qty}")
        alp_side = AlpacaSide.BUY if side == "buy" else AlpacaSide.SELL
        req: MarketOrderRequest | LimitOrderRequest
        if order_type == "limit":
            if limit_price is None:
                raise ValueError("limit order requires limit_price")
            req = LimitOrderRequest(
                symbol=symbol, qty=qty, side=alp_side,
                time_in_force=TimeInForce.DAY,
                client_order_id=client_order_id, limit_price=limit_price,
            )
        else:
            req = MarketOrderRequest(
                symbol=symbol, qty=qty, side=alp_side,
                time_in_force=TimeInForce.DAY,
                client_order_id=client_order_id,
            )
        if dry_run:
            logger.info("[DRY-RUN] sleeve would submit {} {} {} (coid={})",
                        side, qty, symbol, client_order_id)
            return client_order_id
        self._trading.submit_order(req)
        logger.info("Sleeve submitted {} {} {} (coid={})", side, qty, symbol, client_order_id)
        return client_order_id
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/execution/test_alpaca_simple_order.py -v`
Expected: PASS (2 tests). Also run `uv run mypy quant/execution/alpaca.py` — expect clean.

- [ ] **Step 5: Commit**

```bash
git add quant/execution/alpaca.py tests/execution/test_alpaca_simple_order.py
git commit -m "feat(execution): submit_simple_order with caller-supplied COID for intraday loop"
```

---

### Task 6: Sleeve COID generation

**Files:**
- Create: `quant/intraday/live/ids.py`
- Test: `tests/intraday/live/test_ids.py`

- [ ] **Step 1: Write the failing test**

Create `tests/intraday/live/test_ids.py`:

```python
from datetime import UTC, datetime

from quant.intraday.live.ids import make_sleeve_coid


def test_coid_is_namespaced_and_unique_per_seq():
    ts = datetime(2026, 6, 8, 14, 30, 0, tzinfo=UTC)
    a = make_sleeve_coid("QQQ", ts, 0)
    b = make_sleeve_coid("QQQ", ts, 1)
    assert a.startswith("sleeve:QQQ:")
    assert a != b


def test_coid_differs_by_symbol_and_time():
    ts1 = datetime(2026, 6, 8, 14, 30, 0, tzinfo=UTC)
    ts2 = datetime(2026, 6, 8, 14, 31, 0, tzinfo=UTC)
    assert make_sleeve_coid("QQQ", ts1, 0) != make_sleeve_coid("IWM", ts1, 0)
    assert make_sleeve_coid("QQQ", ts1, 0) != make_sleeve_coid("QQQ", ts2, 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/intraday/live/test_ids.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'quant.intraday.live.ids'`

- [ ] **Step 3: Write minimal implementation**

Create `quant/intraday/live/ids.py`:

```python
"""Unique client_order_id generation for the intraday sleeve. Namespaced 'sleeve:'
so attribution and reconciliation can isolate sleeve orders from daily-system ones."""

from __future__ import annotations

from datetime import datetime


def make_sleeve_coid(symbol: str, ts: datetime, seq: int) -> str:
    """sleeve:<SYMBOL>:<epoch-seconds>:<seq> — unique per (symbol, tick, order index)."""
    return f"sleeve:{symbol}:{int(ts.timestamp())}:{seq}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/intraday/live/test_ids.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add quant/intraday/live/ids.py tests/intraday/live/test_ids.py
git commit -m "feat(intraday-live): namespaced unique sleeve COID generator"
```

---

### Task 7: Live quote feed

**Files:**
- Create: `quant/intraday/live/feed.py`
- Test: `tests/intraday/live/test_feed.py`

Wraps alpaca-py `StockHistoricalDataClient.get_stock_latest_quote` to produce `QuoteBar` events for the sleeve universe each tick. The broker client is injected so tests use a fake. On failure it raises `FeedError`; the loop decides to skip new actions (never blind-trades on stale data).

- [ ] **Step 1: Write the failing test**

Create `tests/intraday/live/test_feed.py`:

```python
import pytest

from quant.intraday.data.events import QuoteBar
from quant.intraday.live.feed import FeedError, LiveQuoteFeed


class _FakeQuote:
    def __init__(self, bid, ask, bs=100, as_=100):
        self.bid_price, self.ask_price = bid, ask
        self.bid_size, self.ask_size = bs, as_


class _FakeDataClient:
    def __init__(self, mapping, raise_exc=None):
        self._mapping, self._raise = mapping, raise_exc

    def get_stock_latest_quote(self, request):
        if self._raise:
            raise self._raise
        return self._mapping


def test_latest_quotes_returns_quotebars():
    client = _FakeDataClient({"QQQ": _FakeQuote(100.0, 100.2)})
    feed = LiveQuoteFeed(symbols=["QQQ"], data_client=client)
    bars = feed.latest_quotes()
    assert len(bars) == 1
    qb = bars[0]
    assert isinstance(qb, QuoteBar)
    assert qb.symbol == "QQQ"
    assert qb.bid == 100.0 and qb.ask == 100.2


def test_feed_error_on_client_exception():
    client = _FakeDataClient({}, raise_exc=RuntimeError("network down"))
    feed = LiveQuoteFeed(symbols=["QQQ"], data_client=client)
    with pytest.raises(FeedError):
        feed.latest_quotes()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/intraday/live/test_feed.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'quant.intraday.live.feed'`

- [ ] **Step 3: Write minimal implementation**

Create `quant/intraday/live/feed.py`:

```python
"""Live intraday quote feed: REST-polls Alpaca latest NBBO quotes (sufficient at a
60s cadence) and emits QuoteBar events. The broker data client is injected so the
loop and tests can substitute a fake; real construction is via from_settings()."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Protocol

from quant.intraday.data.events import QuoteBar


class FeedError(RuntimeError):
    """Raised when the upstream quote source fails. The loop must NOT trade on stale
    data — it skips new actions for the tick and retries next tick."""


class _DataClient(Protocol):
    def get_stock_latest_quote(self, request: Any) -> dict[str, Any]: ...


class LiveQuoteFeed:
    def __init__(self, *, symbols: list[str], data_client: _DataClient) -> None:
        self._symbols = symbols
        self._client = data_client

    @classmethod
    def from_settings(cls, *, symbols: list[str], settings: Any = None) -> "LiveQuoteFeed":
        from alpaca.data.historical import StockHistoricalDataClient
        from quant.util.config import Settings
        s = settings or Settings()  # type: ignore[call-arg]
        client = StockHistoricalDataClient(api_key=s.alpaca_api_key, secret_key=s.alpaca_secret_key)
        return cls(symbols=symbols, data_client=client)

    def latest_quotes(self, now: datetime | None = None) -> list[QuoteBar]:
        from alpaca.data.requests import StockLatestQuoteRequest
        ts = now or datetime.now(UTC)
        try:
            req = StockLatestQuoteRequest(symbol_or_symbols=self._symbols)
            raw = self._client.get_stock_latest_quote(req)
        except Exception as exc:  # noqa: BLE001 - normalize any upstream error
            raise FeedError(str(exc)) from exc
        bars: list[QuoteBar] = []
        for sym, q in raw.items():
            bars.append(QuoteBar(
                ts=ts, symbol=sym,
                bid=float(q.bid_price), ask=float(q.ask_price),
                bid_size=int(q.bid_size), ask_size=int(q.ask_size),
            ))
        return bars
```

Note: `StockLatestQuoteRequest` accepts `symbol_or_symbols`; the test's `_FakeDataClient.get_stock_latest_quote` ignores the request object and returns the mapping, so the import inside the method is exercised only at runtime — keep the import inside the method so the test never needs alpaca-py installed differently than production.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/intraday/live/test_feed.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add quant/intraday/live/feed.py tests/intraday/live/test_feed.py
git commit -m "feat(intraday-live): live NBBO quote feed (REST poll, injected client, FeedError)"
```

---

### Task 8: Intraday mean-reversion strategy

**Files:**
- Create: `quant/intraday/live/strategy.py`
- Test: `tests/intraday/live/test_strategy.py`

Implements the existing `IntradayStrategy` protocol (`on_event(event, ctx) -> list[Order]`) from `quant.intraday.strategy`. Maintains a rolling mid-price window per symbol; when the mid deviates beyond `entry_z` standard deviations, fade (short the rich / buy the cheap); exit when it reverts inside `exit_z`. Targets are expressed as orders sized to a fixed per-name share unit (the loop applies caps afterward).

- [ ] **Step 1: Write the failing test**

Create `tests/intraday/live/test_strategy.py`:

```python
from datetime import UTC, datetime

from quant.intraday.data.events import QuoteBar
from quant.intraday.live.config import SleeveConfig
from quant.intraday.live.strategy import MeanReversionStrategy
from quant.intraday.strategy import Order, OrderType, Side


class _Ctx:
    def __init__(self, pos=0):
        self._pos = pos
    def position(self, symbol): return self._pos
    def cash(self): return 0.0
    def nbbo(self, symbol): return None
    def now(self): return datetime(2026, 6, 8, 15, 0, tzinfo=UTC)


def _qb(price, t):
    return QuoteBar(ts=datetime(2026, 6, 8, 15, t, tzinfo=UTC), symbol="QQQ",
                    bid=price - 0.01, ask=price + 0.01, bid_size=100, ask_size=100)


def test_no_order_before_window_full():
    cfg = SleeveConfig(mean_reversion_lookback=5)
    strat = MeanReversionStrategy(cfg, unit_shares=10)
    ctx = _Ctx()
    orders = []
    for i in range(4):
        orders = strat.on_event(_qb(100.0, i), ctx)
    assert orders == []  # window not yet full


def test_fades_upward_deviation_with_short():
    cfg = SleeveConfig(mean_reversion_lookback=5, entry_z=2.0)
    strat = MeanReversionStrategy(cfg, unit_shares=10)
    ctx = _Ctx(pos=0)
    for i in range(5):
        strat.on_event(_qb(100.0, i), ctx)       # flat history, ~zero vol -> spike z huge
    orders = strat.on_event(_qb(101.0, 5), ctx)  # jump up -> fade = SELL
    assert len(orders) == 1
    o = orders[0]
    assert isinstance(o, Order) and o.side is Side.SELL and o.symbol == "QQQ"
    assert o.type is OrderType.MARKET and o.qty == 10


def test_exits_when_reverted_inside_exit_band():
    cfg = SleeveConfig(mean_reversion_lookback=5, entry_z=2.0, exit_z=0.5)
    strat = MeanReversionStrategy(cfg, unit_shares=10)
    ctx = _Ctx(pos=-10)  # already short from a prior fade
    for i in range(5):
        strat.on_event(_qb(100.0, i), ctx)
    orders = strat.on_event(_qb(100.0, 5), ctx)  # back at mean -> exit short = BUY 10
    assert len(orders) == 1 and orders[0].side is Side.BUY and orders[0].qty == 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/intraday/live/test_strategy.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'quant.intraday.live.strategy'`

- [ ] **Step 3: Write minimal implementation**

Create `quant/intraday/live/strategy.py`:

```python
"""Intraday mean-reversion proof-of-life strategy. Implements the shared
IntradayStrategy protocol so it could also be driven by the existing simulator.
Economic rationale: very short-horizon mean reversion in liquid index ETFs from
microstructure noise / liquidity provision. Assumptions: no persistent intraday
drift over the lookback. How it fails: trends/news regimes (it fades the move) and
spread/slippage eating the small edge — which is why the loop also flattens by close
and the sleeve is tightly capped."""

from __future__ import annotations

import statistics
from collections import defaultdict, deque

from quant.intraday.data.events import Bar, Event, QuoteBar
from quant.intraday.live.config import SleeveConfig
from quant.intraday.strategy import IntradayStrategy, Order, OrderType, Side, StrategyContext


class MeanReversionStrategy:
    """z-score fade on a rolling window of mids. Reusable under IntradayStrategy."""

    def __init__(self, config: SleeveConfig, *, unit_shares: int = 10) -> None:
        self._cfg = config
        self._unit = unit_shares
        self._mids: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=config.mean_reversion_lookback)
        )

    def on_event(self, event: Event, ctx: StrategyContext) -> list[Order]:
        if not isinstance(event, QuoteBar):
            return []  # this strategy trades off NBBO mids only
        sym = event.symbol
        window = self._mids[sym]
        window.append(event.mid)
        if len(window) < self._cfg.mean_reversion_lookback:
            return []
        mu = statistics.fmean(window)
        sd = statistics.pstdev(window)
        if sd == 0.0:
            z = 0.0 if event.mid == mu else (1.0 if event.mid > mu else -1.0) * 1e9
        else:
            z = (event.mid - mu) / sd
        pos = ctx.position(sym)
        # Exit first: if we hold and have reverted inside the exit band, flatten.
        if pos != 0 and abs(z) <= self._cfg.exit_z:
            side = Side.BUY if pos < 0 else Side.SELL
            return [Order(symbol=sym, side=side, qty=abs(pos), type=OrderType.MARKET)]
        # Entry: fade a large deviation (only if flat).
        if pos == 0 and abs(z) >= self._cfg.entry_z:
            side = Side.SELL if z > 0 else Side.BUY
            return [Order(symbol=sym, side=side, qty=self._unit, type=OrderType.MARKET)]
        return []


# Structural check: MeanReversionStrategy satisfies the protocol.
_: type[IntradayStrategy] = MeanReversionStrategy  # type: ignore[assignment]
```

Note: the final `_:` line is a static-typing assertion that the class satisfies `IntradayStrategy`; if mypy objects to the assignment form, replace it with `assert isinstance(MeanReversionStrategy(SleeveConfig()), IntradayStrategy)` inside the test instead.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/intraday/live/test_strategy.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add quant/intraday/live/strategy.py tests/intraday/live/test_strategy.py
git commit -m "feat(intraday-live): mean-reversion proof-of-life strategy (IntradayStrategy protocol)"
```

---

### Task 9: Tick journal

**Files:**
- Create: `quant/intraday/live/journal.py`
- Test: `tests/intraday/live/test_journal.py`

Append-only record of each tick's outcome under `data_dir/intraday/live/` (gitignored). Used by `status` and by the drift comparison.

- [ ] **Step 1: Write the failing test**

Create `tests/intraday/live/test_journal.py`:

```python
from datetime import UTC, datetime

from quant.intraday.live.journal import TickRecord, append_tick, read_ticks


def test_append_then_read_roundtrips(tmp_path):
    rec = TickRecord(
        ts=datetime(2026, 6, 8, 15, 0, tzinfo=UTC),
        sleeve_value=1234.5, day_pnl=12.3, round_trips=2,
        n_orders=1, halted=False, note="ok",
    )
    append_tick(tmp_path, rec)
    df = read_ticks(tmp_path)
    assert len(df) == 1
    assert df.iloc[0]["day_pnl"] == 12.3
    assert df.iloc[0]["n_orders"] == 1


def test_append_is_cumulative(tmp_path):
    for i in range(3):
        append_tick(tmp_path, TickRecord(
            ts=datetime(2026, 6, 8, 15, i, tzinfo=UTC),
            sleeve_value=0.0, day_pnl=float(i), round_trips=0,
            n_orders=0, halted=False, note=""))
    assert len(read_ticks(tmp_path)) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/intraday/live/test_journal.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'quant.intraday.live.journal'`

- [ ] **Step 3: Write minimal implementation**

Create `quant/intraday/live/journal.py`:

```python
"""Append-only tick journal for the intraday sleeve. Written under
data_dir/intraday/live/ (gitignored); the source of truth for status + drift."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

_COLS = ["ts", "sleeve_value", "day_pnl", "round_trips", "n_orders", "halted", "note"]


@dataclass(frozen=True)
class TickRecord:
    ts: datetime
    sleeve_value: float
    day_pnl: float
    round_trips: int
    n_orders: int
    halted: bool
    note: str


def _journal_path(data_dir: Path) -> Path:
    return data_dir / "intraday" / "live" / "ticks.parquet"


def append_tick(data_dir: Path, rec: TickRecord) -> None:
    path = _journal_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([asdict(rec)], columns=_COLS)
    if path.exists():
        existing = pd.read_parquet(path)
        row = pd.concat([existing, row], ignore_index=True)
    row.to_parquet(path, index=False)


def read_ticks(data_dir: Path) -> pd.DataFrame:
    path = _journal_path(data_dir)
    if not path.exists():
        return pd.DataFrame(columns=_COLS)
    return pd.read_parquet(path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/intraday/live/test_journal.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add quant/intraday/live/journal.py tests/intraday/live/test_journal.py
git commit -m "feat(intraday-live): append-only tick journal"
```

---

### Task 10: The tick loop (orchestration)

**Files:**
- Create: `quant/intraday/live/loop.py`
- Test: `tests/intraday/live/test_loop.py`

`run_tick` performs ONE lifecycle pass (the heart of the spine); it takes injected collaborators (feed, broker, ledger, clocks) so it is fully testable without network or sleep. `run_loop` is the thin driver that calls `run_tick` every `tick_seconds`.

The broker is any object exposing `submit_simple_order(...)` and `account()` — `AlpacaClient` satisfies this. A "fill" is modeled at the quote mid for ledger purposes on this tick (the real fill price is reconciled later via order status; for the spine's internal P&L we mark at submission mid, which the integration test and drift comparison treat as provisional).

- [ ] **Step 1: Write the failing test**

Create `tests/intraday/live/test_loop.py`:

```python
from datetime import UTC, datetime

from quant.intraday.data.events import QuoteBar
from quant.intraday.live.config import SleeveConfig
from quant.intraday.live.loop import TickDeps, run_tick
from quant.intraday.live.sleeve import SleeveLedger


class _Broker:
    def __init__(self):
        self.orders = []
    def account(self):
        class A: equity = 100_000.0
        return A()
    def submit_simple_order(self, *, symbol, side, qty, client_order_id,
                            order_type="market", limit_price=None, dry_run=False):
        self.orders.append((symbol, side, qty))
        return client_order_id


class _Feed:
    def __init__(self, bars): self._bars = bars
    def latest_quotes(self, now=None): return self._bars


class _Strat:
    """Stub strategy: emits a fixed order list regardless of input."""
    def __init__(self, orders): self._orders = orders
    def on_event(self, event, ctx): return self._orders


def _deps(tmp_path, broker, feed, strat, ledger, now, close):
    return TickDeps(
        data_dir=tmp_path, config=SleeveConfig(), broker=broker, feed=feed,
        strategy=strat, ledger=ledger, now=now, session_open=True, session_close=close,
    )


def _qb(sym, price):
    return QuoteBar(ts=datetime(2026, 6, 8, 15, 0, tzinfo=UTC), symbol=sym,
                    bid=price - 0.01, ask=price + 0.01, bid_size=100, ask_size=100)


def test_session_closed_does_nothing(tmp_path):
    from quant.intraday.strategy import Order, Side
    broker, ledger = _Broker(), SleeveLedger()
    deps = _deps(tmp_path, broker, _Feed([_qb("QQQ", 100.0)]),
                 _Strat([Order("QQQ", Side.BUY, 10)]), ledger,
                 now=datetime(2026, 6, 8, 9, 0, tzinfo=UTC),
                 close=datetime(2026, 6, 8, 20, 0, tzinfo=UTC))
    deps = deps.__class__(**{**deps.__dict__, "session_open": False})
    run_tick(deps)
    assert broker.orders == []


def test_happy_path_submits_capped_order(tmp_path):
    from quant.intraday.strategy import Order, Side
    broker, ledger = _Broker(), SleeveLedger()
    # desired 1000 shares @ ~$100 -> per-trade cap $2k -> clamped to 20
    deps = _deps(tmp_path, broker, _Feed([_qb("QQQ", 100.0)]),
                 _Strat([Order("QQQ", Side.BUY, 1000)]), ledger,
                 now=datetime(2026, 6, 8, 15, 0, tzinfo=UTC),
                 close=datetime(2026, 6, 8, 20, 0, tzinfo=UTC))
    run_tick(deps)
    assert len(broker.orders) == 1
    sym, side, qty = broker.orders[0]
    assert sym == "QQQ" and side == "buy" and qty == 20
    assert ledger.position("QQQ") == 20


def test_daily_loss_breach_flattens_and_halts(tmp_path):
    from quant.intraday.live.halt import load_sleeve_halt
    from quant.intraday.strategy import Order, Side
    broker, ledger = _Broker(), SleeveLedger()
    # Seed a losing long: bought 100 @ $100, now marked far lower.
    from quant.intraday.live.sleeve import Fill
    ledger.record(Fill("QQQ", 100, 100.0))
    deps = _deps(tmp_path, broker, _Feed([_qb("QQQ", 50.0)]),   # huge mark loss
                 _Strat([Order("QQQ", Side.BUY, 10)]), ledger,
                 now=datetime(2026, 6, 8, 15, 0, tzinfo=UTC),
                 close=datetime(2026, 6, 8, 20, 0, tzinfo=UTC))
    run_tick(deps)
    # It must flatten (sell 100), NOT open new, and set the sleeve halt.
    assert ("QQQ", "sell", 100) in broker.orders
    assert ledger.position("QQQ") == 0
    assert load_sleeve_halt(tmp_path).active is True


def test_flat_by_close_flattens_and_skips_entries(tmp_path):
    from quant.intraday.live.sleeve import Fill
    from quant.intraday.strategy import Order, Side
    broker, ledger = _Broker(), SleeveLedger()
    ledger.record(Fill("QQQ", 10, 100.0))
    deps = _deps(tmp_path, broker, _Feed([_qb("QQQ", 100.0)]),
                 _Strat([Order("QQQ", Side.BUY, 10)]), ledger,
                 now=datetime(2026, 6, 8, 19, 50, tzinfo=UTC),  # inside 15-min window
                 close=datetime(2026, 6, 8, 20, 0, tzinfo=UTC))
    run_tick(deps)
    assert ("QQQ", "sell", 10) in broker.orders
    assert ledger.position("QQQ") == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/intraday/live/test_loop.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'quant.intraday.live.loop'`

- [ ] **Step 3: Write minimal implementation**

Create `quant/intraday/live/loop.py`:

```python
"""The intraday tick loop — the spine. run_tick() performs ONE lifecycle pass with
injected collaborators (no network, no sleep) so it is fully unit-testable. run_loop()
is the driver that ticks every config.tick_seconds during the session.

Lifecycle per tick (spec section 3):
  session-check -> guardrails-first (halt/loss/flat) -> pull quotes -> strategy
  -> size+caps -> reconcile vs ledger -> submit (unique COID) -> journal.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from loguru import logger

from quant.governance.halt import load_halt
from quant.intraday.data.events import QuoteBar
from quant.intraday.live.config import SleeveConfig
from quant.intraday.live.feed import FeedError
from quant.intraday.live.guardrails import (
    clamp_qty_to_caps,
    daily_loss_breached,
    in_flat_window,
    trade_budget_exhausted,
)
from quant.intraday.live.halt import load_sleeve_halt, set_sleeve_halt
from quant.intraday.live.ids import make_sleeve_coid
from quant.intraday.live.journal import TickRecord, append_tick
from quant.intraday.live.sleeve import Fill, SleeveLedger
from quant.intraday.strategy import IntradayStrategy, Order, Side


class _Broker(Protocol):
    def account(self) -> Any: ...
    def submit_simple_order(self, *, symbol: str, side: str, qty: int,
                            client_order_id: str, order_type: str = "market",
                            limit_price: float | None = None, dry_run: bool = False) -> str: ...


class _Feed(Protocol):
    def latest_quotes(self, now: datetime | None = None) -> list[QuoteBar]: ...


@dataclass
class TickDeps:
    data_dir: Path
    config: SleeveConfig
    broker: _Broker
    feed: _Feed
    strategy: IntradayStrategy
    ledger: SleeveLedger
    now: datetime
    session_open: bool
    session_close: datetime
    dry_run: bool = False


class _Ctx:
    """StrategyContext backed by the ledger + this tick's marks."""
    def __init__(self, ledger: SleeveLedger, marks: dict[str, float], now: datetime) -> None:
        self._ledger, self._marks, self._now = ledger, marks, now
    def position(self, symbol: str) -> int: return self._ledger.position(symbol)
    def cash(self) -> float: return 0.0
    def nbbo(self, symbol: str) -> QuoteBar | None: return None
    def now(self) -> datetime: return self._now


def _flatten_all(deps: TickDeps, marks: dict[str, float]) -> int:
    """Submit market orders closing every open sleeve position. Returns order count."""
    n = 0
    for sym, pos in list(deps.ledger.positions().items()):
        side = "sell" if pos > 0 else "buy"
        coid = make_sleeve_coid(sym, deps.now, n)
        deps.broker.submit_simple_order(symbol=sym, side=side, qty=abs(pos),
                                        client_order_id=coid, dry_run=deps.dry_run)
        signed = -pos  # closing fill is opposite sign of position
        deps.ledger.record(Fill(symbol=sym, qty=signed, price=marks.get(sym, 0.0)))
        n += 1
    return n


def run_tick(deps: TickDeps) -> None:
    cfg = deps.config
    # 1. Session check.
    if not deps.session_open:
        logger.debug("intraday loop idle (session closed)")
        return

    # 2a. Global halt (governance) — also stops the sleeve.
    if load_halt(deps.data_dir).active or load_sleeve_halt(deps.data_dir).active:
        logger.warning("intraday loop halted; skipping tick")
        return

    # 3. Pull quotes (never trade on stale data).
    try:
        bars = deps.feed.latest_quotes(deps.now)
    except FeedError as exc:
        logger.warning("feed error, skipping tick actions: {}", exc)
        return
    marks = {b.symbol: b.mid for b in bars}

    allocation = cfg.sleeve_allocation(float(deps.broker.account().equity))
    day_pnl = deps.ledger.day_pnl(marks)

    # 2b. Loss kill-switch — flatten + halt BEFORE any new action.
    if daily_loss_breached(day_pnl=day_pnl, sleeve_allocation=allocation, config=cfg):
        n = _flatten_all(deps, marks)
        set_sleeve_halt(deps.data_dir, reason=f"daily loss breach: day_pnl={day_pnl:.2f}")
        _journal(deps, marks, n_orders=n, halted=True, note="loss-halt")
        return

    # 2c. Flat-by-close — flatten and stop opening.
    if in_flat_window(deps.now, deps.session_close, cfg):
        n = _flatten_all(deps, marks)
        _journal(deps, marks, n_orders=n, halted=False, note="flat-by-close")
        return

    # 4. Strategy targets (feed each quote as an event).
    ctx = _Ctx(deps.ledger, marks, deps.now)
    desired: list[Order] = []
    for b in bars:
        desired.extend(deps.strategy.on_event(b, ctx))

    # 5-7. Size, cap, reconcile, submit.
    n_orders = 0
    for order in desired:
        is_open = deps.ledger.position(order.symbol) == 0
        if is_open and trade_budget_exhausted(round_trips=deps.ledger.round_trips, config=cfg):
            continue  # budget only blocks NEW opens; exits always allowed
        price = marks.get(order.symbol)
        if price is None:
            continue
        qty = clamp_qty_to_caps(
            desired_qty=order.qty, price=price,
            gross_notional=deps.ledger.gross_notional(marks),
            sleeve_allocation=allocation, config=cfg,
        )
        if qty <= 0:
            continue
        side = "buy" if order.side is Side.BUY else "sell"
        coid = make_sleeve_coid(order.symbol, deps.now, n_orders)
        deps.broker.submit_simple_order(symbol=order.symbol, side=side, qty=qty,
                                        client_order_id=coid, dry_run=deps.dry_run)
        signed = qty if order.side is Side.BUY else -qty
        deps.ledger.record(Fill(symbol=order.symbol, qty=signed, price=price))
        n_orders += 1

    _journal(deps, marks, n_orders=n_orders, halted=False, note="ok")


def _journal(deps: TickDeps, marks: dict[str, float], *, n_orders: int,
             halted: bool, note: str) -> None:
    append_tick(deps.data_dir, TickRecord(
        ts=deps.now,
        sleeve_value=deps.ledger.gross_notional(marks),
        day_pnl=deps.ledger.day_pnl(marks),
        round_trips=deps.ledger.round_trips,
        n_orders=n_orders, halted=halted, note=note,
    ))


def run_loop(deps_factory: Any, *, max_ticks: int | None = None, sleep_s: float | None = None) -> None:
    """Driver: build fresh TickDeps each tick via deps_factory() and run_tick().
    deps_factory is a zero-arg callable returning a TickDeps for 'now'. max_ticks
    bounds runs (None = forever); sleep_s overrides config cadence (tests pass 0)."""
    count = 0
    while max_ticks is None or count < max_ticks:
        deps = deps_factory()
        try:
            run_tick(deps)
        except Exception:  # noqa: BLE001 - a loop must survive a single bad tick
            logger.exception("intraday tick failed; continuing")
        count += 1
        time.sleep(sleep_s if sleep_s is not None else deps.config.tick_seconds)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/intraday/live/test_loop.py -v`
Expected: PASS (4 tests). Then `uv run mypy quant/intraday/live/loop.py` — expect clean.

- [ ] **Step 5: Commit**

```bash
git add quant/intraday/live/loop.py tests/intraday/live/test_loop.py
git commit -m "feat(intraday-live): tick loop orchestration (guardrails-first lifecycle)"
```

---

### Task 11: Session helper

**Files:**
- Create: `quant/intraday/live/session.py`
- Test: `tests/intraday/live/test_session.py`

Resolves "is the equities session open right now?" and "when does it close today?" from the existing trading calendar, so `run_loop`'s factory can populate `TickDeps.session_open`/`session_close`.

The existing calendar lives at `quant/util/trading_calendar.py` and exposes
`is_trading_day(d: date) -> bool` and `is_early_close(d: date) -> bool` (13:00 ET
early-close days). Reuse them — do NOT invent a new calendar.

- [ ] **Step 1: Write the failing test**

Create `tests/intraday/live/test_session.py`:

```python
from datetime import UTC, datetime

from quant.intraday.live.session import session_state


def test_weekend_is_closed():
    # 2026-06-06 is a Saturday
    st = session_state(datetime(2026, 6, 6, 15, 0, tzinfo=UTC))
    assert st.open is False


def test_weekday_midsession_is_open():
    # 2026-06-08 Monday, 15:00 UTC == 11:00 ET (RTH)
    st = session_state(datetime(2026, 6, 8, 15, 0, tzinfo=UTC))
    assert st.open is True
    assert st.close.tzinfo is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/intraday/live/test_session.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'quant.intraday.live.session'`

- [ ] **Step 3: Write minimal implementation**

Create `quant/intraday/live/session.py`:

```python
"""Equities-session resolver for the intraday loop. RTH 09:30-16:00 ET, with NYSE
early closes (13:00 ET) honored via the existing trading calendar."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from zoneinfo import ZoneInfo

from quant.util import trading_calendar as cal

_ET = ZoneInfo("America/New_York")
_OPEN = time(9, 30)
_CLOSE = time(16, 0)
_EARLY_CLOSE = time(13, 0)


@dataclass(frozen=True)
class SessionState:
    open: bool
    close: datetime  # today's RTH close (tz-aware, in `now`'s tz)


def session_state(now: datetime) -> SessionState:
    et = now.astimezone(_ET)
    close_time = _EARLY_CLOSE if cal.is_early_close(et.date()) else _CLOSE
    close_et = datetime.combine(et.date(), close_time, tzinfo=_ET)
    is_open = cal.is_trading_day(et.date()) and (_OPEN <= et.time() < close_time)
    return SessionState(open=is_open, close=close_et.astimezone(now.tzinfo or _ET))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/intraday/live/test_session.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add quant/intraday/live/session.py tests/intraday/live/test_session.py
git commit -m "feat(intraday-live): RTH session resolver from trading calendar"
```

---

### Task 12: CLI subgroup (`quant intraday live ...`)

**Files:**
- Modify: `quant/intraday/cli.py` (add a `live` group + commands)
- Test: `tests/intraday/live/test_cli.py`

Commands: `run` (start the loop), `status` (last tick + halt state), `halt` / `resume` (sleeve kill-switch), `flat` (manual flatten). `status`, `halt`, `resume` are testable without network; `run`/`flat` touch the broker so the test only checks they exist and `--help` works.

- [ ] **Step 1: Write the failing test**

Create `tests/intraday/live/test_cli.py`:

```python
from click.testing import CliRunner

from quant.intraday.cli import intraday


def test_live_group_exists():
    r = CliRunner().invoke(intraday, ["live", "--help"])
    assert r.exit_code == 0
    for cmd in ("run", "status", "halt", "resume", "flat"):
        assert cmd in r.output


def test_halt_then_status_then_resume(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    runner = CliRunner()
    assert runner.invoke(intraday, ["live", "halt", "--reason", "test"]).exit_code == 0
    out = runner.invoke(intraday, ["live", "status"]).output
    assert "HALTED" in out.upper()
    assert runner.invoke(intraday, ["live", "resume", "--reason", "ok"]).exit_code == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/intraday/live/test_cli.py -v`
Expected: FAIL — `live` is not a command of `intraday`.

- [ ] **Step 3: Write minimal implementation**

In `quant/intraday/cli.py`, add (after the existing `data` group):

```python
from datetime import UTC, datetime
from pathlib import Path

import click

from quant.intraday.live.halt import clear_sleeve_halt, load_sleeve_halt, set_sleeve_halt
from quant.intraday.live.journal import read_ticks
from quant.util.config import Settings


@intraday.group()
def live() -> None:
    """Intraday live-loop (sleeve) commands."""


def _data_dir() -> Path:
    return Settings().data_dir  # type: ignore[call-arg]


@live.command()
def status() -> None:
    """Show sleeve halt state and the last journaled tick."""
    dd = _data_dir()
    halt = load_sleeve_halt(dd)
    click.echo(f"sleeve halt: {'HALTED — ' + halt.reason if halt.active else 'active (not halted)'}")
    df = read_ticks(dd)
    if df.empty:
        click.echo("no ticks journaled yet")
        return
    last = df.iloc[-1]
    click.echo(f"last tick {last['ts']}: day_pnl={last['day_pnl']:.2f} "
               f"round_trips={last['round_trips']} n_orders={last['n_orders']}")


@live.command()
@click.option("--reason", required=True)
def halt(reason: str) -> None:
    """Halt the sleeve (stops the intraday loop only; daily system unaffected)."""
    set_sleeve_halt(_data_dir(), reason=reason, created_at=datetime.now(UTC))
    click.echo(f"sleeve halted: {reason}")


@live.command()
@click.option("--reason", required=True)
def resume(reason: str) -> None:
    """Clear the sleeve halt."""
    clear_sleeve_halt(_data_dir(), reason=reason)
    click.echo(f"sleeve resumed: {reason}")


@live.command()
def flat() -> None:
    """Manually flatten all sleeve positions (placeholder until wired to broker)."""
    click.echo("flatten requested — run via `live run` daemon path in production")


@live.command()
@click.option("--max-ticks", type=int, default=None, help="bound the run (default: forever)")
@click.option("--dry-run", is_flag=True, help="log orders without submitting")
def run(max_ticks: int | None, dry_run: bool) -> None:
    """Start the intraday tick loop. Wires real broker + feed + strategy."""
    from quant.execution.alpaca import AlpacaClient
    from quant.intraday.live.config import SleeveConfig
    from quant.intraday.live.feed import LiveQuoteFeed
    from quant.intraday.live.loop import TickDeps, run_loop
    from quant.intraday.live.session import session_state
    from quant.intraday.live.sleeve import SleeveLedger
    from quant.intraday.live.strategy import MeanReversionStrategy

    cfg = SleeveConfig()
    dd = _data_dir()
    broker = AlpacaClient()
    feed = LiveQuoteFeed.from_settings(symbols=list(cfg.universe))
    strat = MeanReversionStrategy(cfg)
    ledger = SleeveLedger()  # NOTE: rebuilt-from-broker recovery is Task 13's concern

    def factory() -> TickDeps:
        now = datetime.now(UTC)
        ss = session_state(now)
        return TickDeps(
            data_dir=dd, config=cfg, broker=broker, feed=feed, strategy=strat,
            ledger=ledger, now=now, session_open=ss.open, session_close=ss.close,
            dry_run=dry_run,
        )

    run_loop(factory, max_ticks=max_ticks)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/intraday/live/test_cli.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add quant/intraday/cli.py tests/intraday/live/test_cli.py
git commit -m "feat(intraday-live): CLI subgroup (run/status/halt/resume/flat)"
```

---

### Task 13: Crash-recovery — rebuild ledger from broker on startup

**Files:**
- Modify: `quant/intraday/live/loop.py` (add `recover_ledger`)
- Modify: `quant/intraday/cli.py` (call it in `run` before the loop)
- Test: `tests/intraday/live/test_recovery.py`

On startup the in-memory ledger is empty but the broker may hold sleeve positions from before a crash. Rebuild the ledger's positions from Alpaca, filtered to the sleeve universe, before ticking. If a position exists outside tolerance of what the journal expects, start halted.

- [ ] **Step 1: Write the failing test**

Create `tests/intraday/live/test_recovery.py`:

```python
from quant.intraday.live.config import SleeveConfig
from quant.intraday.live.loop import recover_ledger


class _Pos:
    def __init__(self, symbol, qty, avg): self.symbol, self.qty, self.avg_entry_price = symbol, qty, avg


class _Broker:
    def __init__(self, positions): self._p = positions
    def positions(self): return self._p


def test_recovers_only_sleeve_universe_positions():
    cfg = SleeveConfig(universe=("QQQ", "IWM", "DIA"))
    broker = _Broker([_Pos("QQQ", 7, 100.0), _Pos("SPY", 50, 400.0)])  # SPY = daily system
    led = recover_ledger(broker, cfg)
    assert led.position("QQQ") == 7
    assert led.position("SPY") == 0   # ignored: not in sleeve universe


def test_recovers_empty_when_no_sleeve_positions():
    cfg = SleeveConfig()
    led = recover_ledger(_Broker([]), cfg)
    assert led.positions() == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/intraday/live/test_recovery.py -v`
Expected: FAIL — `ImportError: cannot import name 'recover_ledger'`

- [ ] **Step 3: Write minimal implementation**

Add to `quant/intraday/live/loop.py`:

```python
def recover_ledger(broker: Any, config: SleeveConfig) -> SleeveLedger:
    """Rebuild a SleeveLedger from the broker's current positions, restricted to the
    sleeve universe (daily-system holdings are ignored — they are a disjoint set)."""
    ledger = SleeveLedger()
    for p in broker.positions():
        if p.symbol in config.universe and p.qty != 0:
            ledger.record(Fill(symbol=p.symbol, qty=int(p.qty), price=float(p.avg_entry_price)))
    return ledger
```

Then in `quant/intraday/cli.py` `run`, replace `ledger = SleeveLedger()` with:

```python
    from quant.intraday.live.loop import recover_ledger
    ledger = recover_ledger(broker, cfg)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/intraday/live/test_recovery.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add quant/intraday/live/loop.py quant/intraday/cli.py tests/intraday/live/test_recovery.py
git commit -m "feat(intraday-live): rebuild sleeve ledger from broker on startup (crash recovery)"
```

---

### Task 14: Integration test — full session replay through the loop

**Files:**
- Test: `tests/intraday/live/test_integration.py`

Drives many ticks through `run_tick` with a fake clock, fake feed (a scripted price path), and fake broker, asserting the spine's end-to-end invariants: it acts, never exceeds caps, halts on the loss threshold, and flattens by close.

- [ ] **Step 1: Write the failing test**

Create `tests/intraday/live/test_integration.py`:

```python
from datetime import UTC, datetime, timedelta

from quant.intraday.data.events import QuoteBar
from quant.intraday.live.config import SleeveConfig
from quant.intraday.live.halt import load_sleeve_halt
from quant.intraday.live.loop import TickDeps, run_tick
from quant.intraday.live.sleeve import SleeveLedger
from quant.intraday.live.strategy import MeanReversionStrategy


class _Broker:
    def __init__(self): self.orders = []
    def account(self):
        class A: equity = 100_000.0
        return A()
    def submit_simple_order(self, *, symbol, side, qty, client_order_id,
                            order_type="market", limit_price=None, dry_run=False):
        self.orders.append((symbol, side, qty)); return client_order_id


class _Feed:
    def __init__(self, price_fn): self._fn = price_fn
    def latest_quotes(self, now=None):
        p = self._fn(now)
        return [QuoteBar(ts=now, symbol="QQQ", bid=p - 0.01, ask=p + 0.01,
                         bid_size=100, ask_size=100)]


def test_caps_respected_and_flat_by_close(tmp_path):
    cfg = SleeveConfig(per_trade_cap=2_000.0, mean_reversion_lookback=5)
    broker, ledger = _Broker(), SleeveLedger()
    strat = MeanReversionStrategy(cfg, unit_shares=1000)  # huge -> must be capped
    open_t = datetime(2026, 6, 8, 14, 0, tzinfo=UTC)
    close_t = datetime(2026, 6, 8, 20, 0, tzinfo=UTC)

    # Price path: flat, then a spike (triggers a fade), then flat again.
    def price(now):
        i = int((now - open_t).total_seconds() // 60)
        return 100.0 + (3.0 if i == 6 else 0.0)

    for i in range(8):
        now = open_t + timedelta(minutes=i)
        run_tick(TickDeps(data_dir=tmp_path, config=cfg, broker=broker, feed=_Feed(price),
                          strategy=strat, ledger=ledger, now=now,
                          session_open=True, session_close=close_t))

    # Every order must respect the per-trade cap (<=20 shares @ ~$100).
    assert broker.orders, "expected at least one order"
    assert all(qty <= 20 for _, _, qty in broker.orders)

    # Now jump to the flat-by-close window: any open position must be flattened.
    run_tick(TickDeps(data_dir=tmp_path, config=cfg, broker=broker, feed=_Feed(price),
                      strategy=strat, ledger=ledger,
                      now=datetime(2026, 6, 8, 19, 50, tzinfo=UTC),
                      session_open=True, session_close=close_t))
    assert ledger.positions() == {}


def test_loss_halt_stops_subsequent_trading(tmp_path):
    cfg = SleeveConfig(mean_reversion_lookback=2, daily_loss_halt_pct=0.001)
    broker, ledger = _Broker(), SleeveLedger()
    strat = MeanReversionStrategy(cfg, unit_shares=10)
    close_t = datetime(2026, 6, 8, 20, 0, tzinfo=UTC)

    # Crashing price path guarantees a loss once positioned.
    prices = iter([100.0, 100.0, 100.0, 80.0, 80.0])
    last = [100.0]
    def price(now):
        try: last[0] = next(prices)
        except StopIteration: pass
        return last[0]

    for i in range(5):
        now = datetime(2026, 6, 8, 15, i, tzinfo=UTC)
        run_tick(TickDeps(data_dir=tmp_path, config=cfg, broker=broker, feed=_Feed(price),
                          strategy=strat, ledger=ledger, now=now,
                          session_open=True, session_close=close_t))

    assert load_sleeve_halt(tmp_path).active is True
    n_before = len(broker.orders)
    # A further tick must do nothing (halted).
    run_tick(TickDeps(data_dir=tmp_path, config=cfg, broker=broker, feed=_Feed(price),
                      strategy=strat, ledger=ledger,
                      now=datetime(2026, 6, 8, 15, 9, tzinfo=UTC),
                      session_open=True, session_close=close_t))
    assert len(broker.orders) == n_before
```

- [ ] **Step 2: Run the integration test**

Run: `uv run pytest tests/intraday/live/test_integration.py -v`
Expected: PASS (2 tests). If the mean-reversion path doesn't trigger an order in the first test, widen the spike or lower `entry_z` in the test's config until at least one order fires — the assertion of interest is the cap, not the exact trigger.

- [ ] **Step 3: Commit**

```bash
git add tests/intraday/live/test_integration.py
git commit -m "test(intraday-live): end-to-end session replay (caps, flat-by-close, loss-halt)"
```

---

### Task 15: Daily sleeve recon + summary

**Files:**
- Create: `quant/intraday/live/recon.py`
- Modify: `quant/intraday/cli.py` (add `live recon` command)
- Test: `tests/intraday/live/test_recon.py`

Produces the live side of the drift picture (spec §5/§9): summarize a day's journaled ticks (realized day-P&L, round-trips, halts) and reconcile the ledger's sleeve positions against the broker's, flagging any mismatch outside the sleeve universe. Comparing this against a *backtested* expectation needs the intraday mean-reversion backtest baseline, which is **out of scope for the spine** (it belongs to the offline pipeline / a later sub-project) — this task produces the live summary + position recon and explicitly logs that the backtest-drift comparison is deferred.

- [ ] **Step 1: Write the failing test**

Create `tests/intraday/live/test_recon.py`:

```python
from datetime import UTC, datetime

from quant.intraday.live.config import SleeveConfig
from quant.intraday.live.journal import TickRecord, append_tick
from quant.intraday.live.recon import position_mismatches, summarize_day


class _Pos:
    def __init__(self, symbol, qty): self.symbol, self.qty = symbol, qty


class _Broker:
    def __init__(self, positions): self._p = positions
    def positions(self): return self._p


def test_summarize_day_aggregates_journal(tmp_path):
    for i in range(3):
        append_tick(tmp_path, TickRecord(
            ts=datetime(2026, 6, 8, 15, i, tzinfo=UTC), sleeve_value=0.0,
            day_pnl=float(i * 10), round_trips=i, n_orders=1,
            halted=(i == 2), note="x"))
    s = summarize_day(tmp_path)
    assert s["n_ticks"] == 3
    assert s["last_day_pnl"] == 20.0
    assert s["max_round_trips"] == 2
    assert s["halted_any"] is True


def test_position_mismatch_detects_drift():
    cfg = SleeveConfig(universe=("QQQ", "IWM", "DIA"))
    broker = _Broker([_Pos("QQQ", 7), _Pos("SPY", 50)])  # SPY = daily system, ignored
    ledger_positions = {"QQQ": 5}  # ledger thinks 5, broker says 7 -> mismatch
    bad = position_mismatches(ledger_positions, broker, cfg)
    assert bad == {"QQQ": (5, 7)}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/intraday/live/test_recon.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'quant.intraday.live.recon'`

- [ ] **Step 3: Write minimal implementation**

Create `quant/intraday/live/recon.py`:

```python
"""Daily sleeve recon: the LIVE side of the drift picture. Summarizes journaled
ticks and reconciles ledger vs broker sleeve positions. The backtest-vs-live drift
comparison needs the intraday backtest baseline (offline pipeline) and is deferred."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from quant.intraday.live.config import SleeveConfig
from quant.intraday.live.journal import read_ticks


def summarize_day(data_dir: Path) -> dict[str, Any]:
    df = read_ticks(data_dir)
    if df.empty:
        return {"n_ticks": 0, "last_day_pnl": 0.0, "max_round_trips": 0, "halted_any": False}
    return {
        "n_ticks": int(len(df)),
        "last_day_pnl": float(df.iloc[-1]["day_pnl"]),
        "max_round_trips": int(df["round_trips"].max()),
        "halted_any": bool(df["halted"].any()),
    }


def position_mismatches(
    ledger_positions: dict[str, int], broker: Any, config: SleeveConfig,
) -> dict[str, tuple[int, int]]:
    """Return {symbol: (ledger_qty, broker_qty)} for sleeve-universe symbols whose
    ledger and broker positions disagree. Symbols outside the sleeve universe (i.e.
    the daily system's holdings) are ignored."""
    broker_qty = {p.symbol: int(p.qty) for p in broker.positions() if p.symbol in config.universe}
    out: dict[str, tuple[int, int]] = {}
    for sym in config.universe:
        lq = int(ledger_positions.get(sym, 0))
        bq = broker_qty.get(sym, 0)
        if lq != bq:
            out[sym] = (lq, bq)
    return out
```

Then add a `recon` command to the `live` group in `quant/intraday/cli.py`:

```python
@live.command()
def recon() -> None:
    """Summarize today's sleeve journal + reconcile ledger vs broker positions."""
    from quant.execution.alpaca import AlpacaClient
    from quant.intraday.live.config import SleeveConfig
    from quant.intraday.live.loop import recover_ledger
    from quant.intraday.live.recon import position_mismatches, summarize_day

    dd = _data_dir()
    s = summarize_day(dd)
    click.echo(f"ticks={s['n_ticks']} last_day_pnl={s['last_day_pnl']:.2f} "
               f"max_round_trips={s['max_round_trips']} halted_any={s['halted_any']}")
    cfg = SleeveConfig()
    broker = AlpacaClient()
    ledger = recover_ledger(broker, cfg)  # ledger view rebuilt from broker
    bad = position_mismatches(ledger.positions(), broker, cfg)
    click.echo("position recon: OK" if not bad else f"position MISMATCH: {bad}")
    click.echo("note: backtest-vs-live drift comparison deferred (needs intraday backtest baseline)")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/intraday/live/test_recon.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add quant/intraday/live/recon.py quant/intraday/cli.py tests/intraday/live/test_recon.py
git commit -m "feat(intraday-live): daily sleeve recon + summary (live side of drift)"
```

---

### Task 16: launchd service + operator docs

**Files:**
- Create: `deploy/launchd/com.quant.intraday-live.plist`
- Create: `docs/intraday-live-runbook.md`
- Test: `tests/intraday/live/test_plist.py`

- [ ] **Step 1: Write the failing test**

Create `tests/intraday/live/test_plist.py`:

```python
import plistlib
from pathlib import Path


def test_plist_is_valid_and_keepalive():
    path = Path("deploy/launchd/com.quant.intraday-live.plist")
    data = plistlib.loads(path.read_bytes())
    assert data["Label"] == "com.quant.intraday-live"
    assert data["KeepAlive"] is True
    assert any("intraday" in str(a) and "live" in str(a) for a in data["ProgramArguments"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/intraday/live/test_plist.py -v`
Expected: FAIL — `FileNotFoundError`.

- [ ] **Step 3: Write the plist + runbook**

Create `deploy/launchd/com.quant.intraday-live.plist` (adjust the `uv` path and `WorkingDirectory` to the M4's real paths during deploy):

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.quant.intraday-live</string>
  <key>ProgramArguments</key>
  <array>
    <string>/opt/homebrew/bin/uv</string>
    <string>run</string>
    <string>quant</string>
    <string>intraday</string>
    <string>live</string>
    <string>run</string>
  </array>
  <key>WorkingDirectory</key>
  <string>/Users/ajaiupadhyaya/Documents/quant-trading</string>
  <key>KeepAlive</key>
  <true/>
  <key>RunAtLoad</key>
  <true/>
  <key>StandardOutPath</key>
  <string>/Users/ajaiupadhyaya/Documents/quant-trading/data/intraday/live/launchd.out.log</string>
  <key>StandardErrorPath</key>
  <string>/Users/ajaiupadhyaya/Documents/quant-trading/data/intraday/live/launchd.err.log</string>
</dict>
</plist>
```

Create `docs/intraday-live-runbook.md` documenting: what the sleeve is, the guardrail profile, `launchctl load/unload` of the plist, how to halt/resume (`uv run quant intraday live halt/resume --reason ...`), how to read `status`, where the journal lives (`data/intraday/live/ticks.parquet`), and the crash-recovery behavior. Include the explicit warning: **the sleeve trades a disjoint ETF universe (QQQ/IWM/DIA) from the daily system; never add a daily-held symbol to the sleeve universe.**

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/intraday/live/test_plist.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add deploy/launchd/com.quant.intraday-live.plist docs/intraday-live-runbook.md tests/intraday/live/test_plist.py
git commit -m "feat(intraday-live): launchd service + operator runbook"
```

---

### Task 17: Full-suite green + lint/type gate

**Files:** none (verification task)

- [ ] **Step 1: Run the whole intraday-live suite**

Run: `uv run pytest tests/intraday/live/ -v`
Expected: ALL pass.

- [ ] **Step 2: Run the full repo test suite (no regressions in the daily system)**

Run: `uv run pytest -q`
Expected: pass count = prior baseline + the new tests; the daily-system tests are unchanged.

- [ ] **Step 3: Lint + type gate**

Run: `uv run ruff check . && uv run mypy quant`
Expected: clean. Fix any findings (do NOT suppress with blanket ignores).

- [ ] **Step 4: Manual dry-run smoke (optional, requires keys + open market)**

Run: `uv run quant intraday live run --max-ticks 3 --dry-run`
Expected: three ticks log; `[DRY-RUN]` lines if the strategy fires; `uv run quant intraday live status` shows journaled ticks. No real orders.

- [ ] **Step 5: Commit any fixes**

```bash
git add -A
git commit -m "chore(intraday-live): full-suite green + lint/type clean"
```

---

## Notes for the implementer

- **Provisional fills:** the loop marks fills at the quote mid on the submitting tick for *internal* P&L. Real fill prices come back via order status; reconciling the ledger against actual fills is intentionally deferred to sub-project A/the recon job — the spine's job is to act safely and journal, not to be the accounting source of truth.
- **Budget semantics:** the trade-count budget blocks only NEW opens; exits and flatten always proceed. This keeps the sleeve able to de-risk after the budget is spent.
- **Disjoint universe is load-bearing:** the sleeve's isolation from the daily system depends on QQQ/IWM/DIA never overlapping daily holdings. The runbook states this; recovery enforces it by filtering to the sleeve universe.
- **No live learning:** nothing here trains. Strategy params come from `SleeveConfig` (promoted offline). Drift is observed via the journal, not acted on in-loop.
