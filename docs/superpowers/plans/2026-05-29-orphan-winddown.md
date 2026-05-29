# Orphan-Position Wind-Down Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring positions owned by non-live (quarantined) strategies under governance by winding them down to flat — exit-only, ADV-capped, fail-closed — inside the daily rebalance.

**Architecture:** A new pure-ish module `quant/live/winddown.py` (orphan detection + capped exit-order generation). `check_reconciliation` learns a `winddown_slugs` set so it counts orphan books as "expected" during convergence. `run_rebalance` detects orphans before the reconciliation guard, threads them in, and runs an exit-only wind-down block after the live-strategy loop, persisting the post-exit **remaining** snapshot (incl. explicit zeros). CLI renders the wind-down outcomes.

**Tech Stack:** Python 3, pandas, pytest, uv, ruff + mypy. No new deps.

Spec: `docs/superpowers/specs/2026-05-29-orphan-winddown-design.md`

**CRITICAL correctness note (verified against bookkeeping.py:157):** `write_strategy_positions(..., target={})` is a **no-op** (early-return on empty dict). To mark an orphan flat you MUST write the post-exit remaining book *including explicit `qty=0` entries* for fully-exited symbols. `detect_orphans` and `_snapshot_aggregate` therefore treat all-zero snapshots as flat (already true for `_snapshot_aggregate`, which drops zeros at safety.py:63).

**Verified signatures:** `reconcile(target: dict, current: dict, strategy_slug: str) -> list[OrderTemplate]` (empty target ⇒ flatten-only). `OrderTemplate(symbol, qty:int>0, side: OrderSide, strategy_slug)`, `OrderSide.{BUY,SELL}`. `trailing_dollar_adv(bars, symbol, fill_ts: pd.Timestamp, window:int) -> float`. `last_strategy_positions(data_dir, slug) -> dict[str,int]`. `write_strategy_positions(data_dir, asof, slug, target: dict[str,int]) -> Path`. `_snapshot_aggregate(data_dir, slugs: list[str]) -> dict[str,int]` (drops zeros). `check_reconciliation(*, data_dir, alpaca_positions, enabled_slugs, tolerance_shares=1)`. `client.submit_order(order, dry_run) -> coid`.

---

### Task 1: `capped_qty` (ADV participation cap)

**Files:** Create `quant/live/winddown.py`; Test `tests/live/test_winddown.py`

- [ ] **Step 1: Write the failing tests** — create `tests/live/test_winddown.py`:

```python
"""Tests for the orphan wind-down helpers."""

from __future__ import annotations

from quant.live.winddown import capped_qty


def test_cap_binds_when_order_exceeds_participation():
    # ADV $1,000,000, 10% => $100,000 budget; price $100 => 1000 shares max.
    assert capped_qty(5000, 1_000_000.0, 100.0, 0.10) == 1000


def test_cap_passes_through_when_order_within_budget():
    assert capped_qty(200, 1_000_000.0, 100.0, 0.10) == 200


def test_zero_or_negative_adv_is_zero():
    assert capped_qty(500, 0.0, 100.0, 0.10) == 0
    assert capped_qty(500, -1.0, 100.0, 0.10) == 0


def test_nonpositive_price_is_zero():
    assert capped_qty(500, 1_000_000.0, 0.0, 0.10) == 0


def test_nonfinite_inputs_zero():
    assert capped_qty(500, float("nan"), 100.0, 0.10) == 0
    assert capped_qty(500, 1_000_000.0, float("inf"), 0.10) == 0


def test_nonpositive_order_qty_is_zero():
    assert capped_qty(0, 1_000_000.0, 100.0, 0.10) == 0
```

- [ ] **Step 2: Run — expect FAIL** (`ModuleNotFoundError`): `uv run pytest tests/live/test_winddown.py -v`

- [ ] **Step 3: Create `quant/live/winddown.py`** with the module docstring + `capped_qty`:

```python
"""Governed wind-down of orphan positions: exit-only, ADV-capped, fail-closed.

An orphan = a registered slug holding a non-zero position whose governance
state is not LIVE. The owning strategy is NEVER run (it could re-open); we only
reduce its book toward flat. These helpers are pure given their inputs
(snapshot / bars / governance state) so they unit-test without Alpaca.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd

from quant.backtest.impact import trailing_dollar_adv
from quant.execution.orders import OrderSide, OrderTemplate
from quant.execution.reconciler import reconcile


def capped_qty(
    order_qty: int, adv_dollar: float, price: float, participation_fraction: float
) -> int:
    """Largest share qty <= order_qty whose notional stays within
    ``participation_fraction`` of trailing dollar-ADV. Returns 0 when un-sizable
    (non-positive/non-finite ADV or price, or non-positive order qty)."""
    if order_qty <= 0 or participation_fraction <= 0.0:
        return 0
    if not (math.isfinite(adv_dollar) and math.isfinite(price)):
        return 0
    if adv_dollar <= 0.0 or price <= 0.0:
        return 0
    max_shares = int((adv_dollar * participation_fraction) / price)
    return max(0, min(order_qty, max_shares))
```

- [ ] **Step 4: Run — expect 6 passed:** `uv run pytest tests/live/test_winddown.py -v`

- [ ] **Step 5: Lint/type:** `uv run ruff check quant/live/winddown.py tests/live/test_winddown.py && uv run mypy quant/live/winddown.py && uv run ruff format --check quant/live/winddown.py tests/live/test_winddown.py` (reformat those files if needed)

- [ ] **Step 6: Commit:**
```bash
git add quant/live/winddown.py tests/live/test_winddown.py
git commit -m "feat(winddown): ADV participation cap for orphan exits

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `winddown_orders` (exit-only, capped, remaining book)

**Files:** Modify `quant/live/winddown.py`; Test `tests/live/test_winddown.py`

- [ ] **Step 1: Add failing tests** to `tests/live/test_winddown.py`:

```python
import numpy as np
import pandas as pd
from datetime import date

from quant.execution.orders import OrderSide
from quant.live.winddown import winddown_orders


def _bars(symbol: str, price: float, volume: int, n: int = 30) -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-02", periods=n)
    df = pd.DataFrame(
        {f: np.full(n, price) for f in ("open", "high", "low", "close")}
        | {"volume": np.full(n, volume, dtype=np.int64)},
        index=idx,
    )
    df.index.name = "timestamp"
    return pd.concat({symbol: df}, axis=1)


def test_long_orphan_generates_sell_only_and_remaining_zero():
    bars = _bars("SPY", 100.0, 50_000_000)  # ADV $5B, cap never binds
    res = winddown_orders("trend", {"SPY": 70}, bars, date(2024, 2, 10), 0.10)
    assert [o.side for o in res.orders] == [OrderSide.SELL]
    assert res.orders[0].qty == 70
    assert res.orders[0].strategy_slug == "trend"
    assert res.remaining["SPY"] == 0  # fully exited, explicit zero
    assert res.skipped == []


def test_short_orphan_generates_buy_to_cover_only():
    bars = _bars("TLT", 90.0, 50_000_000)
    res = winddown_orders("pairs", {"TLT": -40}, bars, date(2024, 2, 10), 0.10)
    assert [o.side for o in res.orders] == [OrderSide.BUY]
    assert res.orders[0].qty == 40
    assert res.remaining["TLT"] == 0


def test_adv_cap_partial_exit_leaves_remaining():
    # ADV = 100*5000 = $500k; 10% => $50k; price 100 => 500 shares max.
    bars = _bars("DBC", 100.0, 5_000)
    res = winddown_orders("trend", {"DBC": 1200}, bars, date(2024, 2, 10), 0.10)
    assert res.orders[0].qty == 500          # capped
    assert res.remaining["DBC"] == 700       # 1200 - 500 still held
    assert res.orders[0].side == OrderSide.SELL


def test_symbol_with_no_bars_is_skipped_not_silent():
    bars = _bars("SPY", 100.0, 50_000_000)   # only SPY has bars
    res = winddown_orders("trend", {"ZZZ": 10}, bars, date(2024, 2, 10), 0.10)
    assert res.orders == []
    assert "ZZZ" in res.skipped
    assert res.remaining["ZZZ"] == 10        # unchanged (couldn't exit)


def test_never_opens_a_new_symbol():
    bars = _bars("SPY", 100.0, 50_000_000)
    res = winddown_orders("trend", {"SPY": 70}, bars, date(2024, 2, 10), 0.10)
    # every order reduces toward zero; remaining magnitude <= original
    for sym, q in res.remaining.items():
        assert abs(q) <= abs({"SPY": 70}.get(sym, 0))
```

- [ ] **Step 2: Run — expect FAIL** (`winddown_orders` undefined): `uv run pytest tests/live/test_winddown.py -v`

- [ ] **Step 3: Append to `quant/live/winddown.py`:**

```python
@dataclass(frozen=True)
class WindDownResult:
    """Outcome of one orphan slug's wind-down pass."""

    slug: str
    orders: list[OrderTemplate]                     # capped exit orders (qty > 0)
    reference_prices: dict[str, float]
    remaining: dict[str, int]                       # post-exit book, incl. explicit zeros
    skipped: list[str] = field(default_factory=list)  # symbols un-exitable this pass


def _latest_close(bars: pd.DataFrame, symbol: str) -> float | None:
    col = (symbol, "close")
    if col not in bars.columns:
        return None
    series = bars[col].dropna()
    if series.empty:
        return None
    return float(series.iloc[-1])


def winddown_orders(
    slug: str,
    snapshot: dict[str, int],
    bars: pd.DataFrame,
    asof: date,
    participation_fraction: float,
    adv_window: int = 21,
) -> WindDownResult:
    """Exit-only orders reducing ``snapshot`` toward flat, each capped at the ADV
    participation fraction. Forces ``target={}`` into ``reconcile`` so it is
    structurally flatten-only (sell longs / cover shorts; never opens).

    ``remaining`` is the post-exit book: ``current`` minus the capped exit,
    INCLUDING explicit ``0`` for fully-exited symbols and the unchanged qty for
    symbols that could not be sized (so the caller persists a coherent snapshot
    and the orphan converges across rebalances)."""
    raw = reconcile(target={}, current=snapshot, strategy_slug=slug)  # flatten-only
    fill_ts = pd.Timestamp(asof)
    capped: list[OrderTemplate] = []
    reference_prices: dict[str, float] = {}
    skipped: list[str] = []
    remaining: dict[str, int] = {sym: int(q) for sym, q in snapshot.items()}

    for order in raw:
        sym = order.symbol
        price = _latest_close(bars, sym)
        if price is not None:
            reference_prices[sym] = price
        adv = trailing_dollar_adv(bars, sym, fill_ts, adv_window)
        cap = capped_qty(order.qty, adv, price if price is not None else 0.0, participation_fraction)
        if cap <= 0:
            skipped.append(sym)
            continue
        capped.append(
            OrderTemplate(symbol=sym, qty=cap, side=order.side, strategy_slug=slug)
        )
        cur = remaining.get(sym, 0)
        remaining[sym] = cur - cap if order.side is OrderSide.SELL else cur + cap

    return WindDownResult(
        slug=slug,
        orders=capped,
        reference_prices=reference_prices,
        remaining=remaining,
        skipped=skipped,
    )
```

- [ ] **Step 4: Run — expect all pass:** `uv run pytest tests/live/test_winddown.py -v`
- [ ] **Step 5: Lint/type** (same commands as Task 1 Step 5).
- [ ] **Step 6: Commit:**
```bash
git add quant/live/winddown.py tests/live/test_winddown.py
git commit -m "feat(winddown): exit-only capped orders + remaining-book computation

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: `detect_orphans`

**Files:** Modify `quant/live/winddown.py`; Test `tests/live/test_winddown.py`

- [ ] **Step 1: Add failing test** (uses the `tmp_data_dir` fixture from `tests/conftest.py`):

```python
from quant.live.bookkeeping import write_strategy_positions
from quant.live.winddown import detect_orphans


def test_detect_orphans_filters_live_empty_and_unregistered(tmp_data_dir, monkeypatch):
    from datetime import date
    import quant.governance.store as store
    from quant.governance.models import GovernanceState

    class _S:
        def __init__(self, state): self.state = state

    # trend + multi-factor are registered & quarantined; defensive-etf is LIVE.
    monkeypatch.setattr(
        store, "load_strategy_states",
        lambda _p: {
            "defensive-etf-allocation": _S(GovernanceState.LIVE),
            "trend": _S(GovernanceState.QUARANTINED),
            "multi-factor": _S(GovernanceState.QUARANTINED),
            "not-registered": _S(GovernanceState.QUARANTINED),
        },
    )
    write_strategy_positions(tmp_data_dir, date(2026, 5, 26), "defensive-etf-allocation", {"SPY": 10})
    write_strategy_positions(tmp_data_dir, date(2026, 5, 26), "trend", {"SPY": 70})
    write_strategy_positions(tmp_data_dir, date(2026, 5, 26), "multi-factor", {"BAC": 0})  # all-zero => flat

    orphans = detect_orphans(tmp_data_dir)
    assert orphans == ["trend"]  # live excluded, all-zero excluded, unregistered excluded
```

> Note: `detect_orphans` imports `load_strategy_states` from `quant.governance.store`; the test monkeypatches it there. Confirm the import path matches what `detect_orphans` uses.

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Append `detect_orphans` to `quant/live/winddown.py`:**

```python
def detect_orphans(data_dir: Path) -> list[str]:
    """Sorted registered slugs whose governance state is not LIVE and which hold
    a non-zero latest snapshot. Returns [] if governance state is unavailable
    (fail-closed: no governance ⇒ no wind-down rather than guessing)."""
    from quant.governance.models import GovernanceError, GovernanceState
    from quant.governance.store import load_strategy_states, strategy_states_path
    from quant.live.bookkeeping import last_strategy_positions
    from quant.strategies import REGISTRY

    try:
        states = load_strategy_states(strategy_states_path(data_dir))
    except GovernanceError:
        return []

    orphans: list[str] = []
    for slug in REGISTRY:
        state = states.get(slug)
        if state is not None and state.state is GovernanceState.LIVE:
            continue
        snap = {s: q for s, q in last_strategy_positions(data_dir, slug).items() if q != 0}
        if snap:
            orphans.append(slug)
    return sorted(orphans)
```

> If `load_strategy_states` returns a non-dict mapping without `.get`, adapt to `dict(states).get(slug)`. Verify by reading `quant/governance/store.py:load_strategy_states`.

- [ ] **Step 4: Run — expect pass.**
- [ ] **Step 5: Lint/type.**
- [ ] **Step 6: Commit:**
```bash
git add quant/live/winddown.py tests/live/test_winddown.py
git commit -m "feat(winddown): detect_orphans (non-live, non-zero snapshot, registered)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: `check_reconciliation` learns `winddown_slugs`

**Files:** Modify `quant/live/safety.py` (`check_reconciliation` 66-105); Test `tests/live/test_safety.py` (locate with `grep -rl check_reconciliation tests/`)

- [ ] **Step 1: Find the existing reconciliation test file:** `grep -rln "check_reconciliation" tests/` — add tests to that file (e.g. `tests/live/test_safety.py`).

- [ ] **Step 2: Add failing tests** (adapt imports to the existing test's style; `PositionRow` is in `quant.execution.alpaca`):

```python
from datetime import date
from quant.execution.alpaca import PositionRow
from quant.live.bookkeeping import write_strategy_positions
from quant.live.safety import check_reconciliation


def test_reconciliation_counts_winddown_orphan_as_expected(tmp_data_dir):
    # Live strat holds SPY 10; orphan 'trend' still holds DBC 1000 in Alpaca.
    write_strategy_positions(tmp_data_dir, date(2026, 5, 26), "defensive-etf-allocation", {"SPY": 10})
    write_strategy_positions(tmp_data_dir, date(2026, 5, 26), "trend", {"DBC": 1000})
    alpaca = [PositionRow(symbol="SPY", qty=10), PositionRow(symbol="DBC", qty=1000)]
    # Without winddown_slugs: DBC is unexpected => FAIL.
    bad = check_reconciliation(
        data_dir=tmp_data_dir, alpaca_positions=alpaca, enabled_slugs=["defensive-etf-allocation"]
    )
    assert not bad.ok
    # With winddown_slugs: DBC counted as expected => PASS.
    good = check_reconciliation(
        data_dir=tmp_data_dir, alpaca_positions=alpaca,
        enabled_slugs=["defensive-etf-allocation"], winddown_slugs=["trend"],
    )
    assert good.ok


def test_reconciliation_passes_when_orphan_flat(tmp_data_dir):
    write_strategy_positions(tmp_data_dir, date(2026, 5, 26), "defensive-etf-allocation", {"SPY": 10})
    write_strategy_positions(tmp_data_dir, date(2026, 5, 27), "trend", {"DBC": 0})  # flattened
    alpaca = [PositionRow(symbol="SPY", qty=10)]
    res = check_reconciliation(
        data_dir=tmp_data_dir, alpaca_positions=alpaca,
        enabled_slugs=["defensive-etf-allocation"], winddown_slugs=["trend"],
    )
    assert res.ok
```

> Verify the real `PositionRow` constructor fields in `quant/execution/alpaca.py` and adapt the kwargs if they differ (e.g. it may require more fields).

- [ ] **Step 3: Run — expect FAIL** (`winddown_slugs` is an unexpected kwarg).

- [ ] **Step 4: Edit `check_reconciliation`** in `quant/live/safety.py` — add the param and union:

```python
def check_reconciliation(
    *,
    data_dir: Path,
    alpaca_positions: list[PositionRow],
    enabled_slugs: list[str],
    winddown_slugs: list[str] | None = None,
    tolerance_shares: int = 1,
) -> CheckResult:
    ...
    expected = _snapshot_aggregate(data_dir, list(enabled_slugs) + list(winddown_slugs or []))
```

(Only the signature line and the `expected = ...` line at safety.py:79 change; the docstring may note that wind-down slugs are counted as expected during convergence. Everything else is unchanged.)

- [ ] **Step 5: Run — expect pass.**
- [ ] **Step 6: Lint/type** `quant/live/safety.py` + the test file.
- [ ] **Step 7: Commit:**
```bash
git add quant/live/safety.py tests/
git commit -m "feat(winddown): reconciliation counts wind-down orphan books as expected

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Wire wind-down into `run_rebalance`

**Files:** Modify `quant/live/rebalance.py`; Test `tests/live/` (rebalance integration — locate the existing run_rebalance test with `grep -rln "run_rebalance" tests/`)

- [ ] **Step 1: Add `WindDownOutcome` + report field + param.** Near the other dataclasses (after `StrategyRebalanceOutcome`, ~line 55):

```python
@dataclass
class WindDownOutcome:
    slug: str
    exited: dict[str, int] = field(default_factory=dict)   # symbol -> shares exited this pass
    remaining: dict[str, int] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)
    error: str | None = None
```

Add to `RebalanceReport` (after `skipped_reason`): `winddown_outcomes: list[WindDownOutcome] = field(default_factory=list)`.

Add a parameter to `run_rebalance` (after `include_quarantined`): `winddown_participation: float = 0.10,`.

- [ ] **Step 2: Detect orphans + thread into reconciliation.** Right AFTER the `enabled` list is finalized and the "no enabled" fail-closed check (after line 251, before the reconciliation guard at 255):

```python
    from quant.live.winddown import detect_orphans, winddown_orders

    orphans = detect_orphans(settings.data_dir)
```

Then change the reconciliation call (lines 256-260) to pass them:

```python
        recon = check_reconciliation(
            data_dir=settings.data_dir,
            alpaca_positions=client.positions(),
            enabled_slugs=enabled,
            winddown_slugs=orphans,
        )
```

- [ ] **Step 3: Add the wind-down block AFTER the main loop (after line 424), BEFORE `append_trades` (line 426).** Insert:

```python
    # Orphan wind-down: exit-only, ADV-capped, fail-closed. Runs after the live
    # loop so orphan trade rows flush in the same append_trades call below.
    for slug in orphans:
        if slug in halted:
            continue
        if slug not in REGISTRY:
            report.winddown_outcomes.append(
                WindDownOutcome(slug=slug, error="not registered; manual exit required")
            )
            continue
        try:
            wd_bars = _bars_for(REGISTRY[slug], asof, history_days)
        except Exception as exc:
            report.winddown_outcomes.append(WindDownOutcome(slug=slug, error=f"bar fetch failed: {exc!r}"))
            continue
        snapshot = last_strategy_positions(settings.data_dir, slug)
        if not any(q != 0 for q in snapshot.values()):
            continue
        result = winddown_orders(
            slug=slug, snapshot=snapshot, bars=wd_bars, asof=asof,
            participation_fraction=winddown_participation,
        )
        exited: dict[str, int] = {}
        for order in result.orders:
            try:
                coid = client.submit_order(order, dry_run=dry_run)
            except Exception as exc:
                logger.exception("winddown submit_order failed for {} {}", slug, order.symbol)
                report.winddown_outcomes.append(WindDownOutcome(slug=slug, error=f"submit failed: {exc!r}"))
                continue
            exited[order.symbol] = exited.get(order.symbol, 0) + int(order.qty)
            all_trade_rows.append(
                {
                    "date": pd.Timestamp(asof),
                    "strategy": slug,
                    "symbol": order.symbol,
                    "side": str(order.side),
                    "qty": int(order.qty),
                    "client_order_id": coid,
                    "dry_run": bool(dry_run),
                }
            )
        # Persist the post-exit remaining book (incl. explicit zeros) so the
        # orphan converges and reconciliation stops flagging it — LIVE ONLY.
        if not dry_run and record_bookkeeping:
            write_strategy_positions(settings.data_dir, asof, slug, result.remaining)
        report.winddown_outcomes.append(
            WindDownOutcome(slug=slug, exited=exited, remaining=result.remaining, skipped=result.skipped)
        )
```

> Note `OrderSide` is a `StrEnum`, so `str(order.side)` yields `"buy"/"sell"` matching the live loop's trade-row schema.

- [ ] **Step 4: Add the integration tests.** In the existing run_rebalance test file, add a two-rebalance convergence test and a dry-run no-op test. Use the same harness the existing tests use (fake/stub `AlpacaClient`, `tmp_data_dir`, governance states with one LIVE + one QUARANTINED slug holding a snapshot). Assert:
  - Rebalance 1 (live): a wind-down order is submitted for the orphan; afterwards `last_strategy_positions(orphan)` is all-zero; `winddown_outcomes` records the exit.
  - Rebalance 2 (live): `detect_orphans` no longer returns the orphan (flat) and `winddown_outcomes` is empty for it; reconciliation passes.
  - Dry-run: `winddown_orders` may be computed but **no** order is submitted and the orphan snapshot is **unchanged**.

  Model the test on the existing run_rebalance test's client stub. Capture the submitted orders via the stub. Keep `skip_safety_checks` as the existing tests do, but include one test that exercises `check_reconciliation` with the orphan present to prove convergence (or rely on Task 4's unit tests for that path if the rebalance test stubs safety).

- [ ] **Step 5: Run** the new integration tests + the existing rebalance suite: `uv run pytest tests/live/ -v`. Expected: pass (existing tests unaffected — wind-down only acts when orphans exist with non-LIVE governance state, which the existing fixtures don't create unless they do; if an existing test now produces wind-down activity, confirm it's correct and adjust that test's expectation only if it legitimately involves a non-live slug with a snapshot).

- [ ] **Step 6: Lint/type** `quant/live/rebalance.py` + test files.
- [ ] **Step 7: Commit:**
```bash
git add quant/live/rebalance.py tests/
git commit -m "feat(winddown): wire fail-closed orphan wind-down into run_rebalance

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: CLI surfacing

**Files:** Modify `quant/cli.py` (the `rebalance` command + its `run_rebalance` call)

- [ ] **Step 1: Locate the rebalance command** (`grep -n "def rebalance\|run_rebalance(" quant/cli.py`). Add a Click option and thread it:

```python
@click.option("--winddown-participation", default=0.10, show_default=True, type=float,
              help="Max fraction of trailing dollar-ADV to use per orphan exit order.")
```
and pass `winddown_participation=winddown_participation` into the `run_rebalance(...)` call.

- [ ] **Step 2: Render the wind-down table** after the per-strategy table is printed. Read the existing report-rendering block first (it builds a Rich `Table`), then add:

```python
    if report.winddown_outcomes:
        wd = Table(title="Orphan wind-down", show_header=True)
        wd.add_column("Strategy")
        wd.add_column("Exited", justify="right")
        wd.add_column("Remaining", justify="right")
        wd.add_column("Note")
        for o in report.winddown_outcomes:
            exited = ", ".join(f"{s}:{q}" for s, q in sorted(o.exited.items())) or "—"
            remaining = ", ".join(f"{s}:{q}" for s, q in sorted(o.remaining.items()) if q) or "flat"
            note = o.error or ("skipped " + ",".join(o.skipped) if o.skipped else "")
            wd.add_row(o.slug, exited, remaining, note)
        console.print(wd)
```

- [ ] **Step 3: Verify the CLI imports + compiles:** `uv run python -c "import quant.cli"` and `uv run quant rebalance --help` (should list `--winddown-participation`).

- [ ] **Step 4: Lint/type** `quant/cli.py`.
- [ ] **Step 5: Commit:**
```bash
git add quant/cli.py
git commit -m "feat(winddown): surface orphan wind-down in the rebalance CLI

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Full verification

- [ ] **Step 1: Full suite + types + lint + format:**
```bash
uv run pytest -q
uv run mypy quant
uv run ruff check quant tests
uv run ruff format --check quant tests
```
Expected: all clean. If a pre-existing test fails, triage: a failure caused by wind-down only occurs if that test creates a non-LIVE slug with a non-zero snapshot AND exercises run_rebalance live — confirm correctness before adjusting any expectation; never loosen a fail-closed assertion.

- [ ] **Step 2: Dry-run the real rebalance against the live paper account** (read-only — dry-run never submits): `uv run quant rebalance --dry-run` and confirm the wind-down table lists the four orphan slugs (trend/multi-factor/risk-parity/momentum) with their intended exits, and that the snapshots are NOT mutated (dry-run). Report the output.

- [ ] **Step 3: Commit any triaged test updates** (if Step 1 required them), else skip.

---

## Notes
- **Fail-closed invariants** (all enforced + tested): never opens (forces `target={}`); every exit ADV-capped; dry-run never submits and never zeroes snapshots; remaining-book persisted with explicit zeros (live only); `slug not in REGISTRY` skipped+logged; orphans derived from governance state at runtime; halt/market-open guards still gate the whole pass.
- **Out of scope:** strategy revival, the gate-calibration audit.
- **Push:** local commits; do not push to `origin/main` without explicit operator approval (granted this session).
