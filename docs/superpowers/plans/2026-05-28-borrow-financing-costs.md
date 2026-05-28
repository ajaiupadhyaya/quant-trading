# Borrow & Financing Costs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Charge daily short-borrow and margin-debit financing costs in the backtest engine (gap #2, slice 2a), on by default, calendar-day accrual.

**Architecture:** A new pure module `quant/backtest/financing.py` computes a per-day `FinancingCharge` from carried positions + prior-bar close + cash. `run_backtest` calls it at the top of each bar (PIT, prior-bar close, actual/365) and deducts it from cash, accumulating totals into `BacktestResult.metadata`. Two new `BacktestConfig` rate fields thread through `combined.py`; the `quant backtest` CLI table gains a Financing $ column.

**Tech Stack:** Python 3, pandas, pytest, uv (runner), ruff + mypy (lint/type). No new dependencies.

Spec: `docs/superpowers/specs/2026-05-28-borrow-financing-costs-design.md`

---

### Task 1: Pure `financing_charge` function

**Files:**
- Create: `quant/backtest/financing.py`
- Test: `tests/backtest/test_financing.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/backtest/test_financing.py`:

```python
"""Tests for the pure financing-charge model."""

from __future__ import annotations

import pytest

from quant.backtest.financing import FinancingCharge, financing_charge

_BORROW = 50.0  # bps/yr
_FIN = 200.0  # bps/yr


def test_single_short_one_day():
    # short 100 @ $50 = $5,000 notional; 50 bps/yr; 1 day; positive cash.
    c = financing_charge({"AAA": -100}, {"AAA": 50.0}, cash=10_000.0, days_elapsed=1,
                         annual_borrow_bps=_BORROW, annual_financing_bps=_FIN)
    assert c.borrow_cost == pytest.approx(5_000.0 * (50.0 / 1e4) * (1 / 365))
    assert c.margin_financing_cost == 0.0
    assert c.total == pytest.approx(c.borrow_cost)


def test_long_only_positive_cash_is_zero():
    c = financing_charge({"AAA": 100}, {"AAA": 50.0}, cash=10_000.0, days_elapsed=1,
                         annual_borrow_bps=_BORROW, annual_financing_bps=_FIN)
    assert c.borrow_cost == 0.0
    assert c.margin_financing_cost == 0.0


def test_weekend_gap_is_three_days():
    one = financing_charge({"AAA": -100}, {"AAA": 50.0}, 10_000.0, 1, _BORROW, _FIN)
    three = financing_charge({"AAA": -100}, {"AAA": 50.0}, 10_000.0, 3, _BORROW, _FIN)
    assert three.borrow_cost == pytest.approx(3.0 * one.borrow_cost)


def test_margin_debit_is_financed():
    # negative cash -2,000; 200 bps/yr; 1 day; no shorts.
    c = financing_charge({"AAA": 100}, {"AAA": 50.0}, cash=-2_000.0, days_elapsed=1,
                         annual_borrow_bps=_BORROW, annual_financing_bps=_FIN)
    assert c.borrow_cost == 0.0
    assert c.margin_financing_cost == pytest.approx(2_000.0 * (200.0 / 1e4) * (1 / 365))


def test_zero_or_negative_days_is_zero():
    for d in (0, -5):
        c = financing_charge({"AAA": -100}, {"AAA": 50.0}, -2_000.0, d, _BORROW, _FIN)
        assert c.total == 0.0


def test_combined_short_and_debit():
    c = financing_charge({"AAA": -100}, {"AAA": 50.0}, cash=-2_000.0, days_elapsed=1,
                         annual_borrow_bps=_BORROW, annual_financing_bps=_FIN)
    exp_borrow = 5_000.0 * (50.0 / 1e4) * (1 / 365)
    exp_fin = 2_000.0 * (200.0 / 1e4) * (1 / 365)
    assert c.borrow_cost == pytest.approx(exp_borrow)
    assert c.margin_financing_cost == pytest.approx(exp_fin)
    assert c.total == pytest.approx(exp_borrow + exp_fin)


def test_missing_or_nonfinite_prior_close_contributes_zero():
    # one short missing from prior_close, one with NaN price -> both contribute 0.
    c = financing_charge({"AAA": -100, "BBB": -100}, {"BBB": float("nan")},
                         cash=10_000.0, days_elapsed=1,
                         annual_borrow_bps=_BORROW, annual_financing_bps=_FIN)
    assert c.borrow_cost == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/backtest/test_financing.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'quant.backtest.financing'`

- [ ] **Step 3: Write the module**

Create `quant/backtest/financing.py`:

```python
"""Daily borrow + margin-financing costs, computed from carried positions.

Unlike ``engine.apply_costs`` (a per-fill transaction cost), this is a per-day
holding cost on the positions and cash carried overnight. It takes plain rate
floats rather than a ``BacktestConfig`` so it stays standalone and testable and
so ``engine.py`` can import it without a circular dependency.

Costs only — no interest credits (no short rebate, no idle-cash interest).
Undefined / degenerate inputs yield 0.0 components; the function never raises,
mirroring the engine's tolerance for sparse bars.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

_DAYS_PER_YEAR = 365.0


@dataclass(frozen=True)
class FinancingCharge:
    """Breakdown of one bar's financing cost, in dollars."""

    borrow_cost: float
    margin_financing_cost: float

    @property
    def total(self) -> float:
        return self.borrow_cost + self.margin_financing_cost


def financing_charge(
    positions: Mapping[str, int],
    prior_close: Mapping[str, float],
    cash: float,
    days_elapsed: int,
    annual_borrow_bps: float,
    annual_financing_bps: float,
) -> FinancingCharge:
    """Borrow fee on short notional + financing on a margin debit, actual/365.

    ``positions``/``prior_close`` are the holdings carried overnight and the
    PRIOR bar's close prices (no lookahead). A short whose price is missing or
    non-finite contributes 0. ``days_elapsed`` is calendar days since the prior
    bar; ``<= 0`` yields a zero charge.
    """
    if days_elapsed <= 0:
        return FinancingCharge(0.0, 0.0)
    year_frac = days_elapsed / _DAYS_PER_YEAR

    short_notional = 0.0
    for sym, qty in positions.items():
        if qty >= 0:
            continue
        price = prior_close.get(sym)
        if price is None or not math.isfinite(price):
            continue
        short_notional += abs(qty) * price

    borrow_cost = short_notional * (annual_borrow_bps / 1e4) * year_frac
    margin_debit = max(0.0, -cash)
    margin_financing_cost = margin_debit * (annual_financing_bps / 1e4) * year_frac
    return FinancingCharge(borrow_cost=borrow_cost, margin_financing_cost=margin_financing_cost)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/backtest/test_financing.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Lint + type the new files**

Run:
```bash
uv run mypy quant/backtest/financing.py
uv run ruff check quant/backtest/financing.py tests/backtest/test_financing.py
uv run ruff format --check quant/backtest/financing.py tests/backtest/test_financing.py
```
Expected: all clean. (If format --check reports a file would reformat, run `uv run ruff format` on those two files and re-check.)

- [ ] **Step 6: Commit**

```bash
git add quant/backtest/financing.py tests/backtest/test_financing.py
git commit -m "feat(financing): pure daily borrow + margin-financing charge

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Add rate fields to `BacktestConfig`

**Files:**
- Modify: `quant/backtest/engine.py` (`BacktestConfig` ~23-30)
- Test: `tests/backtest/test_engine_costs.py` (`test_default_config_values` ~56-61)

- [ ] **Step 1: Add the default-value assertions to the existing test**

In `tests/backtest/test_engine_costs.py`, extend `test_default_config_values` with two assertions:

```python
def test_default_config_values() -> None:
    cfg = BacktestConfig()
    assert cfg.starting_equity == 100_000.0
    assert cfg.slippage_bps == 5.0
    assert cfg.commission_bps == 0.0
    assert cfg.execution == "next_open"
    assert cfg.annual_borrow_bps == 50.0
    assert cfg.annual_financing_bps == 200.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/backtest/test_engine_costs.py::test_default_config_values -v`
Expected: FAIL — `AttributeError: 'BacktestConfig' object has no attribute 'annual_borrow_bps'`

- [ ] **Step 3: Add the fields**

In `quant/backtest/engine.py`, add two fields to the `BacktestConfig` frozen dataclass after `commission_bps`:

```python
@dataclass(frozen=True)
class BacktestConfig:
    """Engine configuration. All defaults are intentional — change with care."""

    starting_equity: float = 100_000.0
    slippage_bps: float = 5.0
    commission_bps: float = 0.0
    # Financing (gap #2, slice 2a): short borrow fee + margin-debit financing,
    # accrued daily (actual/365). On by default. annual_financing_bps is a flat
    # approximation of the broker call rate and only bites under >1x gross.
    annual_borrow_bps: float = 50.0
    annual_financing_bps: float = 200.0
    execution: Literal["next_open", "close"] = "next_open"
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/backtest/test_engine_costs.py::test_default_config_values -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add quant/backtest/engine.py tests/backtest/test_engine_costs.py
git commit -m "feat(financing): add borrow/financing rate fields to BacktestConfig

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Wire financing into `run_backtest` + populate metadata

**Files:**
- Modify: `quant/backtest/engine.py` (`run_backtest` loop ~130-237)
- Test: `tests/backtest/test_engine_financing.py`

- [ ] **Step 1: Write the failing integration tests**

Create `tests/backtest/test_engine_financing.py`:

```python
"""Integration tests: borrow/financing charged through run_backtest."""

from __future__ import annotations

from datetime import date
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from quant.backtest.engine import BacktestConfig, run_backtest
from quant.strategies.base import Strategy, StrategySpec
from tests.conftest import EqualWeightStrategy


def _flat_bars(symbols: list[str], price: float = 100.0) -> pd.DataFrame:
    """Bars where every field equals `price`, every business day in 2024-Q1."""
    dates = pd.bdate_range("2024-01-02", "2024-03-29")
    frames: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = pd.DataFrame(
            {f: np.full(len(dates), price) for f in ("open", "high", "low", "close")}
            | {"volume": np.full(len(dates), 1_000_000, dtype=np.int64)},
            index=dates,
        )
        df.index.name = "timestamp"
        frames[sym] = df
    return pd.concat(frames, axis=1)


class _FixedShortStrategy(Strategy):
    """Test-only: hold a fixed -100 share short in AAA at every rebalance."""

    spec: ClassVar[StrategySpec] = StrategySpec(
        slug="fixed-short-test",
        name="Fixed Short (test)",
        description="Test fixture: constant 100-share short in AAA.",
        universe=["AAA"],
        rebalance_frequency="monthly",
    )
    default_params: ClassVar[dict[str, object]] = {}

    def __init__(self, bars: pd.DataFrame) -> None:
        super().__init__(params=None)
        self._bars = bars

    def generate_signals(self, asof: date) -> pd.Series:
        return pd.Series({"AAA": -1.0})

    def target_positions(self, asof: date, equity: float) -> dict[str, int]:
        return {"AAA": -100}


def _short_cfg(borrow: float, fin: float) -> BacktestConfig:
    return BacktestConfig(
        starting_equity=1_000_000.0,
        slippage_bps=0.0,
        commission_bps=0.0,
        annual_borrow_bps=borrow,
        annual_financing_bps=fin,
        execution="close",
    )


def test_borrow_matches_expected_and_drains_equity():
    bars = _flat_bars(["AAA"])
    strat = _FixedShortStrategy(bars)
    res = run_backtest(strat, bars, _short_cfg(50.0, 0.0), date(2024, 1, 2), date(2024, 3, 29))
    idx = res.equity_curve.index
    days = (idx[-1] - idx[0]).days  # consecutive-bar gaps telescope to this span
    expected = 100 * 100.0 * (50.0 / 1e4) * (days / 365.0)
    assert res.metadata["financing_cost_total"] == pytest.approx(expected, rel=1e-9)
    assert res.metadata["margin_financing_cost"] == 0.0  # short proceeds keep cash positive

    res0 = run_backtest(strat, bars, _short_cfg(0.0, 0.0), date(2024, 1, 2), date(2024, 3, 29))
    assert res0.metadata["financing_cost_total"] == 0.0
    assert res.ending_equity == pytest.approx(res0.ending_equity - expected, rel=1e-9)


def test_financing_uses_prior_close_not_today_pit():
    # Spike the LAST bar's close to $200. A PIT charge uses the PRIOR close ($100)
    # for every accrual, so the total stays $100-based; using today's close would
    # inflate the final day's charge.
    bars = _flat_bars(["AAA"])
    last = bars.index[-1]
    bars.loc[last, ("AAA", "close")] = 200.0
    strat = _FixedShortStrategy(bars)
    res = run_backtest(strat, bars, _short_cfg(50.0, 0.0), date(2024, 1, 2), date(2024, 3, 29))
    idx = res.equity_curve.index
    days = (idx[-1] - idx[0]).days
    expected = 100 * 100.0 * (50.0 / 1e4) * (days / 365.0)
    assert res.metadata["financing_cost_total"] == pytest.approx(expected, rel=1e-9)


def test_long_only_default_rates_zero_financing(make_bars):
    bars = make_bars(["AAA", "BBB"], date(2024, 1, 1), date(2024, 6, 30), seed=0)
    strat = EqualWeightStrategy(bars=bars)
    res = run_backtest(strat, bars, BacktestConfig(), date(2024, 1, 1), date(2024, 6, 30))
    assert res.metadata["financing_cost_total"] == 0.0
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/backtest/test_engine_financing.py -v`
Expected: FAIL — `KeyError: 'financing_cost_total'` (metadata not yet populated)

- [ ] **Step 3: Add the accumulators + import**

In `quant/backtest/engine.py`, add the import near the top (after the `apply_costs` is defined in this file, so add it with the other imports at the top — `financing` sorts after `engine`'s own module, so import it at module top):

```python
from quant.backtest.financing import financing_charge
```

Then in `run_backtest`, just before the `for ts in history:` loop (after the `pending: list[...] = []` line ~137), initialize the accumulators and prev-timestamp tracker:

```python
    prev_ts: pd.Timestamp | None = None
    borrow_total: float = 0.0
    margin_financing_total: float = 0.0
```

- [ ] **Step 4: Add the top-of-bar financing accrual**

Inside the loop, immediately after `asof: date = ts.date()` (~line 163) and BEFORE the "1. Execute pending fills" block, insert:

```python
        # 0. Accrue overnight financing on positions/cash carried from the prior
        #    bar, priced at the PRIOR bar's close (PIT, no lookahead).
        if prev_ts is not None:
            prior_close = {
                sym: float(bars[(sym, "close")].loc[prev_ts])
                for sym in positions
                if (sym, "close") in bars.columns
            }
            charge = financing_charge(
                positions=positions,
                prior_close=prior_close,
                cash=cash,
                days_elapsed=(ts - prev_ts).days,
                annual_borrow_bps=config.annual_borrow_bps,
                annual_financing_bps=config.annual_financing_bps,
            )
            cash -= charge.total
            borrow_total += charge.borrow_cost
            margin_financing_total += charge.margin_financing_cost
        prev_ts = ts
```

- [ ] **Step 5: Populate metadata on the main return**

Change the final `return BacktestResult(...)` (~line 229) to pass `metadata`:

```python
    return BacktestResult(
        equity_curve=equity_curve,
        returns=returns,
        positions=positions_df,
        trades=trades_df,
        config=config,
        starting_equity=config.starting_equity,
        ending_equity=ending_equity,
        metadata={
            "borrow_cost": borrow_total,
            "margin_financing_cost": margin_financing_total,
            "financing_cost_total": borrow_total + margin_financing_total,
        },
    )
```

(Leave the early empty-history `return` at ~line 119 unchanged — its `metadata` defaults to `{}`.)

- [ ] **Step 6: Run the new integration tests**

Run: `uv run pytest tests/backtest/test_engine_financing.py -v`
Expected: PASS (3 passed)

- [ ] **Step 7: Run the full backtest test group + triage**

Run: `uv run pytest tests/backtest/ -q`

Default-on borrow is **expected** to change results for any test that runs a *shorting* strategy and asserts a specific equity/return/Sharpe value. If such a test now fails:
- Confirm it involves shorts (the strategy holds negative positions) — if so, the shift is the intended cost change; update the expected value in that test.
- A **long-only** test failing (e.g. `_flat_bars` EqualWeight tests, which hold no shorts and keep cash positive) would indicate a BUG — do not paper over it; report it.

Report any failures and how you triaged them; do not silently edit assertions on tests you cannot confirm involve shorts.

- [ ] **Step 8: Lint + type**

Run:
```bash
uv run mypy quant/backtest/engine.py
uv run ruff check quant/backtest/engine.py tests/backtest/test_engine_financing.py
uv run ruff format --check quant/backtest/engine.py tests/backtest/test_engine_financing.py
```
Expected: clean (reformat only files this task changed if needed).

- [ ] **Step 9: Commit**

```bash
git add quant/backtest/engine.py tests/backtest/test_engine_financing.py
git commit -m "feat(financing): accrue daily borrow/financing in run_backtest

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Thread rates through the combined book

**Files:**
- Modify: `quant/backtest/combined.py` (`sub_config` ~100-105)
- Test: `tests/backtest/test_combined_book.py` (add one test)

- [ ] **Step 1: Write the failing test**

In `tests/backtest/test_combined_book.py`, add a test that a shorting sub-strategy incurs financing through the combined book. Put the helper strategy + flat bars inline (or import from the financing test module if already importable; inline is safest):

```python
def test_combined_book_charges_financing_to_shorting_substrategy():
    from datetime import date
    from typing import ClassVar

    import numpy as np
    import pandas as pd

    from quant.backtest.combined import run_combined_book
    from quant.backtest.engine import BacktestConfig
    from quant.strategies.base import Strategy, StrategySpec

    dates = pd.bdate_range("2024-01-02", "2024-03-29")
    df = pd.DataFrame(
        {f: np.full(len(dates), 100.0) for f in ("open", "high", "low", "close")}
        | {"volume": np.full(len(dates), 1_000_000, dtype=np.int64)},
        index=dates,
    )
    df.index.name = "timestamp"
    bars = pd.concat({"AAA": df}, axis=1)

    class _Short(Strategy):
        spec: ClassVar[StrategySpec] = StrategySpec(
            slug="short-test", name="Short (test)", description="100-share AAA short.",
            universe=["AAA"], rebalance_frequency="monthly",
        )
        default_params: ClassVar[dict[str, object]] = {}

        def __init__(self, bars: pd.DataFrame) -> None:
            super().__init__(params=None)
            self._bars = bars

        def generate_signals(self, asof: date) -> pd.Series:
            return pd.Series({"AAA": -1.0})

        def target_positions(self, asof: date, equity: float) -> dict[str, int]:
            return {"AAA": -100}

    cfg = BacktestConfig(starting_equity=1_000_000.0, slippage_bps=0.0,
                         commission_bps=0.0, annual_borrow_bps=50.0,
                         annual_financing_bps=0.0, execution="close")
    result = run_combined_book(
        strategies={"short-test": _Short(bars)},
        bars_per_strategy={"short-test": bars},
        config=cfg,
        start=date(2024, 1, 2),
        end=date(2024, 3, 29),
        allocation={"short-test": 1.0},
    )
    sub = result.per_strategy["short-test"]
    assert sub.metadata["financing_cost_total"] > 0.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/backtest/test_combined_book.py::test_combined_book_charges_financing_to_shorting_substrategy -v`
Expected: FAIL — `financing_cost_total` is 0.0 because `sub_config` doesn't carry the rates yet.

- [ ] **Step 3: Thread the rates into `sub_config`**

In `quant/backtest/combined.py`, update the `sub_config = BacktestConfig(...)` (~line 100) to pass the two new fields:

```python
        sub_config = BacktestConfig(
            starting_equity=slice_equity,
            slippage_bps=config.slippage_bps,
            commission_bps=config.commission_bps,
            annual_borrow_bps=config.annual_borrow_bps,
            annual_financing_bps=config.annual_financing_bps,
            execution=config.execution,
        )
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/backtest/test_combined_book.py -q`
Expected: PASS (existing combined-book tests + the new one). If a pre-existing combined test that runs *real shorting strategies* now asserts a stale value, triage per Task 3 Step 7.

- [ ] **Step 5: Lint + type**

Run:
```bash
uv run mypy quant/backtest/combined.py
uv run ruff check quant/backtest/combined.py tests/backtest/test_combined_book.py
uv run ruff format --check quant/backtest/combined.py tests/backtest/test_combined_book.py
```
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add quant/backtest/combined.py tests/backtest/test_combined_book.py
git commit -m "feat(financing): thread borrow/financing rates through combined book

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Financing column in the `quant backtest` CLI table + full verification

**Files:**
- Modify: `quant/cli.py` (combined-book table ~225-251)

- [ ] **Step 1: Add the column definition**

In `quant/cli.py`, in the combined-book `Table(...)` setup (after the `table.add_column("Max DD", ...)` and the Turnover column added in the prior slice), add a Financing column. Read the current columns first; add this after the last existing column:

```python
    table.add_column("Financing $", justify="right")
```

- [ ] **Step 2: Add the per-strategy cell**

In the `for slug in sorted(result.per_strategy):` loop's `table.add_row(...)`, append a financing cell after the existing last cell:

```python
            f"${float(sub.metadata.get('financing_cost_total', 0.0)):,.0f}",
```

- [ ] **Step 3: Add the COMBINED cell (sum across per-strategy)**

In the `table.add_row("[bold]COMBINED[/]", ...)` block, append:

```python
        f"${sum(float(s.metadata.get('financing_cost_total', 0.0)) for s in result.per_strategy.values()):,.0f}",
```

Confirm by reading the region that the per-strategy and COMBINED `add_row` calls have the SAME number of cells as the table has columns after your additions.

- [ ] **Step 4: Verify the CLI compiles and imports**

Run:
```bash
uv run python -c "import quant.cli"
uv run mypy quant/cli.py
uv run ruff check quant/cli.py
uv run ruff format --check quant/cli.py
```
Expected: clean (reformat `quant/cli.py` only if needed).

- [ ] **Step 5: Full verification (suite, types, lint, format)**

Run each and confirm clean:

```bash
uv run pytest -q
uv run mypy quant
uv run ruff check quant tests
uv run ruff format --check quant tests
```

Expected: full suite passes, mypy strict clean, ruff clean, format clean. If any pre-existing test that runs a *shorting* strategy fails on a stale asserted value, triage per Task 3 Step 7 (update the expectation only when shorts are clearly involved; a long-only failure is a bug). Report the final pass count and any expectations you updated, with the before/after numbers.

- [ ] **Step 6: Commit**

```bash
git add quant/cli.py
git commit -m "feat(financing): Financing \$ column in the quant backtest CLI table

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Notes

- **Default-on is intentional.** Borrow/financing change the economics of shorting strategies; that is the point (charter principle 2). Governance/validation evidence is refreshed later by re-running `quant validate` + `quant governance refresh` — out of scope for this plan.
- **Out of scope:** market-impact model (slice 2b), capacity (slice 2c), per-symbol/hard-to-borrow rates, rate-curve-linked financing, interest credits, sweeping financing in the cost-sensitivity validation sweep, and a financing line in the walk-forward tear-sheet's multi-window aggregation (the cost already shows in every tear-sheet metric).
- **Push:** all commits are local. Do not push to public `origin/main` without explicit operator approval.
