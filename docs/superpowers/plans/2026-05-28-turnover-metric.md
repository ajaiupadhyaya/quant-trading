# Turnover Metric Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add fills-based one-way annualized turnover reporting (charter gap #1, slice 1), surfaced everywhere Sharpe/MaxDD already appear.

**Architecture:** A new `quant/backtest/activity.py` module holds a pure `annualized_turnover(trades, equity_curve)` function that reads the trade ledger + equity curve (not the returns Series `metrics.py` is built around). It is wired into the `_MetricsBundle` dataclass at all three tear-sheet build sites, both HTML templates, and the `quant backtest` CLI table. Capacity is out of scope (deferred to charter gap #2).

**Tech Stack:** Python 3, numpy, pandas, pytest, Jinja2 (tear-sheet templates), Rich (CLI tables), uv (runner), ruff + mypy (lint/type).

Spec: `docs/superpowers/specs/2026-05-28-turnover-metric-design.md`

---

### Task 1: Core `annualized_turnover` function

**Files:**
- Create: `quant/backtest/activity.py`
- Test: `tests/backtest/test_activity.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/backtest/test_activity.py`:

```python
"""Tests for trade-activity metrics."""

from __future__ import annotations

import pandas as pd

from quant.backtest.activity import annualized_turnover


def _ledger(rows: list[tuple[int, float]]) -> pd.DataFrame:
    """Build a minimal trade ledger from (qty, fill_price) pairs."""
    return pd.DataFrame({"qty": [q for q, _ in rows], "fill_price": [p for _, p in rows]})


def _flat_equity(value: float, n_days: int) -> pd.Series:
    idx = pd.date_range("2020-01-01", periods=n_days, freq="B")
    return pd.Series([value] * n_days, index=idx, name="equity")


def test_hand_computed_value():
    # one buy + one sell of $1000 notional each -> two-way $2000, one-way $1000.
    # mean equity $10,000 over exactly one trading year (252d).
    # (1000 / 10000) * (252 / 252) = 0.10
    trades = _ledger([(100, 10.0), (100, 10.0)])
    equity = _flat_equity(10_000.0, 252)
    assert annualized_turnover(trades, equity) == 0.10


def test_full_roundtrip_reads_as_100pct():
    # buy $10,000 then sell $10,000 on a $10,000 book over one year -> 1.0
    trades = _ledger([(1000, 10.0), (1000, 10.0)])
    equity = _flat_equity(10_000.0, 252)
    assert annualized_turnover(trades, equity) == 1.0


def test_annualization_scales_with_window_length():
    # identical trades over half a year -> double the one-year figure.
    trades = _ledger([(100, 10.0), (100, 10.0)])
    assert annualized_turnover(trades, _flat_equity(10_000.0, 126)) == 0.20


def test_homogeneous_in_scale():
    # scaling notional and equity by the same factor leaves turnover unchanged.
    base = annualized_turnover(_ledger([(100, 10.0), (100, 10.0)]), _flat_equity(10_000.0, 252))
    scaled = annualized_turnover(_ledger([(100, 100.0), (100, 100.0)]), _flat_equity(100_000.0, 252))
    assert scaled == base


def test_empty_ledger_is_zero():
    assert annualized_turnover(pd.DataFrame(columns=["qty", "fill_price"]), _flat_equity(10_000.0, 252)) == 0.0


def test_empty_equity_is_zero():
    trades = _ledger([(100, 10.0)])
    assert annualized_turnover(trades, pd.Series(dtype=float)) == 0.0


def test_zero_mean_equity_is_zero():
    trades = _ledger([(100, 10.0)])
    assert annualized_turnover(trades, _flat_equity(0.0, 252)) == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/backtest/test_activity.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'quant.backtest.activity'`

- [ ] **Step 3: Write the module**

Create `quant/backtest/activity.py`:

```python
"""Trade-activity metrics computed from the backtest trade ledger.

Unlike ``metrics.py`` (every function maps a daily-returns Series -> float),
these take the trade ledger plus the equity curve, because turnover -- and,
later, capacity -- are properties of *trading activity*, not of the returns
stream. That different input shape is why this is a separate module.

Undefined results return 0.0 rather than raising, mirroring the ``metrics.py``
convention so tear-sheet rendering never breaks on edge cases.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_TRADING_DAYS_PER_YEAR = 252
_EQUITY_EPS = 1e-9


def annualized_turnover(
    trades: pd.DataFrame,
    equity_curve: pd.Series,
    periods_per_year: int = _TRADING_DAYS_PER_YEAR,
) -> float:
    """One-way, annualized portfolio turnover from the trade ledger.

    ``traded_notional = sum(|qty| * fill_price)`` over every fill;
    ``one_way = traded_notional / 2`` so a full round-trip reads as 100%;
    ``annualized = (one_way / mean_equity) * (periods_per_year / n_days)``.

    Uses actual fills (including slipped fill prices and zero-crossing
    flatten-and-reopen), not an idealized weight diff. ``trades`` must expose
    ``qty`` and ``fill_price`` columns (a ``BacktestResult.trades`` frame).
    Returns 0.0 when undefined (empty ledger, empty/zero-mean equity).
    """
    if trades is None or len(trades) == 0:
        return 0.0
    n_days = len(equity_curve)
    if n_days == 0:
        return 0.0
    mean_equity = float(equity_curve.mean())
    if not np.isfinite(mean_equity) or mean_equity <= _EQUITY_EPS:
        return 0.0
    traded_notional = float((trades["qty"].abs() * trades["fill_price"]).sum())
    one_way = traded_notional / 2.0
    return float((one_way / mean_equity) * (periods_per_year / n_days))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/backtest/test_activity.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add quant/backtest/activity.py tests/backtest/test_activity.py
git commit -m "feat(activity): fills-based one-way annualized turnover

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Export from the package

**Files:**
- Modify: `quant/backtest/__init__.py`

- [ ] **Step 1: Add the import and `__all__` entry**

In `quant/backtest/__init__.py`, add this import after the `engine` import (line 6):

```python
from quant.backtest.activity import annualized_turnover
```

And add `"annualized_turnover",` as the first entry of the `__all__` list (keep it alphabetical-ish; it can go right after the opening bracket):

```python
__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "CombinedResult",
    "WalkforwardResult",
    "WalkforwardWindow",
    "annualized_turnover",
    "iter_windows",
    "run_backtest",
    "run_combined_book",
    "run_walkforward",
    "select_best_params",
    "write_combined_tearsheet",
    "write_tearsheet",
]
```

- [ ] **Step 2: Verify the import resolves**

Run: `uv run python -c "from quant.backtest import annualized_turnover; print('ok')"`
Expected: prints `ok`

- [ ] **Step 3: Commit**

```bash
git add quant/backtest/__init__.py
git commit -m "feat(activity): export annualized_turnover from quant.backtest

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Wire turnover into `tearsheet.py` (dataclass + both build sites)

**Files:**
- Modify: `quant/backtest/tearsheet.py` (`_MetricsBundle` ~44-54, `write_tearsheet` ~275-285, `write_combined_tearsheet` ~371-406)

> Do all `tearsheet.py` edits in this one task/commit. `turnover` is a required field with no default, so the dataclass change and **both** `_MetricsBundle` construction sites must land together or the unedited site fails to construct.

- [ ] **Step 1: Add the import**

At the top of `tearsheet.py`, in the block importing from `quant.backtest.metrics` (the one importing `cagr, max_drawdown, sharpe, ...`), add a sibling import line:

```python
from quant.backtest.activity import annualized_turnover
```

- [ ] **Step 2: Add the field to `_MetricsBundle`**

Add `turnover: float` to the frozen dataclass (after `win_rate`):

```python
@dataclass(frozen=True)
class _MetricsBundle:
    total_return: float
    cagr: float
    sharpe: float
    sortino: float
    max_drawdown: float
    win_rate: float
    turnover: float
    n_trades: int
    starting_equity: float
    ending_equity: float
```

- [ ] **Step 3: Populate turnover in `write_tearsheet` (OOS)**

The `_MetricsBundle(...)` constructed around line 275 uses OOS data. Add the `turnover` kwarg using the OOS ledger + equity curve:

```python
    metrics = _MetricsBundle(
        total_return=total_return(result.oos_returns),
        cagr=cagr(result.oos_returns),
        sharpe=sharpe(result.oos_returns),
        sortino=sortino(result.oos_returns),
        max_drawdown=max_drawdown(result.oos_returns),
        win_rate=win_rate(result.oos_returns),
        turnover=annualized_turnover(result.oos_trades, result.oos_equity_curve),
        n_trades=len(result.oos_trades),
        starting_equity=float(result.combined_result.starting_equity),
        ending_equity=float(result.combined_result.ending_equity),
    )
```

- [ ] **Step 4: Populate turnover in `write_combined_tearsheet`**

Update the `_MetricsBundle(...)` around line 371 to pass `turnover` from the combined ledger + equity curve:

```python
    metrics = _MetricsBundle(
        total_return=total_return(result.returns),
        cagr=cagr(result.returns),
        sharpe=sharpe(result.returns),
        sortino=sortino(result.returns),
        max_drawdown=max_drawdown(result.returns),
        win_rate=win_rate(result.returns),
        turnover=annualized_turnover(result.trades, result.equity_curve),
        n_trades=len(result.trades),
        starting_equity=float(result.starting_equity),
        ending_equity=float(result.ending_equity),
    )
```

- [ ] **Step 5: Add turnover to each per-strategy row**

In the `per_strategy_rows.append({...})` block (around line 394), add a `"turnover"` key from that strategy's own ledger + equity curve:

```python
        per_strategy_rows.append(
            {
                "slug": slug,
                "allocation": result.allocation.get(slug, 0.0),
                "starting_equity": float(sub.starting_equity),
                "ending_equity": float(sub.ending_equity),
                "total_return": total_return(sub.returns),
                "sharpe": sharpe(sub.returns),
                "cagr": cagr(sub.returns),
                "max_drawdown": max_drawdown(sub.returns),
                "turnover": annualized_turnover(sub.trades, sub.equity_curve),
                "n_trades": len(sub.trades),
            }
        )
```

- [ ] **Step 6: Run the full backtest test group**

Run: `uv run pytest tests/backtest/ -v`
Expected: PASS (all backtest tests, including walk-forward and combined-book)

- [ ] **Step 7: Commit**

```bash
git add quant/backtest/tearsheet.py
git commit -m "feat(activity): turnover in _MetricsBundle + both tear-sheet build sites

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Show turnover in both HTML templates

**Files:**
- Modify: `quant/backtest/templates/tearsheet.html.j2` (headline metrics table ~32-40)
- Modify: `quant/backtest/templates/combined_tearsheet.html.j2` (headline ~35-39, per-strategy table ~63-78)

- [ ] **Step 1: Add a Turnover row to the walk-forward headline table**

In `tearsheet.html.j2`, after the `Win Rate (daily)` row (line 37), add:

```html
      <tr><td class="label">Turnover (ann., 1-way)</td><td class="value">{{ "{:.0%}".format(metrics.turnover) }}</td></tr>
```

- [ ] **Step 2: Add a Turnover row to the combined headline table**

In `combined_tearsheet.html.j2`, after the `Sharpe` row (line 35), add:

```html
      <tr><td class="label">Turnover (ann., 1-way)</td><td class="value">{{ "{:.0%}".format(metrics.turnover) }}</td></tr>
```

- [ ] **Step 3: Add a Turnover column to the per-strategy table**

In `combined_tearsheet.html.j2`, update the header row (line 64) to insert a `Turnover` header before `Trades`:

```html
          <th>Total Ret</th><th>Sharpe</th><th>CAGR</th><th>Max DD</th><th>Turnover</th><th>Trades</th>
```

Then in the per-strategy `{% for s in per_strategy %}` body, add a cell before the `n_trades` cell (line 78):

```html
            <td class="num">{{ "{:.0%}".format(s.turnover) }}</td>
            <td class="num">{{ s.n_trades }}</td>
```

- [ ] **Step 4: Verify templates still render**

Run: `uv run pytest tests/backtest/ -k "tearsheet or combined" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add quant/backtest/templates/tearsheet.html.j2 quant/backtest/templates/combined_tearsheet.html.j2
git commit -m "feat(activity): show annualized turnover in both tear-sheet templates

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Add a Turnover column to the `quant backtest` CLI table

**Files:**
- Modify: `quant/cli.py` (combined-book table ~225-251)

- [ ] **Step 1: Add the import**

Ensure `annualized_turnover` is importable in `cli.py`. If `cli.py` imports metric helpers from `quant.backtest.metrics`, add a sibling import near it:

```python
from quant.backtest.activity import annualized_turnover
```

(If `cli.py` imports from `quant.backtest` directly, use `from quant.backtest import annualized_turnover` instead — match the surrounding import style.)

- [ ] **Step 2: Add the column definition**

In the `Table(...)` setup (around line 225-231), add a Turnover column after `Max DD`:

```python
    table.add_column("Max DD", justify="right")
    table.add_column("Turnover", justify="right")
```

- [ ] **Step 3: Add the per-strategy cell**

In the `for slug in sorted(result.per_strategy):` loop (line 232), add the turnover cell to `table.add_row(...)` after the Max DD cell:

```python
        table.add_row(
            slug,
            f"{result.allocation.get(slug, 0):.1%}",
            f"${sub.ending_equity:,.0f}",
            f"{sharpe(sub.returns):.2f}",
            f"{cagr(sub.returns):.2%}",
            f"{max_drawdown(sub.returns):.2%}",
            f"{annualized_turnover(sub.trades, sub.equity_curve):.0%}",
        )
```

- [ ] **Step 4: Add the combined-row cell**

In the `table.add_row("[bold]COMBINED[/]", ...)` block (line 243), add the turnover cell after Max DD:

```python
    table.add_row(
        "[bold]COMBINED[/]",
        "100.0%",
        f"${result.ending_equity:,.0f}",
        f"{sharpe(result.returns):.2f}",
        f"{cagr(result.returns):.2%}",
        f"{max_drawdown(result.returns):.2%}",
        f"{annualized_turnover(result.trades, result.equity_curve):.0%}",
    )
```

- [ ] **Step 5: Verify the CLI imports and the module compiles**

Run: `uv run python -c "import quant.cli"`
Expected: no error (exit 0)

- [ ] **Step 6: Commit**

```bash
git add quant/cli.py
git commit -m "feat(activity): turnover column in the quant backtest CLI table

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Tear-sheet render assertion + full verification

**Files:**
- Modify: `tests/backtest/test_combined_book.py` (or the existing combined tear-sheet render test — locate with the grep below)

- [ ] **Step 1: Find the existing combined tear-sheet render test**

Run: `grep -rln "write_combined_tearsheet\|tearsheet.html" tests/backtest/`
Expected: identifies the test file that renders a combined tear-sheet (e.g. `test_combined_book.py`).

- [ ] **Step 2: Add a turnover render assertion**

In that test, after the tear-sheet HTML is written/read, assert the turnover label is present. Example (adapt the variable name for the read HTML string to the existing test):

```python
    html = (out_dir / "combined_tearsheet.html").read_text(encoding="utf-8")
    assert "Turnover" in html
```

If the existing test does not read the HTML back, add a minimal read of the returned HTML path and the assertion above.

- [ ] **Step 3: Run the new render assertion**

Run: `uv run pytest tests/backtest/ -k "combined" -v`
Expected: PASS

- [ ] **Step 4: Full verification — suite, types, lint, format**

Run each and confirm clean:

```bash
uv run pytest -q
uv run mypy quant
uv run ruff check quant tests
uv run ruff format --check quant tests
```

Expected: full suite passes (prior count 540 + 7 new = 547 region), mypy strict clean, ruff clean, format clean.

> **Note on the registry surface (spec "enumerate registry/CLI scalar surfaces"):** the run registry (`quant/research/registry.py`) logs only purpose-specific metrics dicts (`quant validate` logs `dsr`/`psr`; `sizing`/`hedge compare` log their own) — it does **not** log Sharpe/MaxDD as a general metrics dict that turnover should join. So turnover's scalar surfaces are exactly the two HTML tear-sheets (Task 4) and the `quant backtest` CLI table (Task 5). No registry change is needed.

- [ ] **Step 5: Commit**

```bash
git add tests/backtest/
git commit -m "test(activity): assert turnover renders in the combined tear-sheet

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Notes

- **No new validation gate.** Turnover is a *reported* metric (charter principle 3 says "report turnover"), not a gate. Do not add it to governance/validation gating.
- **Capacity is out of scope** — it lands with charter gap #2 (market-impact model). `activity.py` is its future home.
- **Push:** all commits are local. Do not push to public `origin/main` without explicit operator approval.
