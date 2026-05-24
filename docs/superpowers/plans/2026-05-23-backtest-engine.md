# Plan 2 of 6 — Backtest Engine + Walk-Forward + Tear-Sheet Pipeline

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the research pipeline that turns a `Strategy` subclass into an out-of-sample equity curve and a publishable tear-sheet — a clean daily-frequency simulator, a walk-forward orchestrator with parameter selection, and an HTML tear-sheet writer.

**Architecture:** Hand-rolled deterministic engine (no vectorbt). Day-by-day loop: rebalance on cadence → reconcile vs current → execute next bar with slippage/commission → mark-to-market. Walk-forward sits on top: split history into rolling train/test windows, grid-search params on train, stitch chosen-param backtests across test windows into one OOS curve. Tear-sheet wraps the OOS result with matplotlib charts and a key-metrics table into a single HTML file at `data/backtests/<slug>/tearsheet.html`.

**Tech Stack:** Python 3.12 (already in place), pandas, numpy, matplotlib, jinja2. Reuses `quant.strategies.base.Strategy`, `quant.data.bars.get_bars`, the existing CLI scaffold, and the test fixtures from Plan 1.

**Deviations from spec (explicit decisions):**

| Spec § | Spec says | Plan 2 says | Reason |
|---|---|---|---|
| 6 | `engine.py` uses `vectorbt` | Hand-rolled engine (~250 LOC) | Daily frequency + 5 strategies doesn't need vectorbt's vector machinery; hand-rolled is auditable, fully tested, no opaque math. |
| 4 / 6 | Tear-sheets via `quantstats` | matplotlib + jinja2 templating | `quantstats` is unmaintained, depends on yfinance + IPython transitively, and breaks on recent pandas. Custom gives clean control and a clear seam for Plan 3 panels (deflated Sharpe, regime breakdown, MC fans). |
| 4 | Combinatorial purged CV, deflated Sharpe, MC bootstrap, regime stress | Deferred to Plan 3 | Walk-forward + tear-sheet are independently useful and exercise the engine; the heavier validation layers compose on top. |

**Roadmap (informational):**
- Plan 1 (DONE): Foundation — repo, data layer, Alpaca client, CLI scaffold
- **Plan 2 (this one): Backtest engine + walk-forward + tear-sheet pipeline**
- Plan 3: Combinatorial purged CV + deflated Sharpe + bootstrap + regime stress
- Plan 4: Strategies 1–3 (refined ports from Quant Lab v1)
- Plan 5: Strategies 4–5 (net-new TSMOM + HRP)
- Plan 6: Textual TUI + Alpaca live paper execution + GitHub Actions wiring

---

## Prerequisites

Before Task 1, confirm:

1. Foundation (Plan 1) is on `main`. `uv run quant --help` works end-to-end.
2. `.env` contains valid Alpaca paper keys and a FRED key (needed for the smoke-test `quant data refresh` run at end of plan). `data refresh` calls will populate `data/raw/*.parquet` with real Alpaca bars in Task 8's smoke step.
3. `uv sync --all-extras` works clean.
4. `uv run pytest -q -m "not network and not alpaca and not slow"` is green (60 tests passing in baseline).

All shell commands assume CWD = `~/Documents/quant-trading`. All work happens on a feature branch.

```bash
git checkout main && git pull
git checkout -b feat/backtest-engine
```

---

## File Structure

Files created or modified by this plan, with each file's single responsibility:

```
quant-trading/
├── pyproject.toml                                ← MODIFY: add matplotlib + jinja2 deps
├── quant/
│   ├── strategies/
│   │   └── base.py                               ← MODIFY: add params + default_params
│   ├── backtest/                                 ← NEW package
│   │   ├── __init__.py                           ← re-exports run_backtest, run_walkforward, write_tearsheet
│   │   ├── engine.py                             ← BacktestConfig, BacktestResult, run_backtest
│   │   ├── calendar.py                           ← rebalance-day predicate (daily/weekly/monthly)
│   │   ├── walkforward.py                        ← iter_windows, select_best_params, run_walkforward
│   │   ├── metrics.py                            ← sharpe, sortino, max_dd, cagr, win_rate, total_return
│   │   ├── tearsheet.py                          ← write_tearsheet(): matplotlib charts → base64 → HTML
│   │   └── templates/
│   │       └── tearsheet.html.j2                 ← jinja2 template for the tear-sheet
│   ├── data/
│   │   └── refresh.py                            ← NEW: refresh_universe(symbols, start, end) helper
│   └── cli.py                                    ← MODIFY: wire backtest / tearsheet / data refresh
└── tests/
    ├── conftest.py                               ← MODIFY: add synthetic_bars + EqualWeight test strategy
    ├── strategies/
    │   └── test_strategy_params.py               ← NEW
    ├── backtest/                                 ← NEW package
    │   ├── __init__.py
    │   ├── test_calendar.py
    │   ├── test_metrics.py
    │   ├── test_engine_costs.py
    │   ├── test_engine_run.py
    │   ├── test_walkforward_windows.py
    │   ├── test_walkforward_selection.py
    │   ├── test_walkforward_run.py
    │   └── test_tearsheet.py
    ├── data/
    │   └── test_refresh.py                       ← NEW
    └── test_cli.py                               ← MODIFY: cover backtest / tearsheet / data refresh
```

**Module responsibilities (locked):**

- `engine.py` — deterministic single-pass simulator. Inputs: a `Strategy` instance, a wide bars DataFrame, a `BacktestConfig`, a start/end window. Outputs: `BacktestResult` (equity curve, daily returns, positions, trades). No randomness, no grid search, no walk-forward awareness.
- `calendar.py` — `is_rebalance_day(asof, frequency, history)` predicate. Daily = every bar; weekly = first bar of the ISO week; monthly = first bar of the calendar month. Pure function, no state.
- `metrics.py` — pure-numerics functions on a returns Series. `sharpe`, `sortino`, `max_drawdown`, `cagr`, `total_return`, `win_rate`. All take a `pd.Series` of daily returns and a periods-per-year argument (default 252).
- `walkforward.py` — orchestrator. `iter_windows` is a pure generator; `select_best_params` grid-searches on a train window using the engine; `run_walkforward` stitches per-window selections into a single OOS curve.
- `tearsheet.py` — pure-output: takes a walk-forward result and writes `tearsheet.html`, `walkforward.parquet`, `chosen_params.json` under `data/backtests/<slug>/`.
- `data/refresh.py` — populates the bar cache for a symbol list. Wraps `get_bars` so the CLI command stays thin.

---

## Task 1: Strategy parameter support

**Files:**
- Modify: `quant/strategies/base.py`
- Create: `tests/strategies/test_strategy_params.py`

**Context.** Plan 4/5 strategies will be parameterized (lookbacks, thresholds, etc.). The walk-forward harness needs to be able to instantiate a strategy with `params={"lookback": 12}` etc. Foundation's `Strategy` is parameterless — we add `default_params` (class-level) + a `params` instance dict that merges over defaults. This is the smallest change that lets walk-forward work.

- [ ] **Step 1: Write failing test for `Strategy.__init__` parameter merging**

`tests/strategies/test_strategy_params.py`:

```python
"""Strategy base class parameter merging."""

from __future__ import annotations

from datetime import date
from typing import ClassVar

import pandas as pd

from quant.strategies.base import Strategy, StrategySpec


class _ToyStrategy(Strategy):
    spec: ClassVar[StrategySpec] = StrategySpec(
        slug="toy",
        name="Toy",
        description="Toy strategy used in tests.",
        universe=["AAPL"],
        rebalance_frequency="daily",
    )
    default_params: ClassVar[dict[str, object]] = {"lookback": 10, "scale": 1.0}

    def generate_signals(self, asof: date) -> pd.Series:
        return pd.Series({"AAPL": 1.0})

    def target_positions(self, asof: date, equity: float) -> dict[str, int]:
        return {"AAPL": 1}


def test_default_params_used_when_none_passed() -> None:
    s = _ToyStrategy()
    assert s.params == {"lookback": 10, "scale": 1.0}


def test_params_override_defaults() -> None:
    s = _ToyStrategy(params={"lookback": 20})
    assert s.params == {"lookback": 20, "scale": 1.0}


def test_extra_params_pass_through() -> None:
    s = _ToyStrategy(params={"new_knob": True})
    assert s.params["new_knob"] is True
    assert s.params["lookback"] == 10


def test_default_params_not_mutated_by_instance() -> None:
    s = _ToyStrategy(params={"lookback": 99})
    assert s.params["lookback"] == 99
    assert _ToyStrategy.default_params["lookback"] == 10


def test_strategy_without_default_params_works() -> None:
    """A Strategy subclass that doesn't declare default_params still instantiates."""

    class _NoDefaults(Strategy):
        spec: ClassVar[StrategySpec] = StrategySpec(
            slug="no-defaults",
            name="No Defaults",
            description="-",
            universe=["AAPL"],
            rebalance_frequency="daily",
        )

        def generate_signals(self, asof: date) -> pd.Series:
            return pd.Series(dtype=float)

        def target_positions(self, asof: date, equity: float) -> dict[str, int]:
            return {}

    s = _NoDefaults()
    assert s.params == {}
```

- [ ] **Step 2: Run test to verify failure**

```bash
uv run pytest tests/strategies/test_strategy_params.py -v
```

Expected: ImportError or AttributeError on `default_params` / `params`.

- [ ] **Step 3: Implement parameter support in `Strategy` base**

Modify `quant/strategies/base.py`:

```python
"""Strategy ABC + StrategySpec dataclass.

Concrete strategies land in Plans 4 and 5. Foundation only needs the contract.
Plan 2 adds parameter support so walk-forward can grid-search.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from typing import Any, ClassVar

import pandas as pd


@dataclass(frozen=True)
class StrategySpec:
    """Static metadata about a strategy."""

    slug: str
    name: str
    description: str
    universe: list[str]
    rebalance_frequency: str  # "daily" | "weekly" | "monthly"
    enabled_live: bool = field(default=False)


class Strategy(ABC):
    """Base class for all strategies. Concrete strategies subclass and register."""

    spec: ClassVar[StrategySpec]
    default_params: ClassVar[dict[str, Any]] = {}

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        merged: dict[str, Any] = dict(self.default_params)
        if params:
            merged.update(params)
        self.params: dict[str, Any] = merged

    @abstractmethod
    def generate_signals(self, asof: date) -> pd.Series:
        """Return a Series indexed by symbol with the signal score for each name."""

    @abstractmethod
    def target_positions(self, asof: date, equity: float) -> dict[str, int]:
        """Return target whole-share positions keyed by symbol.

        Positive = long, negative = short, missing/zero = no position.
        """
```

- [ ] **Step 4: Run test to verify pass**

```bash
uv run pytest tests/strategies/test_strategy_params.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Run full suite + lint to confirm no regressions**

```bash
uv run pytest -q -m "not network and not alpaca and not slow"
uv run ruff check . && uv run ruff format --check . && uv run mypy quant
```

Expected: 65 tests pass (60 prior + 5 new), lint and types clean.

- [ ] **Step 6: Commit**

```bash
git add quant/strategies/base.py tests/strategies/test_strategy_params.py
git commit -m "feat(strategies): parameter support on Strategy base"
```

---

## Task 2: Synthetic-bars fixture + EqualWeight test strategy

**Files:**
- Modify: `tests/conftest.py`
- Create: `tests/backtest/__init__.py` (empty)

**Context.** The engine needs deterministic test inputs. We add two reusable fixtures: `synthetic_bars(...)` for wide-format MultiIndex bars, and `EqualWeightStrategy` — a test-only strategy that allocates capital uniformly across its universe. Both live in tests/ so they don't leak into the production registry.

- [ ] **Step 1: Write failing tests for the new fixtures**

Append to `tests/conftest.py`:

```python
"""Shared pytest fixtures and configuration."""

# ... existing imports + fixtures unchanged ...

from datetime import date
from typing import ClassVar

import numpy as np
import pandas as pd

from quant.strategies.base import Strategy, StrategySpec


def synthetic_bars(
    symbols: list[str],
    start: date,
    end: date,
    *,
    seed: int = 0,
    drift: float = 0.0003,
    vol: float = 0.01,
    start_price: float = 100.0,
) -> pd.DataFrame:
    """Generate deterministic wide-format daily bars for [start, end] business days.

    Returns a DataFrame indexed by date with MultiIndex columns (symbol, field)
    where field ∈ {open, high, low, close, volume}. Prices follow a geometric
    random walk; high/low are bracketed around close.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(pd.Timestamp(start), pd.Timestamp(end))
    if len(dates) == 0:
        return pd.DataFrame()
    frames: dict[str, pd.DataFrame] = {}
    for i, sym in enumerate(symbols):
        # Distinct seed per symbol so they're not perfectly correlated.
        shocks = rng.normal(loc=drift, scale=vol, size=len(dates))
        closes = start_price * np.exp(np.cumsum(shocks))
        opens = np.r_[closes[:1], closes[:-1]]  # open = prior close
        highs = np.maximum(opens, closes) * (1.0 + np.abs(rng.normal(0, vol / 4, len(dates))))
        lows = np.minimum(opens, closes) * (1.0 - np.abs(rng.normal(0, vol / 4, len(dates))))
        df = pd.DataFrame(
            {
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": np.full(len(dates), 1_000_000 + i, dtype=np.int64),
            },
            index=dates,
        )
        df.index.name = "timestamp"
        frames[sym] = df
    return pd.concat(frames, axis=1)


class EqualWeightStrategy(Strategy):
    """Test-only strategy: split equity equally across its universe at each rebalance."""

    spec: ClassVar[StrategySpec] = StrategySpec(
        slug="equal-weight-test",
        name="Equal Weight (test)",
        description="Test fixture: uniform allocation across the configured universe.",
        universe=["AAA", "BBB"],
        rebalance_frequency="monthly",
    )
    default_params: ClassVar[dict[str, object]] = {}

    def __init__(
        self,
        bars: pd.DataFrame,
        params: dict[str, object] | None = None,
        universe: list[str] | None = None,
    ) -> None:
        super().__init__(params=params)
        self._bars = bars
        self._universe = universe or list(self.spec.universe)

    def generate_signals(self, asof: date) -> pd.Series:
        return pd.Series({sym: 1.0 for sym in self._universe})

    def target_positions(self, asof: date, equity: float) -> dict[str, int]:
        ts = pd.Timestamp(asof)
        if ts not in self._bars.index:
            return {}
        per_name = equity / max(len(self._universe), 1)
        out: dict[str, int] = {}
        for sym in self._universe:
            price = float(self._bars[(sym, "close")].loc[ts])
            if price <= 0:
                continue
            out[sym] = int(per_name // price)
        return out


@pytest.fixture
def make_bars() -> Callable[..., pd.DataFrame]:
    """Factory fixture: tests call make_bars(symbols, start, end, seed=...) to get bars."""
    return synthetic_bars


@pytest.fixture
def equal_weight_strategy(
    make_bars: Callable[..., pd.DataFrame],
) -> tuple[EqualWeightStrategy, pd.DataFrame]:
    """A 2-symbol EqualWeight strategy + matching synthetic bars for a 1-year window."""
    bars = make_bars(["AAA", "BBB"], date(2024, 1, 1), date(2024, 12, 31), seed=42)
    strat = EqualWeightStrategy(bars=bars)
    return strat, bars
```

At the top of `tests/conftest.py`, add `from collections.abc import Callable` if not already there.

Create empty marker file: `tests/backtest/__init__.py`.

Also write a small sanity test that exercises the fixture. Create `tests/backtest/test_fixtures.py`:

```python
"""Sanity tests for the synthetic_bars fixture and EqualWeightStrategy."""

from __future__ import annotations

from datetime import date

import pandas as pd

from tests.conftest import EqualWeightStrategy, synthetic_bars


def test_synthetic_bars_shape() -> None:
    bars = synthetic_bars(["AAA", "BBB"], date(2024, 1, 1), date(2024, 1, 31))
    assert isinstance(bars.index, pd.DatetimeIndex)
    assert set(bars.columns.get_level_values(0)) == {"AAA", "BBB"}
    assert "close" in bars.columns.get_level_values(1)
    assert len(bars) > 20  # 22-ish business days in Jan 2024


def test_synthetic_bars_deterministic() -> None:
    a = synthetic_bars(["AAA"], date(2024, 1, 1), date(2024, 1, 31), seed=7)
    b = synthetic_bars(["AAA"], date(2024, 1, 1), date(2024, 1, 31), seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_synthetic_bars_different_seeds_differ() -> None:
    a = synthetic_bars(["AAA"], date(2024, 1, 1), date(2024, 1, 31), seed=1)
    b = synthetic_bars(["AAA"], date(2024, 1, 1), date(2024, 1, 31), seed=2)
    assert not a.equals(b)


def test_equal_weight_target_positions() -> None:
    bars = synthetic_bars(["AAA", "BBB"], date(2024, 1, 1), date(2024, 1, 31), seed=0)
    strat = EqualWeightStrategy(bars=bars)
    target = strat.target_positions(date(2024, 1, 5), equity=100_000.0)
    assert set(target.keys()) <= {"AAA", "BBB"}
    assert all(qty > 0 for qty in target.values())
```

- [ ] **Step 2: Run tests to verify failure**

```bash
uv run pytest tests/backtest/test_fixtures.py -v
```

Expected: ImportError on `EqualWeightStrategy` / `synthetic_bars` — fixtures don't exist yet.

- [ ] **Step 3: Apply the conftest.py changes shown in Step 1**

(They're already in Step 1; apply them and the new `tests/backtest/__init__.py`.)

- [ ] **Step 4: Run tests to verify pass**

```bash
uv run pytest tests/backtest/test_fixtures.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Confirm full suite + lint clean**

```bash
uv run pytest -q -m "not network and not alpaca and not slow"
uv run ruff check tests && uv run ruff format --check tests
```

Expected: 69 tests pass; lint clean. (mypy excludes tests, so no mypy run needed here.)

- [ ] **Step 6: Commit**

```bash
git add tests/conftest.py tests/backtest/__init__.py tests/backtest/test_fixtures.py
git commit -m "test: synthetic_bars fixture + EqualWeightStrategy test helper"
```

---

## Task 3: Rebalance-day calendar predicate

**Files:**
- Create: `quant/backtest/__init__.py`
- Create: `quant/backtest/calendar.py`
- Create: `tests/backtest/test_calendar.py`

**Context.** The engine needs to know, for each bar, whether the strategy should rebalance. `daily` → every bar; `weekly` → first bar of each ISO week in `history`; `monthly` → first bar of each calendar month in `history`. Pure function, easy to test.

- [ ] **Step 1: Write failing tests**

`tests/backtest/test_calendar.py`:

```python
"""Tests for quant.backtest.calendar.is_rebalance_day."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from quant.backtest.calendar import is_rebalance_day


@pytest.fixture
def history_jan2024() -> pd.DatetimeIndex:
    """Business days in January 2024 — 22 bars."""
    return pd.bdate_range("2024-01-01", "2024-01-31")


def test_daily_always_rebalances(history_jan2024: pd.DatetimeIndex) -> None:
    for ts in history_jan2024:
        assert is_rebalance_day(ts.date(), "daily", history_jan2024) is True


def test_weekly_only_first_business_day_of_iso_week(history_jan2024: pd.DatetimeIndex) -> None:
    # Jan 2024 ISO weeks: w1 (Mon Jan 1), w2 (Mon Jan 8), w3 (Mon Jan 15),
    # w4 (Mon Jan 22), w5 (Mon Jan 29). Each Monday is a business day in Jan 2024.
    expected_mondays = {date(2024, 1, 1), date(2024, 1, 8), date(2024, 1, 15),
                        date(2024, 1, 22), date(2024, 1, 29)}
    got = {ts.date() for ts in history_jan2024 if is_rebalance_day(ts.date(), "weekly", history_jan2024)}
    assert got == expected_mondays


def test_weekly_handles_holiday_kick(history_jan2024: pd.DatetimeIndex) -> None:
    """If the ISO Monday is a holiday and missing from history, the next bar is the rebalance."""
    # Drop Jan 1, 2024 (assume holiday): first bar of that ISO week becomes Tue Jan 2.
    history = history_jan2024.drop(pd.Timestamp("2024-01-01"))
    got = [ts.date() for ts in history if is_rebalance_day(ts.date(), "weekly", history)]
    assert date(2024, 1, 2) in got
    assert date(2024, 1, 1) not in got


def test_monthly_only_first_business_day_of_month() -> None:
    history = pd.bdate_range("2024-01-01", "2024-03-31")
    rebalances = [ts.date() for ts in history if is_rebalance_day(ts.date(), "monthly", history)]
    assert rebalances == [date(2024, 1, 1), date(2024, 2, 1), date(2024, 3, 1)]


def test_invalid_frequency_raises(history_jan2024: pd.DatetimeIndex) -> None:
    with pytest.raises(ValueError, match="frequency"):
        is_rebalance_day(date(2024, 1, 2), "annually", history_jan2024)
```

- [ ] **Step 2: Run tests to verify failure**

```bash
uv run pytest tests/backtest/test_calendar.py -v
```

Expected: ImportError on `quant.backtest.calendar`.

- [ ] **Step 3: Implement calendar predicate**

Create `quant/backtest/__init__.py`:

```python
"""Backtest engine, walk-forward harness, and tear-sheet generator."""

from __future__ import annotations

from quant.backtest.engine import BacktestConfig, BacktestResult, run_backtest
from quant.backtest.tearsheet import write_tearsheet
from quant.backtest.walkforward import (
    WalkforwardResult,
    iter_windows,
    run_walkforward,
    select_best_params,
)

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "WalkforwardResult",
    "iter_windows",
    "run_backtest",
    "run_walkforward",
    "select_best_params",
    "write_tearsheet",
]
```

Note: this `__init__.py` references symbols implemented in later tasks. To avoid import errors during this task, write a temporary stub form first — but then keep this as the final shape, and don't import this module from anywhere until Task 6+ symbols exist. Better: defer the full `__init__.py` to Task 6. For now, write the minimal form:

```python
"""Backtest engine, walk-forward harness, and tear-sheet generator."""

from __future__ import annotations
```

We'll grow `__init__.py` at the end (Task 9).

Create `quant/backtest/calendar.py`:

```python
"""Predicate: should the strategy rebalance on this bar?"""

from __future__ import annotations

from datetime import date

import pandas as pd

_VALID_FREQUENCIES = {"daily", "weekly", "monthly"}


def is_rebalance_day(
    asof: date,
    frequency: str,
    history: pd.DatetimeIndex,
) -> bool:
    """Return True if `asof` is a rebalance day for `frequency` given `history`.

    `history` is the full index of bars the engine is iterating over. For weekly
    and monthly frequencies, the first bar of each ISO week / calendar month in
    `history` is the rebalance day — this correctly handles holiday-shifted
    Mondays / month-starts without hard-coding a calendar.
    """
    if frequency not in _VALID_FREQUENCIES:
        raise ValueError(
            f"Unknown frequency {frequency!r}; expected one of {sorted(_VALID_FREQUENCIES)}"
        )
    ts = pd.Timestamp(asof)
    if ts not in history:
        return False
    if frequency == "daily":
        return True

    if frequency == "weekly":
        key = (ts.isocalendar().year, ts.isocalendar().week)
        same_week = [
            t for t in history if (t.isocalendar().year, t.isocalendar().week) == key
        ]
        return bool(same_week and same_week[0] == ts)

    # monthly
    same_month = [t for t in history if t.year == ts.year and t.month == ts.month]
    return bool(same_month and same_month[0] == ts)
```

- [ ] **Step 4: Run tests to verify pass**

```bash
uv run pytest tests/backtest/test_calendar.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Lint + types**

```bash
uv run ruff check quant tests && uv run ruff format --check quant tests
uv run mypy quant
```

Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add quant/backtest/__init__.py quant/backtest/calendar.py tests/backtest/test_calendar.py
git commit -m "feat(backtest): rebalance-day calendar predicate"
```

---

## Task 4: Performance metrics

**Files:**
- Create: `quant/backtest/metrics.py`
- Create: `tests/backtest/test_metrics.py`

**Context.** Pure functions on a returns Series. Used by the engine (computes summary metrics on its output), walk-forward (picks best params on Sharpe), and the tear-sheet (top-of-page metrics table). One file, simple, easy to verify.

- [ ] **Step 1: Write failing tests**

`tests/backtest/test_metrics.py`:

```python
"""Tests for quant.backtest.metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.backtest.metrics import (
    cagr,
    max_drawdown,
    sharpe,
    sortino,
    total_return,
    win_rate,
)


@pytest.fixture
def flat_zero_returns() -> pd.Series:
    return pd.Series(np.zeros(252), index=pd.bdate_range("2024-01-01", periods=252))


@pytest.fixture
def constant_positive_returns() -> pd.Series:
    """Constant +0.1% / day for 252 trading days → ~+28.6% annual, vol=0."""
    return pd.Series(np.full(252, 0.001), index=pd.bdate_range("2024-01-01", periods=252))


@pytest.fixture
def alternating_returns() -> pd.Series:
    """+1%, -1%, +1%, -1%, ... — for win-rate and drawdown tests."""
    vals = np.array([0.01, -0.01] * 126)
    return pd.Series(vals, index=pd.bdate_range("2024-01-01", periods=252))


def test_total_return_zero(flat_zero_returns: pd.Series) -> None:
    assert total_return(flat_zero_returns) == pytest.approx(0.0)


def test_total_return_compounds(constant_positive_returns: pd.Series) -> None:
    expected = (1.001 ** 252) - 1
    assert total_return(constant_positive_returns) == pytest.approx(expected, rel=1e-6)


def test_cagr_handles_subyear(constant_positive_returns: pd.Series) -> None:
    # 252 trading days ≈ 1 calendar year of returns → CAGR ≈ total return.
    assert cagr(constant_positive_returns) == pytest.approx(total_return(constant_positive_returns), rel=1e-2)


def test_sharpe_zero_vol_returns_zero(constant_positive_returns: pd.Series) -> None:
    # Sharpe undefined when vol == 0; we return 0.0 by convention.
    assert sharpe(constant_positive_returns) == 0.0


def test_sharpe_positive_for_positive_mean() -> None:
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0005, 0.01, 252), index=pd.bdate_range("2024-01-01", periods=252))
    assert sharpe(r) > 0


def test_sortino_only_penalizes_downside() -> None:
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0005, 0.01, 252), index=pd.bdate_range("2024-01-01", periods=252))
    # Sortino vs Sharpe: same numerator, downside-only denominator → larger or equal.
    assert sortino(r) >= sharpe(r) - 1e-9


def test_max_drawdown_simple(alternating_returns: pd.Series) -> None:
    # After +1% then -1%, equity = 1.01 * 0.99 = 0.9999 → DD ≈ 1.01 → 0.9999 ≈ -0.0099
    dd = max_drawdown(alternating_returns)
    assert dd < 0
    assert dd > -0.5  # Sanity bound — won't be huge for this series


def test_max_drawdown_zero_for_monotone_up(constant_positive_returns: pd.Series) -> None:
    # All-positive returns → equity is monotone-increasing → max DD = 0.
    assert max_drawdown(constant_positive_returns) == pytest.approx(0.0, abs=1e-9)


def test_win_rate(alternating_returns: pd.Series) -> None:
    assert win_rate(alternating_returns) == pytest.approx(0.5)


def test_win_rate_excludes_zero_returns(flat_zero_returns: pd.Series) -> None:
    # Convention: zero returns are excluded from the denominator → undefined → 0.0.
    assert win_rate(flat_zero_returns) == 0.0


def test_metrics_handle_empty_series() -> None:
    empty = pd.Series(dtype=float)
    assert total_return(empty) == 0.0
    assert sharpe(empty) == 0.0
    assert sortino(empty) == 0.0
    assert max_drawdown(empty) == 0.0
    assert cagr(empty) == 0.0
    assert win_rate(empty) == 0.0
```

- [ ] **Step 2: Run tests to verify failure**

```bash
uv run pytest tests/backtest/test_metrics.py -v
```

Expected: ImportError on `quant.backtest.metrics`.

- [ ] **Step 3: Implement metrics**

`quant/backtest/metrics.py`:

```python
"""Performance metrics on a daily-returns Series.

All functions take a pd.Series of daily simple returns and return a float.
By convention, undefined results (empty input, zero vol, no wins) return 0.0
rather than NaN, so tear-sheet rendering never breaks on edge cases.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_TRADING_DAYS_PER_YEAR = 252


def total_return(returns: pd.Series) -> float:
    if len(returns) == 0:
        return 0.0
    return float((1.0 + returns).prod() - 1.0)


def cagr(returns: pd.Series, periods_per_year: int = _TRADING_DAYS_PER_YEAR) -> float:
    if len(returns) == 0:
        return 0.0
    tr = total_return(returns)
    years = len(returns) / periods_per_year
    if years <= 0 or tr <= -1.0:
        return 0.0
    return float((1.0 + tr) ** (1.0 / years) - 1.0)


def sharpe(returns: pd.Series, periods_per_year: int = _TRADING_DAYS_PER_YEAR) -> float:
    if len(returns) == 0:
        return 0.0
    std = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    if std == 0.0:
        return 0.0
    mean = float(returns.mean())
    return float(mean / std * np.sqrt(periods_per_year))


def sortino(returns: pd.Series, periods_per_year: int = _TRADING_DAYS_PER_YEAR) -> float:
    if len(returns) == 0:
        return 0.0
    downside = returns[returns < 0]
    if len(downside) == 0:
        return 0.0
    dd_std = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    if dd_std == 0.0:
        return 0.0
    mean = float(returns.mean())
    return float(mean / dd_std * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    """Return the worst peak-to-trough drawdown as a negative number."""
    if len(returns) == 0:
        return 0.0
    equity = (1.0 + returns).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def win_rate(returns: pd.Series) -> float:
    """Fraction of strictly-positive returns among non-zero returns."""
    if len(returns) == 0:
        return 0.0
    nonzero = returns[returns != 0.0]
    if len(nonzero) == 0:
        return 0.0
    return float((nonzero > 0).mean())
```

- [ ] **Step 4: Run tests to verify pass**

```bash
uv run pytest tests/backtest/test_metrics.py -v
```

Expected: 12 passed.

- [ ] **Step 5: Lint + types**

```bash
uv run ruff check quant tests && uv run ruff format --check quant tests && uv run mypy quant
```

Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add quant/backtest/metrics.py tests/backtest/test_metrics.py
git commit -m "feat(backtest): performance metrics module"
```

---

## Task 5: Backtest engine — cost model

**Files:**
- Create: `quant/backtest/engine.py` (partial — just the cost helper and dataclasses)
- Create: `tests/backtest/test_engine_costs.py`

**Context.** Before the loop, lock down the cost model in isolation. `apply_costs(order_qty, fill_price, config)` returns `(slippage_cost, commission_cost)` as positive dollars to subtract from cash. Slippage moves price against the trade direction (buys fill higher, sells fill lower). Commission is bps of notional.

- [ ] **Step 1: Write failing tests**

`tests/backtest/test_engine_costs.py`:

```python
"""Tests for the engine cost model."""

from __future__ import annotations

import pytest

from quant.backtest.engine import BacktestConfig, apply_costs


def test_buy_slippage_raises_fill_price() -> None:
    cfg = BacktestConfig(slippage_bps=10.0, commission_bps=0.0)  # 10 bps = 0.10%
    fill = apply_costs(qty=100, mid_price=50.0, side="buy", config=cfg)
    # Buy is hit up by 10 bps: 50.00 * 1.001 = 50.05
    assert fill.fill_price == pytest.approx(50.05, abs=1e-6)
    assert fill.slippage_cost == pytest.approx(100 * (50.05 - 50.0), abs=1e-6)
    assert fill.commission_cost == 0.0


def test_sell_slippage_lowers_fill_price() -> None:
    cfg = BacktestConfig(slippage_bps=10.0, commission_bps=0.0)
    fill = apply_costs(qty=100, mid_price=50.0, side="sell", config=cfg)
    # Sell is hit down by 10 bps: 50.00 * 0.999 = 49.95
    assert fill.fill_price == pytest.approx(49.95, abs=1e-6)
    # Slippage cost is measured as cash lost vs mid: (50.0 - 49.95) * 100
    assert fill.slippage_cost == pytest.approx(100 * (50.0 - 49.95), abs=1e-6)


def test_commission_is_bps_of_notional() -> None:
    cfg = BacktestConfig(slippage_bps=0.0, commission_bps=5.0)  # 5 bps = 0.05%
    fill = apply_costs(qty=100, mid_price=50.0, side="buy", config=cfg)
    notional = 100 * fill.fill_price
    assert fill.commission_cost == pytest.approx(notional * 0.0005, abs=1e-6)


def test_zero_costs_returns_mid() -> None:
    cfg = BacktestConfig(slippage_bps=0.0, commission_bps=0.0)
    fill = apply_costs(qty=100, mid_price=42.0, side="buy", config=cfg)
    assert fill.fill_price == 42.0
    assert fill.slippage_cost == 0.0
    assert fill.commission_cost == 0.0


def test_invalid_side_raises() -> None:
    cfg = BacktestConfig()
    with pytest.raises(ValueError):
        apply_costs(qty=10, mid_price=50.0, side="hold", config=cfg)  # type: ignore[arg-type]


def test_zero_qty_costs_zero() -> None:
    cfg = BacktestConfig(slippage_bps=10.0, commission_bps=10.0)
    fill = apply_costs(qty=0, mid_price=50.0, side="buy", config=cfg)
    assert fill.slippage_cost == 0.0
    assert fill.commission_cost == 0.0


def test_default_config_values() -> None:
    cfg = BacktestConfig()
    assert cfg.starting_equity == 100_000.0
    assert cfg.slippage_bps == 5.0
    assert cfg.commission_bps == 0.0
    assert cfg.execution == "next_open"
```

- [ ] **Step 2: Run tests to verify failure**

```bash
uv run pytest tests/backtest/test_engine_costs.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement engine.py (partial — config + cost model only)**

`quant/backtest/engine.py`:

```python
"""Backtest engine.

Daily-frequency, deterministic, single-pass. At each rebalance day the strategy
proposes target positions; the engine reconciles vs current and executes the
diff on the next bar (or the same bar's close, depending on config). Slippage
and commission are charged per trade. Equity is marked to market on every bar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal

import pandas as pd

Side = Literal["buy", "sell"]


@dataclass(frozen=True)
class BacktestConfig:
    """Engine configuration. All defaults are intentional — change with care."""

    starting_equity: float = 100_000.0
    slippage_bps: float = 5.0
    commission_bps: float = 0.0
    execution: Literal["next_open", "close"] = "next_open"


@dataclass(frozen=True)
class FillReport:
    """The result of applying costs to a single order."""

    fill_price: float
    slippage_cost: float
    commission_cost: float


@dataclass(frozen=True)
class BacktestResult:
    """Output of run_backtest."""

    equity_curve: pd.Series          # daily, indexed by date
    returns: pd.Series                # daily simple returns, indexed by date
    positions: pd.DataFrame           # rows=date, cols=symbol, values=shares
    trades: pd.DataFrame              # columns: date, symbol, side, qty, fill_price, slippage_cost, commission_cost, strategy_slug
    config: BacktestConfig
    starting_equity: float
    ending_equity: float
    metadata: dict[str, object] = field(default_factory=dict)


def apply_costs(qty: int, mid_price: float, side: Side, config: BacktestConfig) -> FillReport:
    """Move the mid-price by slippage and compute commission as bps of notional.

    Buy: fill_price = mid * (1 + slippage_bps / 1e4)
    Sell: fill_price = mid * (1 - slippage_bps / 1e4)
    Commission: |qty| * fill_price * commission_bps / 1e4
    """
    if side not in ("buy", "sell"):
        raise ValueError(f"Unknown side {side!r}; expected 'buy' or 'sell'")
    if qty == 0:
        return FillReport(fill_price=mid_price, slippage_cost=0.0, commission_cost=0.0)

    slip = config.slippage_bps / 1e4
    sign = +1.0 if side == "buy" else -1.0
    fill_price = mid_price * (1.0 + sign * slip)
    slippage_cost = abs(qty) * abs(fill_price - mid_price)
    commission_cost = abs(qty) * fill_price * config.commission_bps / 1e4
    return FillReport(
        fill_price=fill_price,
        slippage_cost=slippage_cost,
        commission_cost=commission_cost,
    )


# Forward declaration so the package __init__ can import the name even before
# the run loop lands in Task 6. The Task-6 commit replaces this stub.
def run_backtest(*args: object, **kwargs: object) -> BacktestResult:  # pragma: no cover
    raise NotImplementedError("run_backtest implemented in Task 6")
```

- [ ] **Step 4: Run tests to verify pass**

```bash
uv run pytest tests/backtest/test_engine_costs.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Lint + types**

```bash
uv run ruff check quant tests && uv run ruff format --check quant tests && uv run mypy quant
```

Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add quant/backtest/engine.py tests/backtest/test_engine_costs.py
git commit -m "feat(backtest): engine config + cost model"
```

---

## Task 6: Backtest engine — run loop

**Files:**
- Modify: `quant/backtest/engine.py`
- Create: `tests/backtest/test_engine_run.py`

**Context.** With the cost model in place, the run loop is straightforward. At each bar:

1. Determine current asof = bar.date.
2. If `is_rebalance_day(asof, strategy.spec.rebalance_frequency, history)`, call `strategy.target_positions(asof, current_equity)`. Diff vs current → orders.
3. Schedule each order to fill at `config.execution`:
   - `close` → fill at today's close, recorded under today's date.
   - `next_open` → fill at tomorrow's open, recorded under tomorrow's date. If there is no next bar, drop the order (end of history).
4. On the fill bar, apply costs, debit/credit cash, update positions.
5. After fills are applied (or the bar has no fill), compute mark-to-market equity = cash + sum(positions × close). Append to the equity curve.

Returns is `equity.pct_change().fillna(0)`.

- [ ] **Step 1: Write failing tests for run_backtest**

`tests/backtest/test_engine_run.py`:

```python
"""Tests for run_backtest."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date

import numpy as np
import pandas as pd
import pytest

from quant.backtest.engine import BacktestConfig, BacktestResult, run_backtest
from tests.conftest import EqualWeightStrategy


def _flat_bars(symbols: list[str], price: float = 100.0) -> pd.DataFrame:
    """Bars where every field equals `price`, every business day in 2024-Q1."""
    dates = pd.bdate_range("2024-01-02", "2024-03-29")  # avoid Jan 1 holiday
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


def test_equity_curve_indexed_by_history(
    make_bars: Callable[..., pd.DataFrame],
) -> None:
    bars = make_bars(["AAA", "BBB"], date(2024, 1, 1), date(2024, 6, 30), seed=0)
    strat = EqualWeightStrategy(bars=bars)
    result = run_backtest(strat, bars, BacktestConfig(), date(2024, 1, 1), date(2024, 6, 30))
    assert isinstance(result, BacktestResult)
    assert isinstance(result.equity_curve, pd.Series)
    assert result.equity_curve.index.equals(bars.index)


def test_starting_and_ending_equity(make_bars: Callable[..., pd.DataFrame]) -> None:
    bars = make_bars(["AAA", "BBB"], date(2024, 1, 1), date(2024, 6, 30), seed=0)
    strat = EqualWeightStrategy(bars=bars)
    cfg = BacktestConfig(starting_equity=50_000.0, slippage_bps=0.0)
    result = run_backtest(strat, bars, cfg, date(2024, 1, 1), date(2024, 6, 30))
    assert result.starting_equity == 50_000.0
    # First equity point >= starting equity - 1 cent rounding noise (positions integer).
    assert result.equity_curve.iloc[0] <= 50_000.0  # any positions reduce cash by their notional cost
    assert result.ending_equity == pytest.approx(result.equity_curve.iloc[-1], abs=1e-6)


def test_flat_price_zero_costs_preserves_equity_exactly() -> None:
    """If all prices are constant and costs are zero, equity should be flat at start."""
    bars = _flat_bars(["AAA", "BBB"])
    strat = EqualWeightStrategy(bars=bars)
    cfg = BacktestConfig(starting_equity=100_000.0, slippage_bps=0.0, commission_bps=0.0)
    result = run_backtest(strat, bars, cfg, date(2024, 1, 2), date(2024, 3, 29))
    # All prices constant → equity = cash + shares * 100. Should equal starting_equity bar-for-bar
    # once the first rebalance fills, ± rounding from integer share count.
    assert all(abs(eq - 100_000.0) < 100.0 for eq in result.equity_curve), result.equity_curve.head()


def test_slippage_drains_equity_on_each_rebalance() -> None:
    """Holding flat prices: every rebalance costs slippage; equity decays monotonically."""
    bars = _flat_bars(["AAA", "BBB"])
    strat = EqualWeightStrategy(bars=bars)  # monthly rebalance
    cfg = BacktestConfig(slippage_bps=50.0, commission_bps=0.0)  # 50 bps
    result = run_backtest(strat, bars, cfg, date(2024, 1, 2), date(2024, 3, 29))
    # First rebalance fills on next_open of first day. After that, no further rebalances within month → flat.
    assert result.ending_equity < result.starting_equity


def test_trades_have_required_columns(make_bars: Callable[..., pd.DataFrame]) -> None:
    bars = make_bars(["AAA", "BBB"], date(2024, 1, 1), date(2024, 6, 30), seed=0)
    strat = EqualWeightStrategy(bars=bars)
    result = run_backtest(strat, bars, BacktestConfig(), date(2024, 1, 1), date(2024, 6, 30))
    assert {
        "date", "symbol", "side", "qty", "fill_price",
        "slippage_cost", "commission_cost", "strategy_slug",
    } <= set(result.trades.columns)
    assert len(result.trades) > 0
    assert (result.trades["qty"] > 0).all()
    assert set(result.trades["side"]) <= {"buy", "sell"}


def test_positions_dataframe_tracks_shares(make_bars: Callable[..., pd.DataFrame]) -> None:
    bars = make_bars(["AAA", "BBB"], date(2024, 1, 1), date(2024, 6, 30), seed=0)
    strat = EqualWeightStrategy(bars=bars)
    result = run_backtest(strat, bars, BacktestConfig(), date(2024, 1, 1), date(2024, 6, 30))
    # First few bars: no positions yet (rebalance fills on next_open of first month-start bar).
    # By end of horizon: at least one symbol has positive shares.
    assert (result.positions.iloc[-1] > 0).any()
    assert set(result.positions.columns) <= {"AAA", "BBB"}


def test_close_execution_fills_same_bar() -> None:
    bars = _flat_bars(["AAA", "BBB"])
    strat = EqualWeightStrategy(bars=bars)
    cfg = BacktestConfig(execution="close", slippage_bps=0.0)
    result = run_backtest(strat, bars, cfg, date(2024, 1, 2), date(2024, 3, 29))
    # Close execution: trades on first day, not waiting for next open.
    assert (result.trades["date"] == pd.Timestamp("2024-01-02")).any()


def test_next_open_execution_fills_next_bar() -> None:
    bars = _flat_bars(["AAA", "BBB"])
    strat = EqualWeightStrategy(bars=bars)
    cfg = BacktestConfig(execution="next_open", slippage_bps=0.0)
    result = run_backtest(strat, bars, cfg, date(2024, 1, 2), date(2024, 3, 29))
    # next_open: first month-start rebalance on Jan 2 → fills Jan 3.
    first_trade_date = result.trades["date"].min()
    assert first_trade_date > pd.Timestamp("2024-01-02")


def test_empty_history_returns_empty_result(make_bars: Callable[..., pd.DataFrame]) -> None:
    bars = make_bars(["AAA", "BBB"], date(2024, 1, 1), date(2024, 6, 30), seed=0)
    strat = EqualWeightStrategy(bars=bars)
    # Request a window outside the data → engine should not crash; returns empty curve.
    result = run_backtest(strat, bars, BacktestConfig(), date(2030, 1, 1), date(2030, 12, 31))
    assert len(result.equity_curve) == 0
    assert len(result.trades) == 0


def test_window_slicing(make_bars: Callable[..., pd.DataFrame]) -> None:
    bars = make_bars(["AAA", "BBB"], date(2024, 1, 1), date(2024, 12, 31), seed=0)
    strat = EqualWeightStrategy(bars=bars)
    result = run_backtest(strat, bars, BacktestConfig(), date(2024, 6, 1), date(2024, 6, 30))
    assert result.equity_curve.index.min() >= pd.Timestamp("2024-06-01")
    assert result.equity_curve.index.max() <= pd.Timestamp("2024-06-30")
```

- [ ] **Step 2: Run tests to verify failure**

```bash
uv run pytest tests/backtest/test_engine_run.py -v
```

Expected: NotImplementedError raised by the stub.

- [ ] **Step 3: Implement run_backtest**

Replace the bottom of `quant/backtest/engine.py` (everything from `# Forward declaration` onward) with the real implementation:

```python
def run_backtest(
    strategy: "Strategy",
    bars: pd.DataFrame,
    config: BacktestConfig,
    start: date,
    end: date,
) -> BacktestResult:
    """Simulate `strategy` over `bars` restricted to [start, end].

    `bars` must be a wide DataFrame with MultiIndex columns (symbol, field) and
    a DatetimeIndex. `field` must include at least 'open' and 'close'.
    """
    from quant.backtest.calendar import is_rebalance_day

    # Slice the history to the requested window
    mask = (bars.index >= pd.Timestamp(start)) & (bars.index <= pd.Timestamp(end))
    history: pd.DatetimeIndex = bars.index[mask]

    if len(history) == 0:
        return BacktestResult(
            equity_curve=pd.Series(dtype=float),
            returns=pd.Series(dtype=float),
            positions=pd.DataFrame(),
            trades=pd.DataFrame(
                columns=[
                    "date", "symbol", "side", "qty", "fill_price",
                    "slippage_cost", "commission_cost", "strategy_slug",
                ]
            ),
            config=config,
            starting_equity=config.starting_equity,
            ending_equity=config.starting_equity,
        )

    cash = config.starting_equity
    positions: dict[str, int] = {}
    equity_records: list[float] = []
    position_records: list[dict[str, int]] = []
    trade_records: list[dict[str, object]] = []

    # Pending orders if execution == next_open
    pending: list[tuple[str, int, str]] = []  # (symbol, qty, side)

    history_list = list(history)
    for i, ts in enumerate(history_list):
        asof: date = ts.date()

        # 1. Execute pending fills on today's open (if any from prior bar's rebalance)
        if pending:
            for sym, qty, side in pending:
                if (sym, "open") not in bars.columns:
                    continue
                mid = float(bars[(sym, "open")].loc[ts])
                fill = apply_costs(qty=qty, mid_price=mid, side=side, config=config)
                notional = qty * fill.fill_price
                if side == "buy":
                    cash -= notional + fill.commission_cost
                    positions[sym] = positions.get(sym, 0) + qty
                else:
                    cash += notional - fill.commission_cost
                    positions[sym] = positions.get(sym, 0) - qty
                trade_records.append(
                    {
                        "date": ts,
                        "symbol": sym,
                        "side": side,
                        "qty": qty,
                        "fill_price": fill.fill_price,
                        "slippage_cost": fill.slippage_cost,
                        "commission_cost": fill.commission_cost,
                        "strategy_slug": strategy.spec.slug,
                    }
                )
            pending = []

        # 2. Mark-to-market on today's close to compute equity
        equity = cash
        for sym, qty in positions.items():
            if (sym, "close") in bars.columns and qty != 0:
                equity += qty * float(bars[(sym, "close")].loc[ts])
        equity_records.append(equity)
        position_records.append(dict(positions))

        # 3. Rebalance decision
        if is_rebalance_day(asof, strategy.spec.rebalance_frequency, history):
            try:
                target = strategy.target_positions(asof, equity)
            except Exception:  # noqa: BLE001 - any strategy error stops trading for the day
                target = {}

            new_orders: list[tuple[str, int, str]] = []
            symbols_to_consider = sorted(set(target) | set(positions))
            for sym in symbols_to_consider:
                tgt = target.get(sym, 0)
                cur = positions.get(sym, 0)
                delta = tgt - cur
                if delta == 0:
                    continue
                if (cur > 0 and tgt < 0) or (cur < 0 and tgt > 0):
                    # Flatten then reopen on the other side
                    flatten_side = "sell" if cur > 0 else "buy"
                    new_orders.append((sym, abs(cur), flatten_side))
                    open_side = "buy" if tgt > 0 else "sell"
                    new_orders.append((sym, abs(tgt), open_side))
                else:
                    side = "buy" if delta > 0 else "sell"
                    new_orders.append((sym, abs(delta), side))

            if config.execution == "close":
                # Fill on today's close immediately
                for sym, qty, side in new_orders:
                    if (sym, "close") not in bars.columns:
                        continue
                    mid = float(bars[(sym, "close")].loc[ts])
                    fill = apply_costs(qty=qty, mid_price=mid, side=side, config=config)
                    notional = qty * fill.fill_price
                    if side == "buy":
                        cash -= notional + fill.commission_cost
                        positions[sym] = positions.get(sym, 0) + qty
                    else:
                        cash += notional - fill.commission_cost
                        positions[sym] = positions.get(sym, 0) - qty
                    trade_records.append(
                        {
                            "date": ts,
                            "symbol": sym,
                            "side": side,
                            "qty": qty,
                            "fill_price": fill.fill_price,
                            "slippage_cost": fill.slippage_cost,
                            "commission_cost": fill.commission_cost,
                            "strategy_slug": strategy.spec.slug,
                        }
                    )
            else:
                # Queue for next_open
                pending = new_orders

    equity_curve = pd.Series(equity_records, index=history, name="equity")
    returns = equity_curve.pct_change().fillna(0.0)
    returns.name = "returns"

    positions_df = pd.DataFrame(position_records, index=history).fillna(0).astype(int)
    trades_df = pd.DataFrame(trade_records)
    if trades_df.empty:
        trades_df = pd.DataFrame(
            columns=[
                "date", "symbol", "side", "qty", "fill_price",
                "slippage_cost", "commission_cost", "strategy_slug",
            ]
        )

    return BacktestResult(
        equity_curve=equity_curve,
        returns=returns,
        positions=positions_df,
        trades=trades_df,
        config=config,
        starting_equity=config.starting_equity,
        ending_equity=float(equity_curve.iloc[-1]) if len(equity_curve) else config.starting_equity,
    )
```

At the top of the file add the TYPE_CHECKING import so the Strategy reference doesn't create a circular import at runtime:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quant.strategies.base import Strategy
```

- [ ] **Step 4: Run tests to verify pass**

```bash
uv run pytest tests/backtest/test_engine_run.py -v
```

Expected: 9 passed.

- [ ] **Step 5: Run full suite + lint + types**

```bash
uv run pytest -q -m "not network and not alpaca and not slow"
uv run ruff check . && uv run ruff format --check . && uv run mypy quant
```

Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add quant/backtest/engine.py tests/backtest/test_engine_run.py
git commit -m "feat(backtest): run_backtest day-by-day simulator"
```

---

## Task 7: Walk-forward — window generator

**Files:**
- Create: `quant/backtest/walkforward.py` (partial — just `iter_windows`)
- Create: `tests/backtest/test_walkforward_windows.py`

**Context.** Rolling train/test windows over a date range. Default: 5 years train, 1 year test, 6 month step (matches spec §4 item 1). Pure generator, no engine calls. Each yielded window is `(train_start, train_end, test_start, test_end)` — all `date` objects, inclusive.

- [ ] **Step 1: Write failing tests**

`tests/backtest/test_walkforward_windows.py`:

```python
"""Tests for quant.backtest.walkforward.iter_windows."""

from __future__ import annotations

from datetime import date

import pytest

from quant.backtest.walkforward import WalkforwardWindow, iter_windows


def test_first_window_starts_at_beginning() -> None:
    windows = list(iter_windows(date(2010, 1, 1), date(2020, 12, 31)))
    assert windows[0].train_start == date(2010, 1, 1)


def test_default_5y_train_1y_test() -> None:
    windows = list(iter_windows(date(2010, 1, 1), date(2020, 12, 31)))
    first = windows[0]
    assert first.train_end == date(2014, 12, 31)
    assert first.test_start == date(2015, 1, 1)
    assert first.test_end == date(2015, 12, 31)


def test_default_6m_step() -> None:
    windows = list(iter_windows(date(2010, 1, 1), date(2020, 12, 31)))
    assert windows[1].train_start == date(2010, 7, 1)
    assert windows[1].test_start == date(2015, 7, 1)


def test_windows_stop_when_test_exceeds_end() -> None:
    windows = list(iter_windows(date(2010, 1, 1), date(2017, 12, 31)))
    # 5y train requires train_start <= 2012-12-31 → only a few step positions valid.
    for w in windows:
        assert w.test_end <= date(2017, 12, 31)


def test_custom_train_test_step() -> None:
    windows = list(
        iter_windows(
            start=date(2020, 1, 1),
            end=date(2024, 12, 31),
            train_years=2,
            test_years=1,
            step_months=12,
        )
    )
    assert windows[0].train_start == date(2020, 1, 1)
    assert windows[0].train_end == date(2021, 12, 31)
    assert windows[0].test_start == date(2022, 1, 1)
    assert windows[0].test_end == date(2022, 12, 31)
    assert windows[1].train_start == date(2021, 1, 1)


def test_empty_when_window_doesnt_fit() -> None:
    # 5y train + 1y test = 6y total; only 2y of data.
    windows = list(iter_windows(date(2020, 1, 1), date(2022, 1, 1)))
    assert windows == []


def test_window_is_dataclass() -> None:
    w = WalkforwardWindow(
        train_start=date(2010, 1, 1),
        train_end=date(2014, 12, 31),
        test_start=date(2015, 1, 1),
        test_end=date(2015, 12, 31),
    )
    assert w.train_start == date(2010, 1, 1)


def test_invalid_arguments_raise() -> None:
    with pytest.raises(ValueError):
        list(iter_windows(date(2020, 1, 1), date(2010, 1, 1)))  # end before start
    with pytest.raises(ValueError):
        list(iter_windows(date(2010, 1, 1), date(2020, 1, 1), train_years=0))
```

- [ ] **Step 2: Run tests to verify failure**

```bash
uv run pytest tests/backtest/test_walkforward_windows.py -v
```

Expected: ImportError on `quant.backtest.walkforward`.

- [ ] **Step 3: Implement iter_windows**

`quant/backtest/walkforward.py`:

```python
"""Walk-forward harness: rolling train/test windows, grid search per window."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import date, timedelta

from dateutil.relativedelta import relativedelta


@dataclass(frozen=True)
class WalkforwardWindow:
    train_start: date
    train_end: date
    test_start: date
    test_end: date


def iter_windows(
    start: date,
    end: date,
    train_years: int = 5,
    test_years: int = 1,
    step_months: int = 6,
) -> Iterator[WalkforwardWindow]:
    """Yield rolling train/test windows over [start, end].

    The first window's train_start = `start`. Each subsequent window steps the
    train_start forward by `step_months`. A window is yielded only if its
    test_end <= `end`.
    """
    if end <= start:
        raise ValueError(f"end ({end}) must be > start ({start})")
    if train_years <= 0 or test_years <= 0 or step_months <= 0:
        raise ValueError("train_years, test_years, step_months must all be positive")

    train_start = start
    while True:
        train_end = train_start + relativedelta(years=train_years) - timedelta(days=1)
        test_start = train_end + timedelta(days=1)
        test_end = test_start + relativedelta(years=test_years) - timedelta(days=1)
        if test_end > end:
            return
        yield WalkforwardWindow(
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        )
        train_start = train_start + relativedelta(months=step_months)
```

Note: `python-dateutil` is already in pyproject.toml deps (Plan 1).

- [ ] **Step 4: Run tests to verify pass**

```bash
uv run pytest tests/backtest/test_walkforward_windows.py -v
```

Expected: 8 passed.

- [ ] **Step 5: Lint + types**

```bash
uv run ruff check quant tests && uv run ruff format --check quant tests && uv run mypy quant
```

Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add quant/backtest/walkforward.py tests/backtest/test_walkforward_windows.py
git commit -m "feat(backtest): walk-forward window generator"
```

---

## Task 8: Walk-forward — parameter selection + orchestrator

**Files:**
- Modify: `quant/backtest/walkforward.py`
- Create: `tests/backtest/test_walkforward_selection.py`
- Create: `tests/backtest/test_walkforward_run.py`

**Context.** Build the two remaining pieces:

1. `select_best_params(strategy_factory, param_grid, bars, window, config) -> dict` — Cartesian-product the grid, run the engine on `(window.train_start, window.train_end)` for each candidate, return the params with the highest Sharpe on the train window.

2. `run_walkforward(strategy_factory, param_grid, bars, start, end, config) -> WalkforwardResult` — for each window, select best params on train, run engine on test, concat results. Returns a `WalkforwardResult` with `oos_equity_curve` (stitched), `oos_returns`, `oos_trades`, `per_window_params: list[(window, params)]`, `combined_result: BacktestResult` (the stitched-as-one for downstream tearsheet).

**Strategy factory contract.** Since strategies may need bars-at-construction (like `EqualWeightStrategy` in tests), the harness accepts a `strategy_factory: Callable[[dict, pd.DataFrame], Strategy]` instead of bare classes. Production strategies (Plan 4+) will provide their own simple factories.

- [ ] **Step 1: Write failing tests for select_best_params**

`tests/backtest/test_walkforward_selection.py`:

```python
"""Tests for select_best_params."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from typing import Any, ClassVar

import pandas as pd
import pytest

from quant.backtest.engine import BacktestConfig
from quant.backtest.walkforward import WalkforwardWindow, select_best_params
from quant.strategies.base import Strategy, StrategySpec
from tests.conftest import EqualWeightStrategy


class _TiltedStrategy(Strategy):
    """Long-only allocates 100% to "AAA" when params['tilt'] = 'aaa' else "BBB"."""

    spec: ClassVar[StrategySpec] = StrategySpec(
        slug="tilted-test",
        name="Tilted (test)",
        description="-",
        universe=["AAA", "BBB"],
        rebalance_frequency="monthly",
    )
    default_params: ClassVar[dict[str, Any]] = {"tilt": "aaa"}

    def __init__(self, bars: pd.DataFrame, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        self._bars = bars

    def generate_signals(self, asof: date) -> pd.Series:
        return pd.Series({"AAA": 1.0, "BBB": 1.0})

    def target_positions(self, asof: date, equity: float) -> dict[str, int]:
        ts = pd.Timestamp(asof)
        if ts not in self._bars.index:
            return {}
        sym = "AAA" if self.params["tilt"] == "aaa" else "BBB"
        price = float(self._bars[(sym, "close")].loc[ts])
        return {sym: int(equity // price)}


def test_select_best_params_picks_higher_sharpe(make_bars: Callable[..., pd.DataFrame]) -> None:
    # Construct bars where AAA strongly trends up and BBB trends down.
    bars_aaa = make_bars(["AAA"], date(2020, 1, 1), date(2024, 12, 31), seed=0, drift=0.002)
    bars_bbb = make_bars(["BBB"], date(2020, 1, 1), date(2024, 12, 31), seed=0, drift=-0.001)
    bars = pd.concat([bars_aaa, bars_bbb], axis=1)

    def factory(params: dict[str, Any], bars_for_strategy: pd.DataFrame) -> Strategy:
        return _TiltedStrategy(bars=bars_for_strategy, params=params)

    window = WalkforwardWindow(
        train_start=date(2020, 1, 1),
        train_end=date(2023, 12, 31),
        test_start=date(2024, 1, 1),
        test_end=date(2024, 12, 31),
    )

    best = select_best_params(
        strategy_factory=factory,
        param_grid={"tilt": ["aaa", "bbb"]},
        bars=bars,
        window=window,
        config=BacktestConfig(slippage_bps=0.0),
    )
    assert best == {"tilt": "aaa"}


def test_select_best_params_handles_empty_grid(make_bars: Callable[..., pd.DataFrame]) -> None:
    bars = make_bars(["AAA", "BBB"], date(2020, 1, 1), date(2024, 12, 31), seed=0)

    def factory(params: dict[str, Any], bars_for_strategy: pd.DataFrame) -> Strategy:
        return EqualWeightStrategy(bars=bars_for_strategy, params=params)

    window = WalkforwardWindow(
        train_start=date(2020, 1, 1),
        train_end=date(2023, 12, 31),
        test_start=date(2024, 1, 1),
        test_end=date(2024, 12, 31),
    )
    best = select_best_params(
        strategy_factory=factory,
        param_grid={},
        bars=bars,
        window=window,
        config=BacktestConfig(),
    )
    assert best == {}


def test_select_best_params_explores_full_grid(
    make_bars: Callable[..., pd.DataFrame],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify Cartesian product: 2x3 grid → 6 engine calls."""
    bars = make_bars(["AAA", "BBB"], date(2020, 1, 1), date(2024, 12, 31), seed=0)

    call_count = 0

    def factory(params: dict[str, Any], bars_for_strategy: pd.DataFrame) -> Strategy:
        nonlocal call_count
        call_count += 1
        return EqualWeightStrategy(bars=bars_for_strategy, params=params)

    window = WalkforwardWindow(
        train_start=date(2020, 1, 1),
        train_end=date(2023, 12, 31),
        test_start=date(2024, 1, 1),
        test_end=date(2024, 12, 31),
    )
    select_best_params(
        strategy_factory=factory,
        param_grid={"a": [1, 2], "b": [10, 20, 30]},
        bars=bars,
        window=window,
        config=BacktestConfig(),
    )
    assert call_count == 6
```

- [ ] **Step 2: Write failing tests for run_walkforward**

`tests/backtest/test_walkforward_run.py`:

```python
"""Tests for run_walkforward."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from typing import Any

import pandas as pd

from quant.backtest.engine import BacktestConfig, BacktestResult
from quant.backtest.walkforward import WalkforwardResult, run_walkforward
from quant.strategies.base import Strategy
from tests.conftest import EqualWeightStrategy


def _factory(params: dict[str, Any], bars: pd.DataFrame) -> Strategy:
    return EqualWeightStrategy(bars=bars, params=params)


def test_oos_curve_starts_at_first_test_window(make_bars: Callable[..., pd.DataFrame]) -> None:
    bars = make_bars(["AAA", "BBB"], date(2010, 1, 1), date(2020, 12, 31), seed=0)
    result = run_walkforward(
        strategy_factory=_factory,
        param_grid={"_dummy": [1]},  # single-point grid
        bars=bars,
        start=date(2010, 1, 1),
        end=date(2020, 12, 31),
        config=BacktestConfig(slippage_bps=0.0),
    )
    assert isinstance(result, WalkforwardResult)
    assert result.oos_equity_curve.index.min() >= pd.Timestamp("2015-01-01")


def test_per_window_params_present(make_bars: Callable[..., pd.DataFrame]) -> None:
    bars = make_bars(["AAA", "BBB"], date(2010, 1, 1), date(2020, 12, 31), seed=0)
    result = run_walkforward(
        strategy_factory=_factory,
        param_grid={"_dummy": [1]},
        bars=bars,
        start=date(2010, 1, 1),
        end=date(2020, 12, 31),
        config=BacktestConfig(),
    )
    assert len(result.per_window_params) > 0
    for window, params in result.per_window_params:
        assert params == {"_dummy": 1}


def test_combined_result_has_full_oos_history(make_bars: Callable[..., pd.DataFrame]) -> None:
    bars = make_bars(["AAA", "BBB"], date(2010, 1, 1), date(2020, 12, 31), seed=0)
    result = run_walkforward(
        strategy_factory=_factory,
        param_grid={"_dummy": [1]},
        bars=bars,
        start=date(2010, 1, 1),
        end=date(2020, 12, 31),
        config=BacktestConfig(),
    )
    assert isinstance(result.combined_result, BacktestResult)
    # The combined result's equity_curve should equal the stitched OOS curve.
    pd.testing.assert_series_equal(
        result.combined_result.equity_curve, result.oos_equity_curve, check_names=False
    )


def test_oos_curve_monotone_chronological(make_bars: Callable[..., pd.DataFrame]) -> None:
    bars = make_bars(["AAA", "BBB"], date(2010, 1, 1), date(2020, 12, 31), seed=0)
    result = run_walkforward(
        strategy_factory=_factory,
        param_grid={"_dummy": [1]},
        bars=bars,
        start=date(2010, 1, 1),
        end=date(2020, 12, 31),
        config=BacktestConfig(),
    )
    idx = result.oos_equity_curve.index
    assert (idx[1:] > idx[:-1]).all()


def test_no_windows_returns_empty_result(make_bars: Callable[..., pd.DataFrame]) -> None:
    bars = make_bars(["AAA", "BBB"], date(2020, 1, 1), date(2021, 12, 31), seed=0)
    # 2y of data, default 5y train + 1y test → no fit-able window.
    result = run_walkforward(
        strategy_factory=_factory,
        param_grid={"_dummy": [1]},
        bars=bars,
        start=date(2020, 1, 1),
        end=date(2021, 12, 31),
        config=BacktestConfig(),
    )
    assert len(result.oos_equity_curve) == 0
    assert len(result.per_window_params) == 0
```

- [ ] **Step 3: Run both failing tests to verify**

```bash
uv run pytest tests/backtest/test_walkforward_selection.py tests/backtest/test_walkforward_run.py -v
```

Expected: ImportError on `select_best_params` / `run_walkforward` / `WalkforwardResult`.

- [ ] **Step 4: Append to `quant/backtest/walkforward.py`**

Add these imports at the top of the file (after the existing imports):

```python
from collections.abc import Callable
from itertools import product
from typing import Any, TYPE_CHECKING

import pandas as pd

from quant.backtest.engine import BacktestConfig, BacktestResult, run_backtest
from quant.backtest.metrics import sharpe
from quant.util.logging import logger

if TYPE_CHECKING:
    from quant.strategies.base import Strategy
```

Append these new definitions to the bottom of the file:

```python
StrategyFactory = Callable[[dict[str, Any], pd.DataFrame], "Strategy"]


@dataclass(frozen=True)
class WalkforwardResult:
    """Output of run_walkforward."""

    oos_equity_curve: pd.Series
    oos_returns: pd.Series
    oos_trades: pd.DataFrame
    per_window_params: list[tuple[WalkforwardWindow, dict[str, Any]]]
    combined_result: BacktestResult


def _iter_grid(param_grid: dict[str, Sequence[Any]]) -> Iterator[dict[str, Any]]:
    """Cartesian product of `param_grid`. Yields one dict per combo.

    An empty grid yields one empty dict (the strategy's defaults are used).
    """
    if not param_grid:
        yield {}
        return
    keys = list(param_grid.keys())
    for combo in product(*(param_grid[k] for k in keys)):
        yield dict(zip(keys, combo, strict=True))


def select_best_params(
    strategy_factory: StrategyFactory,
    param_grid: dict[str, Sequence[Any]],
    bars: pd.DataFrame,
    window: WalkforwardWindow,
    config: BacktestConfig,
) -> dict[str, Any]:
    """Grid-search `param_grid` on `[window.train_start, window.train_end]`, return best by Sharpe."""
    best_params: dict[str, Any] = {}
    best_score: float = float("-inf")

    for params in _iter_grid(param_grid):
        strat = strategy_factory(params, bars)
        result = run_backtest(strat, bars, config, window.train_start, window.train_end)
        score = sharpe(result.returns)
        if score > best_score:
            best_score = score
            best_params = params

    return best_params


def run_walkforward(
    strategy_factory: StrategyFactory,
    param_grid: dict[str, Sequence[Any]],
    bars: pd.DataFrame,
    start: date,
    end: date,
    config: BacktestConfig,
    train_years: int = 5,
    test_years: int = 1,
    step_months: int = 6,
) -> WalkforwardResult:
    """Walk-forward orchestrator. Yields one OOS curve stitched from per-window backtests."""
    oos_equity_pieces: list[pd.Series] = []
    oos_trades_pieces: list[pd.DataFrame] = []
    per_window_params: list[tuple[WalkforwardWindow, dict[str, Any]]] = []

    cumulative_equity: float = config.starting_equity

    for window in iter_windows(start, end, train_years, test_years, step_months):
        logger.info(
            "Walk-forward: train {}..{} → test {}..{}",
            window.train_start, window.train_end, window.test_start, window.test_end,
        )
        best = select_best_params(strategy_factory, param_grid, bars, window, config)

        # Run test window with chosen params, seeded with current cumulative equity
        test_config = BacktestConfig(
            starting_equity=cumulative_equity,
            slippage_bps=config.slippage_bps,
            commission_bps=config.commission_bps,
            execution=config.execution,
        )
        test_strat = strategy_factory(best, bars)
        test_result = run_backtest(
            test_strat, bars, test_config, window.test_start, window.test_end
        )
        if len(test_result.equity_curve) == 0:
            continue
        oos_equity_pieces.append(test_result.equity_curve)
        oos_trades_pieces.append(test_result.trades)
        per_window_params.append((window, best))
        cumulative_equity = test_result.ending_equity

    if not oos_equity_pieces:
        empty_series = pd.Series(dtype=float)
        empty_trades = pd.DataFrame(
            columns=[
                "date", "symbol", "side", "qty", "fill_price",
                "slippage_cost", "commission_cost", "strategy_slug",
            ]
        )
        empty_combined = BacktestResult(
            equity_curve=empty_series,
            returns=empty_series,
            positions=pd.DataFrame(),
            trades=empty_trades,
            config=config,
            starting_equity=config.starting_equity,
            ending_equity=config.starting_equity,
        )
        return WalkforwardResult(
            oos_equity_curve=empty_series,
            oos_returns=empty_series,
            oos_trades=empty_trades,
            per_window_params=[],
            combined_result=empty_combined,
        )

    oos_equity = pd.concat(oos_equity_pieces)
    oos_equity = oos_equity[~oos_equity.index.duplicated(keep="last")].sort_index()
    oos_returns = oos_equity.pct_change().fillna(0.0)
    oos_trades = pd.concat(oos_trades_pieces, ignore_index=True) if oos_trades_pieces else pd.DataFrame()

    combined = BacktestResult(
        equity_curve=oos_equity,
        returns=oos_returns,
        positions=pd.DataFrame(),  # positions aren't stitched cleanly; defer to per-window if needed
        trades=oos_trades,
        config=config,
        starting_equity=config.starting_equity,
        ending_equity=float(oos_equity.iloc[-1]),
        metadata={"walkforward": True, "n_windows": len(per_window_params)},
    )

    return WalkforwardResult(
        oos_equity_curve=oos_equity,
        oos_returns=oos_returns,
        oos_trades=oos_trades,
        per_window_params=per_window_params,
        combined_result=combined,
    )
```

- [ ] **Step 5: Run tests to verify pass**

```bash
uv run pytest tests/backtest/test_walkforward_selection.py tests/backtest/test_walkforward_run.py -v
```

Expected: all pass.

- [ ] **Step 6: Run full suite + lint + types**

```bash
uv run pytest -q -m "not network and not alpaca and not slow"
uv run ruff check . && uv run ruff format --check . && uv run mypy quant
```

Expected: green.

- [ ] **Step 7: Commit**

```bash
git add quant/backtest/walkforward.py tests/backtest/test_walkforward_selection.py tests/backtest/test_walkforward_run.py
git commit -m "feat(backtest): walk-forward parameter selection + orchestrator"
```

---

## Task 9: Tear-sheet generator

**Files:**
- Modify: `pyproject.toml` (add matplotlib + jinja2 deps)
- Create: `quant/backtest/templates/tearsheet.html.j2`
- Create: `quant/backtest/tearsheet.py`
- Create: `tests/backtest/test_tearsheet.py`
- Modify: `quant/backtest/__init__.py` (add `write_tearsheet` re-export)

**Context.** Take a `WalkforwardResult` + slug + output dir, write three files:
- `data/backtests/<slug>/tearsheet.html` — rendered HTML with embedded base64 PNGs
- `data/backtests/<slug>/walkforward.parquet` — the OOS equity curve as parquet
- `data/backtests/<slug>/chosen_params.json` — per-window params history

Charts: equity curve (line), drawdown (filled), monthly returns heatmap, returns distribution histogram. Top-of-page table: total return, CAGR, Sharpe, Sortino, max DD, win rate, n trades, n windows.

The HTML template is jinja2-rendered; charts are matplotlib `Figure`s converted to base64 PNG data URIs and embedded inline (single self-contained file, no external assets).

- [ ] **Step 1: Add matplotlib + jinja2 deps and re-sync**

Modify `pyproject.toml` `[project]` dependencies list — append:

```toml
    "matplotlib>=3.8",
    "jinja2>=3.1",
```

Then:

```bash
uv sync --all-extras
```

Add to `[[tool.mypy.overrides]]` the `matplotlib.*` module so we don't fight stubs:

```toml
[[tool.mypy.overrides]]
module = ["yfinance.*", "fredapi.*", "alpaca.*", "matplotlib.*"]
ignore_missing_imports = true
```

- [ ] **Step 2: Write failing tests**

`tests/backtest/test_tearsheet.py`:

```python
"""Tests for write_tearsheet."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from quant.backtest.engine import BacktestConfig
from quant.backtest.tearsheet import write_tearsheet
from quant.backtest.walkforward import run_walkforward
from quant.strategies.base import Strategy
from tests.conftest import EqualWeightStrategy


def _factory(params: dict[str, Any], bars: pd.DataFrame) -> Strategy:
    return EqualWeightStrategy(bars=bars, params=params)


def test_tearsheet_writes_three_files(
    tmp_path: Path,
    make_bars: Callable[..., pd.DataFrame],
) -> None:
    bars = make_bars(["AAA", "BBB"], date(2010, 1, 1), date(2020, 12, 31), seed=0)
    wf = run_walkforward(
        strategy_factory=_factory,
        param_grid={"_dummy": [1]},
        bars=bars,
        start=date(2010, 1, 1),
        end=date(2020, 12, 31),
        config=BacktestConfig(),
    )

    out_dir = tmp_path / "backtests" / "equal-weight-test"
    write_tearsheet(
        result=wf,
        slug="equal-weight-test",
        strategy_name="Equal Weight (test)",
        out_dir=out_dir,
    )

    assert (out_dir / "tearsheet.html").exists()
    assert (out_dir / "walkforward.parquet").exists()
    assert (out_dir / "chosen_params.json").exists()


def test_tearsheet_html_contains_strategy_name(
    tmp_path: Path,
    make_bars: Callable[..., pd.DataFrame],
) -> None:
    bars = make_bars(["AAA", "BBB"], date(2010, 1, 1), date(2020, 12, 31), seed=0)
    wf = run_walkforward(
        strategy_factory=_factory,
        param_grid={"_dummy": [1]},
        bars=bars,
        start=date(2010, 1, 1),
        end=date(2020, 12, 31),
        config=BacktestConfig(),
    )

    out_dir = tmp_path / "backtests" / "equal-weight-test"
    write_tearsheet(
        result=wf,
        slug="equal-weight-test",
        strategy_name="Equal Weight (test)",
        out_dir=out_dir,
    )

    html = (out_dir / "tearsheet.html").read_text()
    assert "Equal Weight (test)" in html
    # Embedded charts use base64 data URIs:
    assert "data:image/png;base64," in html
    # Key metrics:
    assert "Sharpe" in html
    assert "Max Drawdown" in html
    assert "CAGR" in html


def test_tearsheet_walkforward_parquet_matches_oos_curve(
    tmp_path: Path,
    make_bars: Callable[..., pd.DataFrame],
) -> None:
    bars = make_bars(["AAA", "BBB"], date(2010, 1, 1), date(2020, 12, 31), seed=0)
    wf = run_walkforward(
        strategy_factory=_factory,
        param_grid={"_dummy": [1]},
        bars=bars,
        start=date(2010, 1, 1),
        end=date(2020, 12, 31),
        config=BacktestConfig(),
    )

    out_dir = tmp_path / "backtests" / "equal-weight-test"
    write_tearsheet(
        result=wf,
        slug="equal-weight-test",
        strategy_name="Equal Weight (test)",
        out_dir=out_dir,
    )

    on_disk = pd.read_parquet(out_dir / "walkforward.parquet")
    assert "equity" in on_disk.columns
    assert len(on_disk) == len(wf.oos_equity_curve)


def test_tearsheet_chosen_params_json_shape(
    tmp_path: Path,
    make_bars: Callable[..., pd.DataFrame],
) -> None:
    bars = make_bars(["AAA", "BBB"], date(2010, 1, 1), date(2020, 12, 31), seed=0)
    wf = run_walkforward(
        strategy_factory=_factory,
        param_grid={"_dummy": [1]},
        bars=bars,
        start=date(2010, 1, 1),
        end=date(2020, 12, 31),
        config=BacktestConfig(),
    )

    out_dir = tmp_path / "backtests" / "equal-weight-test"
    write_tearsheet(
        result=wf,
        slug="equal-weight-test",
        strategy_name="Equal Weight (test)",
        out_dir=out_dir,
    )

    payload = json.loads((out_dir / "chosen_params.json").read_text())
    assert "windows" in payload
    assert isinstance(payload["windows"], list)
    assert len(payload["windows"]) > 0
    first = payload["windows"][0]
    for key in ("train_start", "train_end", "test_start", "test_end", "params"):
        assert key in first


def test_tearsheet_empty_walkforward(tmp_path: Path) -> None:
    """A walk-forward with no fit-able windows produces a minimal tear-sheet, no crash."""
    from quant.backtest.engine import BacktestResult
    from quant.backtest.walkforward import WalkforwardResult

    empty_series = pd.Series(dtype=float, name="equity")
    empty_trades = pd.DataFrame(
        columns=["date", "symbol", "side", "qty", "fill_price",
                 "slippage_cost", "commission_cost", "strategy_slug"]
    )
    empty_combined = BacktestResult(
        equity_curve=empty_series, returns=empty_series, positions=pd.DataFrame(),
        trades=empty_trades, config=BacktestConfig(),
        starting_equity=100_000.0, ending_equity=100_000.0,
    )
    empty_wf = WalkforwardResult(
        oos_equity_curve=empty_series, oos_returns=empty_series, oos_trades=empty_trades,
        per_window_params=[], combined_result=empty_combined,
    )
    out_dir = tmp_path / "backtests" / "empty"
    write_tearsheet(result=empty_wf, slug="empty", strategy_name="Empty", out_dir=out_dir)
    assert (out_dir / "tearsheet.html").exists()
    html = (out_dir / "tearsheet.html").read_text()
    assert "no walk-forward windows" in html.lower()
```

- [ ] **Step 3: Run failing tests**

```bash
uv run pytest tests/backtest/test_tearsheet.py -v
```

Expected: ImportError on `quant.backtest.tearsheet`.

- [ ] **Step 4: Implement the template**

`quant/backtest/templates/tearsheet.html.j2`:

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{{ strategy_name }} — Tear-Sheet</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
         margin: 2rem; color: #1a1a1a; max-width: 1000px; }
  h1 { font-size: 1.6rem; margin-bottom: 0.25rem; }
  .subtitle { color: #555; margin-bottom: 1.5rem; }
  table.metrics { border-collapse: collapse; margin-bottom: 2rem; }
  table.metrics td { padding: 0.4rem 1.25rem 0.4rem 0; font-variant-numeric: tabular-nums; }
  table.metrics td.label { color: #555; }
  table.metrics td.value { font-weight: 600; text-align: right; }
  h2 { font-size: 1.1rem; margin-top: 2rem; border-bottom: 1px solid #ddd; padding-bottom: 0.25rem; }
  img.chart { max-width: 100%; height: auto; display: block; margin: 0.5rem 0 1rem 0; }
  table.windows { border-collapse: collapse; font-size: 0.85rem; }
  table.windows th, table.windows td { border: 1px solid #ddd; padding: 0.25rem 0.6rem; }
  table.windows th { background: #f5f5f5; text-align: left; }
  .empty { color: #777; font-style: italic; }
</style>
</head>
<body>
  <h1>{{ strategy_name }}</h1>
  <div class="subtitle">slug <code>{{ slug }}</code> · {{ n_windows }} walk-forward windows · OOS {{ oos_start }} → {{ oos_end }}</div>

  {% if n_windows == 0 %}
    <p class="empty">This run produced no walk-forward windows — try a longer history or smaller train_years.</p>
  {% else %}
    <h2>Key metrics (OOS)</h2>
    <table class="metrics">
      <tr><td class="label">Total Return</td><td class="value">{{ "{:+.2%}".format(metrics.total_return) }}</td></tr>
      <tr><td class="label">CAGR</td><td class="value">{{ "{:+.2%}".format(metrics.cagr) }}</td></tr>
      <tr><td class="label">Sharpe</td><td class="value">{{ "{:.2f}".format(metrics.sharpe) }}</td></tr>
      <tr><td class="label">Sortino</td><td class="value">{{ "{:.2f}".format(metrics.sortino) }}</td></tr>
      <tr><td class="label">Max Drawdown</td><td class="value">{{ "{:.2%}".format(metrics.max_drawdown) }}</td></tr>
      <tr><td class="label">Win Rate (daily)</td><td class="value">{{ "{:.1%}".format(metrics.win_rate) }}</td></tr>
      <tr><td class="label">Trades</td><td class="value">{{ metrics.n_trades }}</td></tr>
      <tr><td class="label">Starting Equity</td><td class="value">${{ "{:,.0f}".format(metrics.starting_equity) }}</td></tr>
      <tr><td class="label">Ending Equity</td><td class="value">${{ "{:,.0f}".format(metrics.ending_equity) }}</td></tr>
    </table>

    <h2>Equity curve (OOS)</h2>
    <img class="chart" src="data:image/png;base64,{{ charts.equity }}" alt="OOS equity curve">

    <h2>Drawdown</h2>
    <img class="chart" src="data:image/png;base64,{{ charts.drawdown }}" alt="Drawdown">

    <h2>Monthly returns</h2>
    <img class="chart" src="data:image/png;base64,{{ charts.monthly }}" alt="Monthly returns heatmap">

    <h2>Returns distribution</h2>
    <img class="chart" src="data:image/png;base64,{{ charts.distribution }}" alt="Returns histogram">

    <h2>Walk-forward windows</h2>
    <table class="windows">
      <thead>
        <tr><th>Train start</th><th>Train end</th><th>Test start</th><th>Test end</th><th>Chosen params</th></tr>
      </thead>
      <tbody>
        {% for w in windows %}
          <tr>
            <td>{{ w.train_start }}</td>
            <td>{{ w.train_end }}</td>
            <td>{{ w.test_start }}</td>
            <td>{{ w.test_end }}</td>
            <td><code>{{ w.params }}</code></td>
          </tr>
        {% endfor %}
      </tbody>
    </table>
  {% endif %}
</body>
</html>
```

- [ ] **Step 5: Implement the tear-sheet writer**

`quant/backtest/tearsheet.py`:

```python
"""HTML tear-sheet writer.

Renders a self-contained HTML report (charts embedded as base64 PNGs) for a
walk-forward result. Also writes the OOS equity curve as parquet and the
per-window chosen params as JSON to the same directory.
"""

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless rendering; no GUI required
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from quant.backtest.metrics import (
    cagr,
    max_drawdown,
    sharpe,
    sortino,
    total_return,
    win_rate,
)
from quant.backtest.walkforward import WalkforwardResult


_TEMPLATE_DIR = Path(__file__).parent / "templates"


@dataclass(frozen=True)
class _MetricsBundle:
    total_return: float
    cagr: float
    sharpe: float
    sortino: float
    max_drawdown: float
    win_rate: float
    n_trades: int
    starting_equity: float
    ending_equity: float


def _fig_to_base64(fig: "plt.Figure") -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _equity_chart(equity: pd.Series) -> str:
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(equity.index, equity.values, color="#1a3a8f", linewidth=1.2)
    ax.set_ylabel("Equity ($)")
    ax.set_title("OOS Equity Curve")
    ax.grid(True, alpha=0.3)
    return _fig_to_base64(fig)


def _drawdown_chart(equity: pd.Series) -> str:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    fig, ax = plt.subplots(figsize=(9, 2.5))
    ax.fill_between(dd.index, dd.values, 0, color="#c0392b", alpha=0.4)
    ax.plot(dd.index, dd.values, color="#c0392b", linewidth=0.8)
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_title("Drawdown")
    ax.grid(True, alpha=0.3)
    return _fig_to_base64(fig)


def _monthly_chart(returns: pd.Series) -> str:
    if len(returns) == 0:
        fig, ax = plt.subplots(figsize=(9, 2.0))
        ax.text(0.5, 0.5, "no monthly data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return _fig_to_base64(fig)

    monthly = (1.0 + returns).resample("ME").prod() - 1.0
    pivot = pd.DataFrame(
        {"year": monthly.index.year, "month": monthly.index.month, "ret": monthly.values}
    ).pivot(index="year", columns="month", values="ret")
    pivot = pivot.reindex(columns=range(1, 13))

    fig, ax = plt.subplots(figsize=(9, max(2.0, 0.35 * len(pivot))))
    cmap = plt.get_cmap("RdYlGn")
    vmax = float(np.nanmax(np.abs(pivot.values))) if pivot.size else 0.05
    vmin = -vmax
    im = ax.imshow(pivot.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(12), labels=["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax.set_yticks(range(len(pivot.index)), labels=[str(y) for y in pivot.index])
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.iat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.1%}", ha="center", va="center", fontsize=7, color="#222")
    fig.colorbar(im, ax=ax, fraction=0.03, format=plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_title("Monthly Returns")
    return _fig_to_base64(fig)


def _distribution_chart(returns: pd.Series) -> str:
    fig, ax = plt.subplots(figsize=(9, 2.5))
    if len(returns) > 0:
        ax.hist(returns.values, bins=60, color="#1a3a8f", alpha=0.7)
    ax.set_ylabel("Frequency")
    ax.set_xlabel("Daily return")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:+.1%}"))
    ax.set_title("Daily-returns Distribution")
    ax.grid(True, alpha=0.3)
    return _fig_to_base64(fig)


def write_tearsheet(
    result: WalkforwardResult,
    slug: str,
    strategy_name: str,
    out_dir: Path,
) -> Path:
    """Render the HTML tear-sheet + sidecar parquet + JSON. Return the HTML path."""
    out_dir.mkdir(parents=True, exist_ok=True)

    n_windows = len(result.per_window_params)
    oos_start = (
        str(result.oos_equity_curve.index.min().date()) if n_windows > 0 else "—"
    )
    oos_end = (
        str(result.oos_equity_curve.index.max().date()) if n_windows > 0 else "—"
    )

    metrics = _MetricsBundle(
        total_return=total_return(result.oos_returns),
        cagr=cagr(result.oos_returns),
        sharpe=sharpe(result.oos_returns),
        sortino=sortino(result.oos_returns),
        max_drawdown=max_drawdown(result.oos_returns),
        win_rate=win_rate(result.oos_returns),
        n_trades=int(len(result.oos_trades)),
        starting_equity=float(result.combined_result.starting_equity),
        ending_equity=float(result.combined_result.ending_equity),
    )

    charts: dict[str, str] = {}
    if n_windows > 0:
        charts = {
            "equity": _equity_chart(result.oos_equity_curve),
            "drawdown": _drawdown_chart(result.oos_equity_curve),
            "monthly": _monthly_chart(result.oos_returns),
            "distribution": _distribution_chart(result.oos_returns),
        }

    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("tearsheet.html.j2")

    html = template.render(
        strategy_name=strategy_name,
        slug=slug,
        n_windows=n_windows,
        oos_start=oos_start,
        oos_end=oos_end,
        metrics=metrics,
        charts=charts,
        windows=[
            {
                "train_start": str(w.train_start),
                "train_end": str(w.train_end),
                "test_start": str(w.test_start),
                "test_end": str(w.test_end),
                "params": params,
            }
            for w, params in result.per_window_params
        ],
    )

    html_path = out_dir / "tearsheet.html"
    html_path.write_text(html, encoding="utf-8")

    # Sidecar parquet
    equity_df = result.oos_equity_curve.to_frame(name="equity")
    equity_df.to_parquet(out_dir / "walkforward.parquet")

    # Sidecar JSON
    payload = {
        "slug": slug,
        "strategy_name": strategy_name,
        "n_windows": n_windows,
        "windows": [
            {
                "train_start": str(w.train_start),
                "train_end": str(w.train_end),
                "test_start": str(w.test_start),
                "test_end": str(w.test_end),
                "params": params,
            }
            for w, params in result.per_window_params
        ],
    }
    (out_dir / "chosen_params.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return html_path
```

- [ ] **Step 6: Update `quant/backtest/__init__.py` to re-export the public API**

Replace the contents with:

```python
"""Backtest engine, walk-forward harness, and tear-sheet generator."""

from __future__ import annotations

from quant.backtest.engine import BacktestConfig, BacktestResult, run_backtest
from quant.backtest.tearsheet import write_tearsheet
from quant.backtest.walkforward import (
    WalkforwardResult,
    WalkforwardWindow,
    iter_windows,
    run_walkforward,
    select_best_params,
)

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "WalkforwardResult",
    "WalkforwardWindow",
    "iter_windows",
    "run_backtest",
    "run_walkforward",
    "select_best_params",
    "write_tearsheet",
]
```

- [ ] **Step 7: Run tests to verify pass**

```bash
uv run pytest tests/backtest/test_tearsheet.py -v
```

Expected: 5 passed.

- [ ] **Step 8: Run full suite + lint + types**

```bash
uv run pytest -q -m "not network and not alpaca and not slow"
uv run ruff check . && uv run ruff format --check . && uv run mypy quant
```

Expected: green.

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml uv.lock quant/backtest/__init__.py quant/backtest/tearsheet.py quant/backtest/templates/tearsheet.html.j2 tests/backtest/test_tearsheet.py
git commit -m "feat(backtest): HTML tear-sheet generator (matplotlib + jinja2)"
```

---

## Task 10: Data refresh helper + CLI wiring

**Files:**
- Create: `quant/data/refresh.py`
- Modify: `quant/cli.py` (replace stubs for `data refresh`, `backtest`, `tearsheet`)
- Create: `tests/data/test_refresh.py`
- Modify: `tests/test_cli.py`

**Context.** Three CLI commands now functional:

1. `quant data refresh` — fetches the ETF universe (always) and the S&P 500 snapshot (always), plus the union of `spec.universe` across registered strategies (currently empty, will grow in Plans 4/5). Writes parquet bar caches via the existing `get_bars`.

2. `quant backtest <slug>` — runs full walk-forward + writes tear-sheet. Since no concrete strategies are registered until Plan 4, this command can be smoke-tested with the test strategy via a `--use-test-strategy` flag, *or* it can simply require a registered strategy and report `(no strategies registered yet)` from the existing helper. Plan choice: keep it strict (real slugs only); the smoke test below registers the `EqualWeightStrategy` temporarily.

3. `quant tearsheet <slug>` — opens `data/backtests/<slug>/tearsheet.html` in the default browser via `webbrowser.open`.

For real `quant backtest <slug>` to function in this plan, the strategy must be registered AND provide a way for the CLI to construct it with bars. Since Foundation strategies follow the no-bars-constructor pattern, we introduce a per-strategy `build(bars: pd.DataFrame) -> Strategy` *optional* classmethod on `Strategy`, defaulting to `return cls(params=params)`. The test override in Plan 2 is a no-op since Plan 2 has no registered strategies; the wiring is verified via the smoke test.

- [ ] **Step 1: Add `Strategy.build` classmethod**

Modify `quant/strategies/base.py` — append inside the `Strategy` class:

```python
    @classmethod
    def build(cls, bars: pd.DataFrame, params: dict[str, Any] | None = None) -> "Strategy":
        """Construct a strategy instance, given the bars frame and params.

        Default: ignore bars and instantiate with params only. Strategies that
        need bars at construction (e.g. for caching signal series) override.
        """
        del bars  # unused at the base class
        return cls(params=params)
```

(The `import pandas as pd` is already at the top of `base.py`.)

- [ ] **Step 2: Write failing tests for refresh + CLI**

`tests/data/test_refresh.py`:

```python
"""Tests for quant.data.refresh."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from quant.data.refresh import refresh_caches


def test_refresh_calls_get_bars_with_union_of_universes(
    tmp_data_dir: Path,
    fake_env: None,
) -> None:
    fake_aapl = pd.DataFrame(
        {"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1]},
        index=pd.DatetimeIndex([pd.Timestamp("2024-01-02")], name="timestamp"),
    )
    fake_spy = pd.DataFrame(
        {"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1]},
        index=pd.DatetimeIndex([pd.Timestamp("2024-01-02")], name="timestamp"),
    )

    def fake_fetch(symbols, start, end, settings):
        return {sym: fake_aapl if sym != "SPY" else fake_spy for sym in symbols}

    with (
        patch("quant.data.bars._fetch_alpaca", side_effect=fake_fetch),
        patch("quant.data.universe.sp500_constituents", return_value=["AAPL", "MSFT"]),
    ):
        report = refresh_caches(start=date(2024, 1, 1), end=date(2024, 1, 5))

    assert report.symbols_fetched >= 8 + 2  # 8 ETFs + at least AAPL,MSFT
    # ETF universe always included:
    for etf in ("SPY", "TLT", "GLD"):
        assert etf in report.symbols
```

Modify `tests/test_cli.py` — add CLI tests:

```python
"""Tests for the Click CLI."""

# ... existing test imports + body unchanged ...

from datetime import date
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

import pandas as pd
from click.testing import CliRunner

from quant.cli import cli
from quant.strategies import REGISTRY
from quant.strategies.base import Strategy, StrategySpec


class _CLIToyStrategy(Strategy):
    spec: ClassVar[StrategySpec] = StrategySpec(
        slug="cli-toy",
        name="CLI Toy",
        description="-",
        universe=["AAA", "BBB"],
        rebalance_frequency="monthly",
    )
    default_params: ClassVar[dict[str, object]] = {}

    def generate_signals(self, asof: date) -> pd.Series:
        return pd.Series({"AAA": 1.0})

    def target_positions(self, asof: date, equity: float) -> dict[str, int]:
        return {"AAA": 1}

    @classmethod
    def build(cls, bars: pd.DataFrame, params: dict[str, object] | None = None) -> "Strategy":
        return cls(params=params)


def test_data_refresh_command(tmp_data_dir: Path, fake_env: None) -> None:
    runner = CliRunner()
    with patch("quant.data.refresh.refresh_caches") as mock_refresh:
        mock_refresh.return_value = type(
            "R", (), {"symbols": ["SPY"], "symbols_fetched": 1, "rows_total": 5, "elapsed_s": 0.1}
        )()
        result = runner.invoke(cli, ["data", "refresh"])
    assert result.exit_code == 0, result.output
    assert "symbols_fetched" in result.output.lower() or "fetched" in result.output.lower()


def test_backtest_command_unknown_strategy(fake_env: None) -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["backtest", "definitely-not-a-strategy"])
    assert result.exit_code != 0
    assert "unknown strategy" in result.output.lower()


def test_backtest_command_runs_registered_strategy(
    tmp_data_dir: Path,
    fake_env: None,
) -> None:
    REGISTRY["cli-toy"] = _CLIToyStrategy
    try:
        runner = CliRunner()
        with patch("quant.data.bars._fetch_alpaca") as mock_alpaca:
            # Provide enough synthetic data via the bars fetch mock.
            dates = pd.bdate_range("2010-01-01", "2024-12-31")
            df = pd.DataFrame(
                {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1},
                index=pd.DatetimeIndex(dates, name="timestamp"),
            )
            mock_alpaca.return_value = {"AAA": df, "BBB": df}
            result = runner.invoke(cli, ["backtest", "cli-toy", "--quick"])
        assert result.exit_code == 0, result.output
        # Check the tear-sheet was written:
        out_dir = tmp_data_dir / "backtests" / "cli-toy"
        assert (out_dir / "tearsheet.html").exists()
    finally:
        REGISTRY.pop("cli-toy", None)


def test_tearsheet_command_opens_html(tmp_data_dir: Path, fake_env: None) -> None:
    # Pre-create a fake tear-sheet
    out_dir = tmp_data_dir / "backtests" / "stub"
    out_dir.mkdir(parents=True)
    (out_dir / "tearsheet.html").write_text("<html></html>")

    runner = CliRunner()
    with patch("webbrowser.open") as mock_open:
        result = runner.invoke(cli, ["tearsheet", "stub"])
    assert result.exit_code == 0, result.output
    mock_open.assert_called_once()


def test_tearsheet_command_missing_file(tmp_data_dir: Path, fake_env: None) -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["tearsheet", "nonexistent"])
    assert result.exit_code != 0
    assert "tearsheet" in result.output.lower()
```

Note: the existing `test_cli.py` from Plan 1 needs to be kept; only add the above. Also note we register `cli-toy` directly into `REGISTRY` (not via `@register`) because we want CLI tests to be self-contained — the cleanup in `finally` removes it.

- [ ] **Step 3: Run failing tests**

```bash
uv run pytest tests/data/test_refresh.py tests/test_cli.py -v
```

Expected: ImportError on `quant.data.refresh.refresh_caches`; CLI tests fail because subcommands are still stubs.

- [ ] **Step 4: Implement refresh_caches**

`quant/data/refresh.py`:

```python
"""Bulk bar-cache refresh for the union of standard + registered universes."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date

from quant.data.bars import BarRequest, get_bars
from quant.data.universe import ETF_UNIVERSE, sp500_constituents
from quant.strategies import REGISTRY
from quant.util.logging import logger


@dataclass(frozen=True)
class RefreshReport:
    symbols: list[str]
    symbols_fetched: int
    rows_total: int
    elapsed_s: float
    errors: list[str] = field(default_factory=list)


def _union_universe() -> list[str]:
    """Standard universes (ETFs + S&P 500) ∪ registered strategy universes."""
    symbols: set[str] = set(ETF_UNIVERSE)
    try:
        symbols.update(sp500_constituents())
    except Exception as exc:  # noqa: BLE001 - network-flaky fallback acceptable here
        logger.warning("Could not fetch S&P 500 constituents: {}", exc)

    for cls in REGISTRY.values():
        symbols.update(cls.spec.universe)
    return sorted(symbols)


def refresh_caches(
    start: date,
    end: date,
    *,
    chunk_size: int = 50,
) -> RefreshReport:
    """Fetch bars for the union universe over [start, end] and update the parquet cache.

    Symbols are fetched in chunks of `chunk_size` to keep individual API calls small.
    Errors are collected and returned; individual chunk failures do NOT stop the refresh.
    """
    t0 = time.monotonic()
    symbols = _union_universe()
    errors: list[str] = []
    rows_total = 0
    fetched_count = 0

    for i in range(0, len(symbols), chunk_size):
        chunk = symbols[i : i + chunk_size]
        try:
            df = get_bars(BarRequest(symbols=chunk, start=start, end=end))
            rows_total += int(df.shape[0]) if not df.empty else 0
            fetched_count += len(chunk)
            logger.info("refresh chunk {}-{}: {} symbols", i, i + len(chunk), len(chunk))
        except Exception as exc:  # noqa: BLE001 - chunk-level resilience
            msg = f"chunk {i}-{i + len(chunk)}: {exc!r}"
            errors.append(msg)
            logger.error(msg)

    return RefreshReport(
        symbols=symbols,
        symbols_fetched=fetched_count,
        rows_total=rows_total,
        elapsed_s=time.monotonic() - t0,
        errors=errors,
    )
```

- [ ] **Step 5: Wire the CLI subcommands**

Modify `quant/cli.py`:

1. Add imports at the top:

```python
import webbrowser
from datetime import date, timedelta

from quant.backtest import BacktestConfig, run_walkforward, write_tearsheet
from quant.data.bars import BarRequest, get_bars
from quant.data.refresh import refresh_caches
```

2. Replace the `backtest` command body:

```python
@cli.command(help="Run full walk-forward backtest for <strategy> and write tear-sheet.")
@click.argument("strategy")
@click.option("--quick", is_flag=True, help="Skip combinatorial CV + bootstrap (Plan 3-only knobs).")
@click.option(
    "--start", default="2010-01-01", show_default=True,
    help="History start date (YYYY-MM-DD)."
)
@click.option(
    "--end", default=None, help="History end date (YYYY-MM-DD). Default: today."
)
def backtest(strategy: str, quick: bool, start: str, end: str | None) -> None:
    _require_strategy(strategy)
    if quick:
        # Plan 2 has no Plan-3 knobs to skip; flag is reserved for forward-compat.
        logger.info("--quick: skipping Plan-3 validation layers (none active in this plan).")

    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end) if end else date.today()

    settings = Settings()  # type: ignore[call-arg]
    strategy_cls = REGISTRY[strategy]
    universe = list(strategy_cls.spec.universe)

    console.print(f"[bold]Fetching bars for {len(universe)} symbols...[/bold]")
    bars = get_bars(BarRequest(symbols=universe, start=start_date, end=end_date))
    if bars.empty:
        raise click.ClickException(
            f"No bars returned for {strategy!r} over {start_date}..{end_date}."
        )

    def factory(params: dict[str, object], bars_for_strategy):  # type: ignore[no-untyped-def]
        return strategy_cls.build(bars=bars_for_strategy, params=params)

    console.print("[bold]Running walk-forward...[/bold]")
    result = run_walkforward(
        strategy_factory=factory,
        param_grid={},  # concrete strategies will define their own; Plan 2 uses defaults only
        bars=bars,
        start=start_date,
        end=end_date,
        config=BacktestConfig(),
    )

    out_dir = settings.data_dir / "backtests" / strategy
    html_path = write_tearsheet(
        result=result,
        slug=strategy,
        strategy_name=strategy_cls.spec.name,
        out_dir=out_dir,
    )
    console.print(f"[green]Wrote {html_path}[/green]")
```

3. Replace the `tearsheet` command body:

```python
@cli.command(help="Open the HTML tear-sheet for <strategy> in your default browser.")
@click.argument("strategy")
def tearsheet(strategy: str) -> None:
    _require_strategy(strategy)
    settings = Settings()  # type: ignore[call-arg]
    path = settings.data_dir / "backtests" / strategy / "tearsheet.html"
    if not path.exists():
        raise click.ClickException(
            f"No tearsheet at {path}. Run `quant backtest {strategy}` first."
        )
    webbrowser.open(path.resolve().as_uri())
    console.print(f"Opened {path}")
```

4. Replace the `data refresh` command body:

```python
@data.command("refresh", help="Refresh bar caches for ETFs + S&P 500 + registered universes.")
@click.option(
    "--start", default="2010-01-01", show_default=True, help="Start date (YYYY-MM-DD)."
)
@click.option(
    "--end", default=None, help="End date (YYYY-MM-DD). Default: today."
)
def data_refresh(start: str, end: str | None) -> None:
    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end) if end else date.today()
    console.print(f"[bold]Refreshing caches over {start_date}..{end_date}...[/bold]")
    report = refresh_caches(start=start_date, end=end_date)
    table = Table(title="Refresh report")
    table.add_column("Field")
    table.add_column("Value", justify="right")
    table.add_row("symbols_fetched", str(report.symbols_fetched))
    table.add_row("rows_total", str(report.rows_total))
    table.add_row("elapsed_s", f"{report.elapsed_s:.1f}")
    table.add_row("errors", str(len(report.errors)))
    console.print(table)
    if report.errors:
        console.print("[red]First 5 errors:[/red]")
        for err in report.errors[:5]:
            console.print(f"  {err}")
```

The old `backtest`, `tearsheet`, and `data_refresh` stub bodies that raised `ClickException` are now replaced. The other stubs (`validate`, `rebalance`, `journal`, `monitor`) stay as Plan-3/Plan-6 placeholders.

- [ ] **Step 6: Run tests to verify pass**

```bash
uv run pytest tests/data/test_refresh.py tests/test_cli.py -v
```

Expected: all pass.

- [ ] **Step 7: Run full suite + lint + types**

```bash
uv run pytest -q -m "not network and not alpaca and not slow"
uv run ruff check . && uv run ruff format --check . && uv run mypy quant
```

Expected: green.

- [ ] **Step 8: Smoke-test the CLI end-to-end with live Alpaca paper data**

```bash
uv run quant data refresh --start 2024-01-01 --end 2024-01-31
```

Expected: table with symbols_fetched > 0 and errors=0 (or only network errors if Alpaca is slow).

Then verify cached files landed:

```bash
ls -lh data/raw | head -20
```

Expected: parquet files for SPY, TLT, IEF, etc.

- [ ] **Step 9: Commit**

```bash
git add quant/strategies/base.py quant/data/refresh.py quant/cli.py tests/data/test_refresh.py tests/test_cli.py
git commit -m "feat(cli): wire data refresh + backtest + tearsheet commands"
```

---

## Task 11: README + finishing

**Files:**
- Modify: `README.md`

**Context.** Document the now-working backtest pipeline so a future contributor can run `quant backtest <slug>` without re-reading the plan.

- [ ] **Step 1: Update README — replace the "Backtest" section (or create one)**

Add (or replace) a section after the existing CLI section in `README.md`:

```markdown
## Running a backtest

Once at least one strategy is registered (Plans 4-5), run the full walk-forward
pipeline:

    uv run quant backtest <slug>

This:

1. Fetches daily bars for the strategy's universe (Alpaca primary, yfinance backup).
2. Runs walk-forward analysis: rolling 5-year train / 1-year test / 6-month step.
3. For each train window, grid-searches the strategy's parameter space and picks
   the best by in-sample Sharpe.
4. Stitches the OOS test segments into one continuous equity curve.
5. Writes the HTML tear-sheet + sidecar parquet + JSON to
   `data/backtests/<slug>/`.

Open the tear-sheet:

    uv run quant tearsheet <slug>

Refresh the bar cache for the union of all registered universes + ETFs +
S&P 500 (run this before a fresh backtest if the cache is stale):

    uv run quant data refresh --start 2010-01-01

The tear-sheet renders: OOS equity curve, drawdown, monthly returns heatmap,
returns distribution histogram, plus the per-window chosen-parameters table.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: backtest pipeline usage in README"
```

- [ ] **Step 3: Final integration check + branch summary**

```bash
uv run pytest -q -m "not network and not alpaca and not slow"
uv run ruff check . && uv run ruff format --check .
uv run mypy quant
uv run quant --help
uv run quant strategies        # should print the empty registered-strategies table
```

Then:

```bash
git log --oneline main..HEAD
```

Expected: ~10 commits on the feature branch covering Tasks 1–11.

---

## Self-review checklist (after completing all tasks)

1. **Spec coverage:**
   - §6 `engine.py` → ✅ Task 6 (`run_backtest`)
   - §6 `walkforward.py` → ✅ Tasks 7, 8 (`iter_windows`, `select_best_params`, `run_walkforward`)
   - §6 `tearsheet.py` → ✅ Task 9 (`write_tearsheet`)
   - §6 `data/refresh` wiring → ✅ Task 10 (`refresh_caches` + CLI)
   - §4 items 1 (walk-forward) and 9 (tear-sheet) → ✅
   - §4 items 2-8 (CPCV, DSR, MC bootstrap, regime, OOS holdout, cost sensitivity, PSR) → deferred to Plan 3 (called out in header)
   - §5.1 `quant backtest`, `quant tearsheet`, `quant data refresh` → ✅
   - §8 ≥80% coverage requirement: should be met (engine and walkforward are heavily tested; tear-sheet has 5 tests covering both populated and empty paths).

2. **Type consistency:**
   - `BacktestConfig.execution: Literal["next_open", "close"]` used the same in Task 5 dataclass and Task 6 run loop ✓
   - `BacktestResult.equity_curve` is `pd.Series` everywhere ✓
   - `Strategy.build(bars, params)` signature consistent between Task 10 base addition and CLI factory ✓
   - `param_grid: dict[str, Sequence[Any]]` consistent across `select_best_params` and `run_walkforward` ✓
   - `WalkforwardResult.per_window_params: list[tuple[WalkforwardWindow, dict[str, Any]]]` consistent between Task 8 dataclass and Task 9 template renderer ✓

3. **No placeholders:** every code step shows complete code (no "TBD", no "similar to Task N").

4. **Concrete strategies dependency:** Plan 2 CLI tests register `_CLIToyStrategy` ad-hoc, which exercises the path; once Plan 4 lands, the same CLI commands will work against real strategies without further changes.

---

## Branch-completion handoff

Per the workflow established in Plan 1: when all tasks above are green on `feat/backtest-engine` (lint + mypy + pytest with 60+22 new tests ≈ 82 total passing), use `superpowers:finishing-a-development-branch` to merge to `main` and delete the feature branch. Then Plan 3 (combinatorial purged CV + deflated Sharpe + bootstrap + regime stress) can begin against the same engine API exposed in `quant/backtest/__init__.py`.
