# Market-Impact Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a size-dependent square-root market-impact cost to the backtest engine (gap #2, slice 2b), on by default, layered on the existing flat half-spread.

**Architecture:** A new pure module `quant/backtest/impact.py` computes `market_impact_bps` from participation (= trade notional / trailing PIT dollar-ADV) and a `trailing_dollar_adv` helper. `apply_costs` gains an additive `impact_bps` parameter; `_execute_fill` computes ADV + notional + impact per fill and passes it in. Two new `BacktestConfig` fields (`impact_coef_bps=100.0`, `adv_window=21`) drive it.

**Tech Stack:** Python 3, numpy, pandas, pytest, uv (runner), ruff + mypy (lint/type). No new dependencies.

Spec: `docs/superpowers/specs/2026-05-28-market-impact-model-design.md`

---

### Task 1: Pure impact module — `market_impact_bps` + `trailing_dollar_adv`

**Files:**
- Create: `quant/backtest/impact.py`
- Test: `tests/backtest/test_impact.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/backtest/test_impact.py`:

```python
"""Tests for the pure market-impact model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.backtest.impact import market_impact_bps, trailing_dollar_adv


def _bars(symbol: str, closes: list[float], volumes: list[int]) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=len(closes))
    df = pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes,
         "volume": np.array(volumes, dtype=np.int64)},
        index=dates,
    )
    df.index.name = "timestamp"
    return pd.concat({symbol: df}, axis=1)


# ---- market_impact_bps ----

def test_participation_one_returns_coef():
    # notional == adv -> participation 1 -> coef * sqrt(1) = coef
    assert market_impact_bps(1_000_000.0, 1_000_000.0, 100.0) == pytest.approx(100.0)


def test_participation_quarter_is_half_coef():
    # participation 0.25 -> sqrt = 0.5 -> coef * 0.5
    assert market_impact_bps(250_000.0, 1_000_000.0, 100.0) == pytest.approx(50.0)


def test_impact_is_concave_in_size():
    small = market_impact_bps(1_000_000.0, 1_000_000_000.0, 100.0)
    big = market_impact_bps(2_000_000.0, 1_000_000_000.0, 100.0)
    assert big < 2.0 * small  # square-root: doubling size less than doubles impact


def test_nonpositive_adv_is_zero():
    assert market_impact_bps(1_000_000.0, 0.0, 100.0) == 0.0
    assert market_impact_bps(1_000_000.0, -5.0, 100.0) == 0.0


def test_nonpositive_notional_is_zero():
    assert market_impact_bps(0.0, 1_000_000.0, 100.0) == 0.0
    assert market_impact_bps(-10.0, 1_000_000.0, 100.0) == 0.0


def test_nonfinite_inputs_are_zero():
    assert market_impact_bps(float("nan"), 1_000_000.0, 100.0) == 0.0
    assert market_impact_bps(1_000_000.0, float("inf"), 100.0) == 0.0


# ---- trailing_dollar_adv ----

def test_adv_mean_over_strictly_prior_window():
    # closes 10,10,10; volumes 100,200,300 -> dollar vol 1000,2000,3000.
    bars = _bars("AAA", [10.0, 10.0, 10.0], [100, 200, 300])
    fill_ts = bars.index[2]  # third bar; prior bars are index 0 and 1
    # window 5 but only 2 prior bars -> mean(1000, 2000) = 1500
    assert trailing_dollar_adv(bars, "AAA", fill_ts, window=5) == pytest.approx(1500.0)


def test_adv_excludes_fill_bar_volume_pit():
    # Spiking the FILL bar's own volume must not change ADV (uses strictly-prior bars).
    bars = _bars("AAA", [10.0, 10.0, 10.0], [100, 200, 10_000_000])
    fill_ts = bars.index[2]
    assert trailing_dollar_adv(bars, "AAA", fill_ts, window=5) == pytest.approx(1500.0)


def test_adv_respects_window_length():
    bars = _bars("AAA", [10.0, 10.0, 10.0, 10.0], [100, 200, 300, 400])
    fill_ts = bars.index[3]  # prior: 100,200,300 -> dollar 1000,2000,3000
    # window 2 -> last two prior bars: 2000,3000 -> mean 2500
    assert trailing_dollar_adv(bars, "AAA", fill_ts, window=2) == pytest.approx(2500.0)


def test_adv_no_prior_history_is_zero():
    bars = _bars("AAA", [10.0, 10.0], [100, 200])
    assert trailing_dollar_adv(bars, "AAA", bars.index[0], window=5) == 0.0


def test_adv_missing_symbol_is_zero():
    bars = _bars("AAA", [10.0, 10.0], [100, 200])
    assert trailing_dollar_adv(bars, "ZZZ", bars.index[1], window=5) == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/backtest/test_impact.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'quant.backtest.impact'`

- [ ] **Step 3: Write the module**

Create `quant/backtest/impact.py`:

```python
"""Square-root market-impact model + trailing dollar-ADV.

Pure functions taking plain values / the bars frame — no ``BacktestConfig``
import, so ``engine.py`` can import this without a circular dependency.

The impact is the size-dependent term added on top of the engine's flat
half-spread (``slippage_bps``). Undefined / degenerate inputs return 0.0 impact
(cannot estimate) rather than raising, mirroring the engine's tolerance for
sparse bars.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def market_impact_bps(
    trade_notional: float,
    adv_dollar: float,
    impact_coef_bps: float,
) -> float:
    """Square-root impact in bps: ``impact_coef_bps * sqrt(notional / adv)``.

    ``impact_coef_bps`` is the impact at 100%-of-ADV participation. Returns 0.0
    when any input is non-finite, or when ``trade_notional`` or ``adv_dollar``
    is non-positive (impact cannot be estimated).
    """
    if not (
        math.isfinite(trade_notional)
        and math.isfinite(adv_dollar)
        and math.isfinite(impact_coef_bps)
    ):
        return 0.0
    if trade_notional <= 0.0 or adv_dollar <= 0.0:
        return 0.0
    participation = trade_notional / adv_dollar
    return float(impact_coef_bps * math.sqrt(participation))


def trailing_dollar_adv(
    bars: pd.DataFrame,
    symbol: str,
    fill_ts: pd.Timestamp,
    window: int,
) -> float:
    """Mean ``close * volume`` over the ``window`` bars strictly before ``fill_ts``.

    PIT: the fill bar's own volume is excluded (only rows with index < fill_ts).
    Returns 0.0 if the (symbol, close/volume) columns are absent, there is no
    prior history, or every prior dollar-volume is non-finite.
    """
    close_col = (symbol, "close")
    vol_col = (symbol, "volume")
    if close_col not in bars.columns or vol_col not in bars.columns:
        return 0.0
    prior_index = bars.index[bars.index < fill_ts]
    if len(prior_index) == 0:
        return 0.0
    tail = prior_index[-window:]
    closes = bars[close_col].loc[tail].to_numpy(dtype=float)
    volumes = bars[vol_col].loc[tail].to_numpy(dtype=float)
    dollar_vol = closes * volumes
    dollar_vol = dollar_vol[np.isfinite(dollar_vol)]
    if len(dollar_vol) == 0:
        return 0.0
    return float(dollar_vol.mean())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/backtest/test_impact.py -v`
Expected: PASS (11 passed)

- [ ] **Step 5: Lint + type the new files**

Run:
```bash
uv run mypy quant/backtest/impact.py
uv run ruff check quant/backtest/impact.py tests/backtest/test_impact.py
uv run ruff format --check quant/backtest/impact.py tests/backtest/test_impact.py
```
Expected: clean. (If format --check reports a file would reformat, run `uv run ruff format` on those two files and re-check.)

- [ ] **Step 6: Commit**

```bash
git add quant/backtest/impact.py tests/backtest/test_impact.py
git commit -m "feat(impact): square-root market-impact model + trailing dollar-ADV

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Add impact fields to `BacktestConfig`

**Files:**
- Modify: `quant/backtest/engine.py` (`BacktestConfig` ~23-35)
- Test: `tests/backtest/test_engine_costs.py` (`test_default_config_values`)

- [ ] **Step 1: Extend the default-value test**

In `tests/backtest/test_engine_costs.py`, `test_default_config_values` currently asserts the existing defaults (starting_equity, slippage_bps, commission_bps, execution, annual_borrow_bps, annual_financing_bps). Add two assertions at the end:

```python
    assert cfg.impact_coef_bps == 100.0
    assert cfg.adv_window == 21
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/backtest/test_engine_costs.py::test_default_config_values -v`
Expected: FAIL — `AttributeError: 'BacktestConfig' object has no attribute 'impact_coef_bps'`

- [ ] **Step 3: Add the fields**

In `quant/backtest/engine.py`, the current `BacktestConfig` is:

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

Insert the two impact fields after `annual_financing_bps` and before `execution` (keep `execution` last):

```python
    annual_borrow_bps: float = 50.0
    annual_financing_bps: float = 200.0
    # Market impact (gap #2, slice 2b): size-dependent square-root cost added on
    # top of the flat slippage_bps half-spread. impact_coef_bps is the impact at
    # 100%-of-ADV participation (provisional calibration). On by default.
    impact_coef_bps: float = 100.0
    adv_window: int = 21
    execution: Literal["next_open", "close"] = "next_open"
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/backtest/test_engine_costs.py::test_default_config_values -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add quant/backtest/engine.py tests/backtest/test_engine_costs.py
git commit -m "feat(impact): add impact_coef_bps + adv_window to BacktestConfig

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Wire impact into `apply_costs` + `_execute_fill`

**Files:**
- Modify: `quant/backtest/engine.py` (`apply_costs` ~58-79, `_execute_fill` ~139-160)
- Test: `tests/backtest/test_engine_costs.py` (apply_costs unit), `tests/backtest/test_engine_impact.py` (new integration), `tests/backtest/test_engine_run.py` (intent fix)

- [ ] **Step 1: Write the failing apply_costs unit test**

In `tests/backtest/test_engine_costs.py`, add:

```python
def test_impact_adds_to_spread() -> None:
    cfg = BacktestConfig(slippage_bps=10.0, commission_bps=0.0)
    fill = apply_costs(qty=100, mid_price=50.0, side="buy", config=cfg, impact_bps=20.0)
    # total slip = (10 + 20) / 1e4 = 0.003 -> fill = 50.00 * 1.003 = 50.15
    assert fill.fill_price == pytest.approx(50.15, abs=1e-6)
    assert fill.slippage_cost == pytest.approx(100 * (50.15 - 50.0), abs=1e-6)


def test_impact_defaults_to_zero() -> None:
    cfg = BacktestConfig(slippage_bps=10.0, commission_bps=0.0)
    fill = apply_costs(qty=100, mid_price=50.0, side="buy", config=cfg)
    # No impact_bps passed -> behaves exactly as before (50 * 1.001).
    assert fill.fill_price == pytest.approx(50.05, abs=1e-6)
```

- [ ] **Step 2: Write the failing integration tests**

Create `tests/backtest/test_engine_impact.py`:

```python
"""Integration tests: market impact charged through run_backtest."""

from __future__ import annotations

from datetime import date
from typing import ClassVar

import numpy as np
import pandas as pd

from quant.backtest.engine import BacktestConfig, run_backtest
from quant.strategies.base import Strategy, StrategySpec


def _bars(symbol: str, price: float, volume: int) -> pd.DataFrame:
    """Flat-price bars for 2024-Q1 with a constant per-bar volume."""
    dates = pd.bdate_range("2024-01-02", "2024-03-29")
    df = pd.DataFrame(
        {f: np.full(len(dates), price) for f in ("open", "high", "low", "close")}
        | {"volume": np.full(len(dates), volume, dtype=np.int64)},
        index=dates,
    )
    df.index.name = "timestamp"
    return pd.concat({symbol: df}, axis=1)


class _FixedLongStrategy(Strategy):
    """Test-only: hold a fixed 1,000-share long in AAA at every rebalance."""

    spec: ClassVar[StrategySpec] = StrategySpec(
        slug="fixed-long-test",
        name="Fixed Long (test)",
        description="Test fixture: constant 1,000-share long in AAA.",
        universe=["AAA"],
        rebalance_frequency="monthly",
    )
    default_params: ClassVar[dict[str, object]] = {}

    def __init__(self, bars: pd.DataFrame) -> None:
        super().__init__(params=None)
        self._bars = bars

    def generate_signals(self, asof: date) -> pd.Series:
        return pd.Series({"AAA": 1.0})

    def target_positions(self, asof: date, equity: float) -> dict[str, int]:
        return {"AAA": 1000}


def _cfg(impact_coef_bps: float, adv_window: int = 21) -> BacktestConfig:
    # next_open execution so the first rebalance (bar 1) fills on bar 2, which has
    # bar 1 as strictly-prior history for ADV (a close-execution fill on bar 1 would
    # have no prior history -> ADV 0 -> impact 0). Zero spread/commission/financing
    # so impact is the only cost in play.
    return BacktestConfig(
        starting_equity=10_000_000.0,
        slippage_bps=0.0,
        commission_bps=0.0,
        annual_borrow_bps=0.0,
        annual_financing_bps=0.0,
        impact_coef_bps=impact_coef_bps,
        adv_window=adv_window,
        execution="next_open",
    )


def test_low_adv_costs_more_than_high_adv():
    # Same 1,000-share buy; lower ADV (lower volume) -> higher participation -> more impact.
    low = run_backtest(_FixedLongStrategy(_bars("AAA", 100.0, 5_000)),
                       _bars("AAA", 100.0, 5_000), _cfg(100.0),
                       date(2024, 1, 2), date(2024, 3, 29))
    high = run_backtest(_FixedLongStrategy(_bars("AAA", 100.0, 50_000_000)),
                        _bars("AAA", 100.0, 50_000_000), _cfg(100.0),
                        date(2024, 1, 2), date(2024, 3, 29))
    assert low.ending_equity < high.ending_equity


def test_zero_coef_disables_impact():
    # impact_coef_bps=0 with zero spread/commission/financing and flat prices ->
    # equity is preserved bar-for-bar (± integer-share rounding), i.e. no cost.
    bars = _bars("AAA", 100.0, 5_000)
    res = run_backtest(_FixedLongStrategy(bars), bars, _cfg(0.0),
                       date(2024, 1, 2), date(2024, 3, 29))
    assert all(abs(eq - 10_000_000.0) < 100.0 for eq in res.equity_curve)


def test_impact_uses_prior_volume_not_fill_bar_pit():
    # Spiking the FILL bar's own volume must not change the impact charged
    # (ADV uses strictly-prior bars). Find the first trade date, spike its volume,
    # and confirm ending equity is unchanged vs the unspiked run.
    bars = _bars("AAA", 100.0, 5_000)
    res = run_backtest(_FixedLongStrategy(bars), bars, _cfg(100.0),
                       date(2024, 1, 2), date(2024, 3, 29))
    first_trade_ts = res.trades["date"].min()

    spiked = _bars("AAA", 100.0, 5_000)
    spiked.loc[first_trade_ts, ("AAA", "volume")] = 9_999_999_999
    res_spiked = run_backtest(_FixedLongStrategy(spiked), spiked, _cfg(100.0),
                              date(2024, 1, 2), date(2024, 3, 29))
    assert res_spiked.ending_equity == res.ending_equity
```

- [ ] **Step 3: Run both new test files to verify they fail**

Run: `uv run pytest tests/backtest/test_engine_costs.py::test_impact_adds_to_spread tests/backtest/test_engine_impact.py -v`
Expected: FAIL — `apply_costs() got an unexpected keyword argument 'impact_bps'` (and the integration tests error/fail).

- [ ] **Step 4: Add the `impact_bps` parameter to `apply_costs`**

In `quant/backtest/engine.py`, the current `apply_costs` is:

```python
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
```

Change it to accept `impact_bps` (additive on the half-spread) and update the docstring:

```python
def apply_costs(
    qty: int, mid_price: float, side: Side, config: BacktestConfig, impact_bps: float = 0.0
) -> FillReport:
    """Move the mid-price by spread + market impact; commission as bps of notional.

    The fill price moves by ``(slippage_bps + impact_bps) / 1e4``: ``slippage_bps``
    is the flat half-spread, ``impact_bps`` the size-dependent market impact
    computed upstream (0.0 when impact is disabled or ADV is unknown).
    ``slippage_cost`` therefore captures spread + impact combined.

    Buy: fill_price = mid * (1 + (slippage_bps + impact_bps) / 1e4)
    Sell: fill_price = mid * (1 - (slippage_bps + impact_bps) / 1e4)
    Commission: |qty| * fill_price * commission_bps / 1e4
    """
    if side not in ("buy", "sell"):
        raise ValueError(f"Unknown side {side!r}; expected 'buy' or 'sell'")
    if qty == 0:
        return FillReport(fill_price=mid_price, slippage_cost=0.0, commission_cost=0.0)

    slip = (config.slippage_bps + impact_bps) / 1e4
    sign = +1.0 if side == "buy" else -1.0
    fill_price = mid_price * (1.0 + sign * slip)
    slippage_cost = abs(qty) * abs(fill_price - mid_price)
    commission_cost = abs(qty) * fill_price * config.commission_bps / 1e4
    return FillReport(
        fill_price=fill_price,
        slippage_cost=slippage_cost,
        commission_cost=commission_cost,
    )
```

- [ ] **Step 5: Add the import + wire `_execute_fill`**

At the TOP of `quant/backtest/engine.py`, add the import (impact.py imports nothing from engine, so no circular import):

```python
from quant.backtest.impact import market_impact_bps, trailing_dollar_adv
```

The current `_execute_fill` (nested in `run_backtest`) starts:

```python
    def _execute_fill(ts: pd.Timestamp, sym: str, qty: int, side: Side, mid: float) -> None:
        nonlocal cash
        fill = apply_costs(qty=qty, mid_price=mid, side=side, config=config)
```

Change the body to compute impact before calling `apply_costs`:

```python
    def _execute_fill(ts: pd.Timestamp, sym: str, qty: int, side: Side, mid: float) -> None:
        nonlocal cash
        adv = trailing_dollar_adv(bars, sym, ts, config.adv_window)
        impact = market_impact_bps(abs(qty) * mid, adv, config.impact_coef_bps)
        fill = apply_costs(qty=qty, mid_price=mid, side=side, config=config, impact_bps=impact)
```

(The rest of `_execute_fill` — cash/position updates and the trade record — is unchanged.)

- [ ] **Step 6: Run the new tests to verify they pass**

Run: `uv run pytest tests/backtest/test_engine_costs.py tests/backtest/test_engine_impact.py -v`
Expected: PASS (apply_costs unit tests + 3 integration tests).

- [ ] **Step 7: Intent-fix the zero-cost test + run the full backtest group with triage**

First, the **known intent fix**: `tests/backtest/test_engine_run.py::test_flat_price_zero_costs_preserves_equity_exactly` builds `BacktestConfig(starting_equity=100_000.0, slippage_bps=0.0, commission_bps=0.0)` intending **zero costs**, then asserts equity stays within $100 of start. Impact now defaults on (coef 100), so that config is no longer cost-free. Preserve the test's intent by disabling impact in its config:

```python
    cfg = BacktestConfig(
        starting_equity=100_000.0, slippage_bps=0.0, commission_bps=0.0, impact_coef_bps=0.0
    )
```

Then run the full backtest group:

Run: `uv run pytest tests/backtest/ -q`

Triage any OTHER failures: impact default-on affects every backtest with trades (long and short). If a failing test runs a real strategy and asserts a specific equity/return/Sharpe value, the shift is the intended cost change — update that expected value and quote before/after in your report. If a test's INTENT is "zero costs" (sets slippage_bps=0 + commission_bps=0 and asserts near-exact equity), preserve intent by adding `impact_coef_bps=0.0` to its config (as above) rather than loosening the assertion. If you cannot confidently classify a failure, STOP and report the full output for the controller to triage. Do not loosen tolerances to paper over a real shift.

- [ ] **Step 8: Lint + type**

Run:
```bash
uv run mypy quant/backtest/engine.py
uv run ruff check quant/backtest/engine.py tests/backtest/test_engine_costs.py tests/backtest/test_engine_impact.py tests/backtest/test_engine_run.py
uv run ruff format --check quant/backtest/engine.py tests/backtest/test_engine_costs.py tests/backtest/test_engine_impact.py tests/backtest/test_engine_run.py
```
Expected: clean (reformat only files this task changed if needed).

- [ ] **Step 9: Commit**

```bash
git add quant/backtest/engine.py tests/backtest/test_engine_costs.py tests/backtest/test_engine_impact.py tests/backtest/test_engine_run.py
git commit -m "feat(impact): charge size-scaled market impact per fill in run_backtest

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Full verification

**Files:** none (verification only; plus any shorting/long-strategy test-expectation updates triaged here).

- [ ] **Step 1: Full suite, types, lint, format**

Run each and confirm clean:

```bash
uv run pytest -q
uv run mypy quant
uv run ruff check quant tests
uv run ruff format --check quant tests
```

Expected: full suite passes, mypy strict clean, ruff clean, format clean. The full suite includes validation/governance tests that run real strategies (long and short) through backtests; default-on impact may shift any that assert exact equity/return/Sharpe/gate values. Triage exactly as in Task 3 Step 7: real-strategy cost shift → update the expected value (quote before/after + name the strategy); "zero costs" intent test → add `impact_coef_bps=0.0`; cannot classify → STOP and report. Report the final pass count and every expectation you changed.

- [ ] **Step 2: Commit any triaged test updates**

If Step 1 required test-expectation updates, commit them:

```bash
git add tests/
git commit -m "test(impact): refresh expected values shifted by default-on market impact

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

(If no updates were needed, skip this commit.)

---

## Notes

- **Default-on is intentional** (charter principle 2). Impact is near-zero for the current small/liquid books but is the input to capacity (slice 2c). Governance/validation evidence is refreshed later by re-running `quant validate` + `quant governance refresh` — out of scope here.
- **`slippage_cost` now means spread + impact** (impact is folded into the fill price). No separate `impact_cost` ledger column in this slice.
- **Out of scope:** capacity (slice 2c), per-name volatility scaling, an impact cap, a separate `impact_cost` column, sweeping impact in the cost-sensitivity sweep, permanent-vs-temporary impact split.
- **Push:** all commits are local. Do not push to public `origin/main` without explicit operator approval (granted this session).
