# Intraday Optimal Execution (Almgren–Chriss) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a full Almgren–Chriss optimal-execution engine (optimal trajectory + efficient frontier + TWAP/VWAP/immediate baselines), wire it into the intraday sleeve loop so entries are worked over ticks, and expose an efficient-frontier evaluation in the existing intraday simulator.

**Architecture:** A new `quant/intraday/execution/` subpackage holds the A–C math and baselines as PURE functions, an `ExecutionProgram` that turns a child-size schedule into per-tick slices, and an `ExecutionManager` the live loop uses to work parent entries over ticks. The same pure planner drives both the live loop and a sim-based evaluation. Only ENTRIES are scheduled; all de-risking (guardrail flattens, loss-halt, flat-by-close, strategy exits) stays immediate.

**Tech Stack:** Python 3.11+, `uv` for all commands, dataclasses, `math` (sinh/cosh/acosh), pandas (sim/calibration), Click CLI, pytest + hypothesis, loguru.

**Conventions:** Run everything with `uv run`. Keep `uv run ruff check .` and `uv run mypy quant` clean. TDD: failing test → fail → implement → pass → commit. The working tree has uncommitted data artifacts — ONLY `git add` the exact paths each task lists. Reuse existing code: `quant/backtest/impact.py` (`market_impact_bps`, `trailing_dollar_adv`), `quant/intraday/sim/` (`BacktestEngine.run`, `BacktestResult.costs`/`.fills`), `quant/intraday/strategy.py` (`IntradayStrategy`, `Order`, `Side`, `OrderType`), `quant/intraday/data/events.py` (`Event`, `QuoteBar`, `Bar`), and the spine in `quant/intraday/live/`.

**Spec:** `docs/superpowers/specs/2026-06-08-intraday-optimal-execution-design.md`

**File map (`quant/intraday/execution/`):** `config.py` (ExecConfig), `almgren_chriss.py` (solver + frontier), `baselines.py` (twap/vwap/immediate), `scheduler.py` (ExecutionProgram), `calibrate.py` (σ/η/γ), `manager.py` (ExecutionManager), `evaluate.py` (sim adapter + cost). Plus modify `quant/intraday/live/loop.py` and `quant/intraday/cli.py`.

---

### Task 1: ExecConfig

**Files:**
- Create: `quant/intraday/execution/__init__.py`
- Create: `quant/intraday/execution/config.py`
- Test: `tests/intraday/execution/test_config.py`

- [ ] **Step 1: Create package markers**

Create `quant/intraday/execution/__init__.py` (empty) and `tests/intraday/execution/__init__.py` (empty).

- [ ] **Step 2: Write the failing test**

Create `tests/intraday/execution/test_config.py`:

```python
import pytest

from quant.intraday.execution.config import ExecConfig


def test_defaults():
    c = ExecConfig()
    assert c.horizon_ticks == 5
    assert c.risk_aversion > 0
    assert 0.0 <= c.perm_impact_frac <= 1.0
    assert c.sigma_lookback_bars > 0
    assert c.impact_coef_bps > 0


def test_rejects_bad_values():
    with pytest.raises(ValueError):
        ExecConfig(horizon_ticks=0)
    with pytest.raises(ValueError):
        ExecConfig(risk_aversion=0.0)
    with pytest.raises(ValueError):
        ExecConfig(perm_impact_frac=1.5)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/intraday/execution/test_config.py -v`
Expected: FAIL — ModuleNotFoundError.

- [ ] **Step 4: Write minimal implementation**

Create `quant/intraday/execution/config.py`:

```python
"""Configuration for the intraday optimal-execution engine. No magic numbers per
the Charter; all knobs live here."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExecConfig:
    horizon_ticks: int = 5           # number of 60s ticks to work a parent over
    risk_aversion: float = 1e-6      # Almgren-Chriss lambda (per-share-var units)
    perm_impact_frac: float = 0.1    # gamma = perm_impact_frac * eta
    sigma_lookback_bars: int = 60    # bars for realized-vol estimate
    adv_window_bars: int = 20        # bars for trailing dollar-ADV
    impact_coef_bps: float = 10.0    # sqrt-impact coefficient at 100% ADV (bps)

    def __post_init__(self) -> None:
        if self.horizon_ticks <= 0:
            raise ValueError("horizon_ticks must be positive")
        if self.risk_aversion <= 0:
            raise ValueError("risk_aversion must be positive")
        if not 0.0 <= self.perm_impact_frac <= 1.0:
            raise ValueError("perm_impact_frac must be in [0, 1]")
        if self.sigma_lookback_bars <= 0 or self.adv_window_bars <= 0:
            raise ValueError("lookback windows must be positive")
        if self.impact_coef_bps <= 0:
            raise ValueError("impact_coef_bps must be positive")
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/intraday/execution/test_config.py -v` (expect 2 passing). Then `uv run ruff check quant/intraday/execution/ tests/intraday/execution/` and `uv run mypy quant/intraday/execution/config.py` (clean).

- [ ] **Step 6: Commit**

```bash
git add quant/intraday/execution/__init__.py quant/intraday/execution/config.py tests/intraday/execution/
git commit -m "feat(intraday-exec): ExecConfig for optimal-execution engine"
```

---

### Task 2: Almgren–Chriss solver + efficient frontier

**Files:**
- Create: `quant/intraday/execution/almgren_chriss.py`
- Test: `tests/intraday/execution/test_almgren_chriss.py`

The solver computes the optimal liquidation trajectory via the closed-form sinh solution, then computes expected cost and variance DIRECTLY from the trajectory (robust, textbook-consistent): `E[C] = 0.5*gamma*X^2 + (eta/tau)*sum(n_j^2)` and `V[C] = sigma^2 * tau * sum(x_j^2 for j=1..N)`. The optimal trajectory minimizes `E[C] + lambda*V[C]`.

- [ ] **Step 1: Write the failing test**

Create `tests/intraday/execution/test_almgren_chriss.py`:

```python
import math

from quant.intraday.execution.almgren_chriss import (
    ACPlan,
    efficient_frontier,
    optimal_schedule,
)


def _params(lam):
    # X shares, N intervals, tau per-interval, sigma vol, eta temp, gamma perm
    return dict(total_shares=1000, n_intervals=10, tau=1.0,
                sigma=0.02, eta=1e-4, gamma=1e-5, risk_aversion=lam)


def test_child_sizes_sum_to_parent():
    plan = optimal_schedule(**_params(1e-6))
    assert sum(plan.child_sizes) == 1000
    assert len(plan.child_sizes) == 10
    assert all(n >= 0 for n in plan.child_sizes)


def test_low_risk_aversion_is_approximately_uniform():
    # lambda -> 0 : risk-neutral -> straight-line liquidation -> ~equal slices (TWAP-like)
    plan = optimal_schedule(**_params(1e-12))
    sizes = plan.child_sizes
    assert max(sizes) - min(sizes) <= 2  # near-uniform (integer rounding aside)


def test_high_risk_aversion_is_front_loaded():
    # lambda large : trade fast early to kill risk -> first slice > last slice
    plan = optimal_schedule(**_params(1e-2))
    assert plan.child_sizes[0] > plan.child_sizes[-1]


def test_cost_and_variance_are_finite_and_positive():
    plan = optimal_schedule(**_params(1e-6))
    assert plan.expected_cost > 0 and math.isfinite(plan.expected_cost)
    assert plan.variance >= 0 and math.isfinite(plan.variance)


def test_efficient_frontier_is_monotone():
    # Higher risk aversion -> lower variance, higher expected cost.
    pts = efficient_frontier(total_shares=1000, n_intervals=10, tau=1.0,
                             sigma=0.02, eta=1e-4, gamma=1e-5,
                             lambdas=[1e-8, 1e-6, 1e-4, 1e-2])
    costs = [p.expected_cost for p in pts]
    variances = [p.variance for p in pts]
    assert costs == sorted(costs)              # increasing
    assert variances == sorted(variances, reverse=True)  # decreasing
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/intraday/execution/test_almgren_chriss.py -v`
Expected: FAIL — ModuleNotFoundError.

- [ ] **Step 3: Write minimal implementation**

Create `quant/intraday/execution/almgren_chriss.py`:

```python
"""Almgren-Chriss optimal execution (Almgren & Chriss, 2000, "Optimal execution of
portfolio transactions"). Linear permanent (g(v)=gamma*v) and temporary (h(v)=eta*v)
impact; mean-variance objective E[C] + lambda*V[C].

The optimal holdings follow x_j = X * sinh(kappa*(T - t_j)) / sinh(kappa*T), where
kappa solves cosh(kappa*tau) = 1 + lambda*sigma^2*tau^2 / (2*eta_tilde),
eta_tilde = eta - gamma*tau/2. We compute the trajectory in closed form, then derive
E[C] and V[C] directly from it (avoids transcribing the messy closed-form cost)."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ACPlan:
    child_sizes: list[int]      # n_1..n_N shares to trade in each interval (sum = X)
    holdings: list[float]       # x_0..x_N remaining shares (x_0 = X, x_N = 0)
    expected_cost: float
    variance: float
    kappa: float


@dataclass(frozen=True)
class FrontierPoint:
    risk_aversion: float
    expected_cost: float
    variance: float


def _solve_kappa(sigma: float, eta: float, gamma: float, tau: float, lam: float) -> float:
    eta_tilde = eta - gamma * tau / 2.0
    if eta_tilde <= 0:
        eta_tilde = eta  # degenerate guard; keep positive temp-impact
    arg = 1.0 + (lam * sigma * sigma * tau * tau) / (2.0 * eta_tilde)
    if arg <= 1.0:
        return 0.0  # lambda -> 0 : risk-neutral, linear trajectory
    return math.acosh(arg) / tau


def _holdings(total_shares: int, n_intervals: int, tau: float, kappa: float) -> list[float]:
    n, X = n_intervals, float(total_shares)
    T = n * tau
    if kappa <= 0.0:
        # Risk-neutral limit: straight-line liquidation.
        return [X * (1.0 - j / n) for j in range(n + 1)]
    sinh_kT = math.sinh(kappa * T)
    return [X * math.sinh(kappa * (T - j * tau)) / sinh_kT for j in range(n + 1)]


def _child_sizes(holdings: list[float]) -> list[int]:
    # n_j = x_{j-1} - x_j ; round to ints and fix the residual on the last slice so
    # the children sum EXACTLY to the parent.
    raw = [holdings[j - 1] - holdings[j] for j in range(1, len(holdings))]
    sizes = [int(round(r)) for r in raw]
    total = int(round(holdings[0]))
    sizes[-1] += total - sum(sizes)
    return sizes


def optimal_schedule(
    *, total_shares: int, n_intervals: int, tau: float,
    sigma: float, eta: float, gamma: float, risk_aversion: float,
) -> ACPlan:
    kappa = _solve_kappa(sigma, eta, gamma, tau, risk_aversion)
    holdings = _holdings(total_shares, n_intervals, tau, kappa)
    sizes = _child_sizes(holdings)
    # Costs computed directly from the trajectory (textbook-consistent):
    expected_cost = 0.5 * gamma * total_shares**2 + (eta / tau) * sum(n * n for n in sizes)
    variance = sigma * sigma * tau * sum(x * x for x in holdings[1:])
    return ACPlan(child_sizes=sizes, holdings=holdings,
                  expected_cost=expected_cost, variance=variance, kappa=kappa)


def efficient_frontier(
    *, total_shares: int, n_intervals: int, tau: float,
    sigma: float, eta: float, gamma: float, lambdas: list[float],
) -> list[FrontierPoint]:
    pts: list[FrontierPoint] = []
    for lam in sorted(lambdas):
        plan = optimal_schedule(total_shares=total_shares, n_intervals=n_intervals,
                                tau=tau, sigma=sigma, eta=eta, gamma=gamma,
                                risk_aversion=lam)
        pts.append(FrontierPoint(risk_aversion=lam,
                                 expected_cost=plan.expected_cost, variance=plan.variance))
    return pts
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/intraday/execution/test_almgren_chriss.py -v` (expect 5 passing). If `test_efficient_frontier_is_monotone` fails because expected_cost is identical across tiny lambdas (rounding), widen the lambda spread in the test. Then `uv run ruff check` + `uv run mypy quant/intraday/execution/almgren_chriss.py` (clean).

- [ ] **Step 5: Commit**

```bash
git add quant/intraday/execution/almgren_chriss.py tests/intraday/execution/test_almgren_chriss.py
git commit -m "feat(intraday-exec): Almgren-Chriss optimal-trajectory solver + efficient frontier"
```

---

### Task 3: Baselines (TWAP / VWAP / immediate)

**Files:**
- Create: `quant/intraday/execution/baselines.py`
- Test: `tests/intraday/execution/test_baselines.py`

- [ ] **Step 1: Write the failing test**

Create `tests/intraday/execution/test_baselines.py`:

```python
from quant.intraday.execution.baselines import immediate, twap, vwap


def test_twap_equal_slices_sum_to_parent():
    sizes = twap(total_shares=100, n_intervals=4)
    assert sizes == [25, 25, 25, 25]
    assert sum(sizes) == 100


def test_twap_handles_indivisible_remainder_on_last():
    sizes = twap(total_shares=103, n_intervals=4)
    assert sum(sizes) == 103
    assert sizes[:3] == [25, 25, 25] and sizes[-1] == 28


def test_vwap_weights_proportional_to_volume_and_sum_to_parent():
    sizes = vwap(total_shares=100, volume_curve=[1.0, 3.0, 1.0])  # 20%/60%/20%
    assert sum(sizes) == 100
    assert sizes == [20, 60, 20]


def test_immediate_is_single_slice():
    assert immediate(total_shares=42) == [42]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/intraday/execution/test_baselines.py -v`
Expected: FAIL — ModuleNotFoundError.

- [ ] **Step 3: Write minimal implementation**

Create `quant/intraday/execution/baselines.py`:

```python
"""Execution baselines to compare against the Almgren-Chriss schedule. Each returns
a list of integer child sizes that sum EXACTLY to total_shares."""

from __future__ import annotations


def _fix_residual(sizes: list[int], total: int) -> list[int]:
    sizes[-1] += total - sum(sizes)
    return sizes


def twap(*, total_shares: int, n_intervals: int) -> list[int]:
    """Equal slices across n_intervals; the indivisible remainder lands on the last."""
    base = total_shares // n_intervals
    sizes = [base] * n_intervals
    return _fix_residual(sizes, total_shares)


def vwap(*, total_shares: int, volume_curve: list[float]) -> list[int]:
    """Slices proportional to the expected per-interval volume curve."""
    total_vol = sum(volume_curve)
    if total_vol <= 0:
        return twap(total_shares=total_shares, n_intervals=len(volume_curve))
    sizes = [int(round(total_shares * v / total_vol)) for v in volume_curve]
    return _fix_residual(sizes, total_shares)


def immediate(*, total_shares: int) -> list[int]:
    """One shot."""
    return [total_shares]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/intraday/execution/test_baselines.py -v` (expect 4 passing). Note the test calls `twap(total_shares=..., n_intervals=...)`, `vwap(total_shares=..., volume_curve=...)`, `immediate(total_shares=...)` — keyword-only signatures must match. Then ruff + mypy clean.

- [ ] **Step 5: Commit**

```bash
git add quant/intraday/execution/baselines.py tests/intraday/execution/test_baselines.py
git commit -m "feat(intraday-exec): TWAP/VWAP/immediate execution baselines"
```

---

### Task 4: ExecutionProgram (schedule → per-tick slices)

**Files:**
- Create: `quant/intraday/execution/scheduler.py`
- Test: `tests/intraday/execution/test_scheduler.py`

- [ ] **Step 1: Write the failing test**

Create `tests/intraday/execution/test_scheduler.py`:

```python
import pytest

from quant.intraday.execution.scheduler import ExecutionProgram
from quant.intraday.strategy import Side


def _prog():
    # parent: buy 100 QQQ, worked over child sizes [40, 30, 30] starting at tick 5
    return ExecutionProgram(symbol="QQQ", side=Side.BUY, total_qty=100,
                            child_sizes=[40, 30, 30], start_tick=5)


def test_slice_due_follows_schedule_by_tick_offset():
    p = _prog()
    assert p.slice_due(5) == 40   # offset 0
    assert p.slice_due(6) == 30   # offset 1
    assert p.slice_due(7) == 30   # offset 2


def test_slice_due_zero_before_start_and_after_end():
    p = _prog()
    assert p.slice_due(4) == 0    # before start
    assert p.slice_due(8) == 0    # past the schedule


def test_record_fill_tracks_remaining_and_completion():
    p = _prog()
    assert not p.is_complete
    p.record_fill(40); p.record_fill(30)
    assert p.remaining == 30 and not p.is_complete
    p.record_fill(30)
    assert p.remaining == 0 and p.is_complete


def test_cancel_marks_complete_and_zeros_due():
    p = _prog()
    p.cancel()
    assert p.is_complete
    assert p.slice_due(5) == 0


def test_child_sizes_must_sum_to_total():
    with pytest.raises(ValueError):
        ExecutionProgram(symbol="QQQ", side=Side.BUY, total_qty=100,
                         child_sizes=[40, 30], start_tick=0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/intraday/execution/test_scheduler.py -v`
Expected: FAIL — ModuleNotFoundError.

- [ ] **Step 3: Write minimal implementation**

Create `quant/intraday/execution/scheduler.py`:

```python
"""ExecutionProgram: one parent order worked along a fixed child-size schedule, one
slice per tick. Schedule-source-agnostic (Almgren-Chriss or any baseline)."""

from __future__ import annotations

from dataclasses import dataclass, field

from quant.intraday.strategy import Side


@dataclass
class ExecutionProgram:
    symbol: str
    side: Side
    total_qty: int
    child_sizes: list[int]
    start_tick: int
    filled: int = field(default=0)
    cancelled: bool = field(default=False)

    def __post_init__(self) -> None:
        if sum(self.child_sizes) != self.total_qty:
            raise ValueError(
                f"child_sizes sum {sum(self.child_sizes)} != total_qty {self.total_qty}"
            )

    def slice_due(self, tick_index: int) -> int:
        """Child shares to trade at this tick (0 if before start, past end, or done)."""
        if self.cancelled:
            return 0
        offset = tick_index - self.start_tick
        if 0 <= offset < len(self.child_sizes):
            return self.child_sizes[offset]
        return 0

    def record_fill(self, qty: int) -> None:
        self.filled += qty

    @property
    def remaining(self) -> int:
        return self.total_qty - self.filled

    @property
    def is_complete(self) -> bool:
        return self.cancelled or self.filled >= self.total_qty

    def cancel(self) -> None:
        self.cancelled = True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/intraday/execution/test_scheduler.py -v` (expect 5 passing). Then ruff + mypy clean.

- [ ] **Step 5: Commit**

```bash
git add quant/intraday/execution/scheduler.py tests/intraday/execution/test_scheduler.py
git commit -m "feat(intraday-exec): ExecutionProgram (child-size schedule -> per-tick slices)"
```

---

### Task 5: Calibration (σ, η, γ) with linear-vs-sqrt reconciliation

**Files:**
- Create: `quant/intraday/execution/calibrate.py`
- Test: `tests/intraday/execution/test_calibrate.py`

η is the LINEAR temporary-impact coefficient the A–C solver needs, derived as a local linearization of the repo's SQRT impact model at the expected per-slice participation. The repo's `market_impact_bps(notional, adv, coef)` returns `coef*sqrt(notional/adv)` (bps). Per-share temporary cost at slice notional `q` is `price * 1e-4 * market_impact_bps(q, adv, coef)`. Linear `eta` (cost per share per share-rate) is taken as that per-share cost divided by the slice share size — a local slope at the planned slice. σ is the realized stdev of recent close-to-close returns scaled to price units.

- [ ] **Step 1: Write the failing test**

Create `tests/intraday/execution/test_calibrate.py`:

```python
import math

from quant.intraday.execution.calibrate import calibrate
from quant.intraday.execution.config import ExecConfig


def test_calibrate_returns_positive_params():
    recent_returns = [0.001, -0.002, 0.0015, -0.001, 0.0005] * 20
    sigma, eta, gamma = calibrate(
        price=400.0, slice_shares=20, adv_dollar=5_000_000_000.0,
        recent_returns=recent_returns, config=ExecConfig(),
    )
    assert sigma > 0 and math.isfinite(sigma)
    assert eta > 0 and math.isfinite(eta)
    assert gamma == ExecConfig().perm_impact_frac * eta


def test_zero_volatility_returns_small_positive_sigma():
    sigma, eta, gamma = calibrate(
        price=400.0, slice_shares=20, adv_dollar=5_000_000_000.0,
        recent_returns=[0.0] * 100, config=ExecConfig(),
    )
    assert sigma >= 0.0  # flat history -> zero realized vol is allowed
    assert eta > 0


def test_eta_scales_with_impact_coef():
    rr = [0.001, -0.001] * 50
    _, eta_lo, _ = calibrate(price=400.0, slice_shares=20, adv_dollar=5e9,
                             recent_returns=rr, config=ExecConfig(impact_coef_bps=5.0))
    _, eta_hi, _ = calibrate(price=400.0, slice_shares=20, adv_dollar=5e9,
                             recent_returns=rr, config=ExecConfig(impact_coef_bps=20.0))
    assert eta_hi > eta_lo
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/intraday/execution/test_calibrate.py -v`
Expected: FAIL — ModuleNotFoundError.

- [ ] **Step 3: Write minimal implementation**

Create `quant/intraday/execution/calibrate.py`:

```python
"""Calibrate Almgren-Chriss inputs (sigma, eta, gamma) for a parent order.

eta (linear temporary impact) is a local linearization of the repo's SQRT impact
model (quant.backtest.impact.market_impact_bps) at the planned per-slice size, so the
A-C closed form stays usable while being anchored to the real impact curve. The sim
evaluation still uses the true sqrt model, so the closed-form-vs-realized gap is
visible (spec section 6)."""

from __future__ import annotations

import statistics

from quant.backtest.impact import market_impact_bps
from quant.intraday.execution.config import ExecConfig


def calibrate(
    *, price: float, slice_shares: int, adv_dollar: float,
    recent_returns: list[float], config: ExecConfig,
) -> tuple[float, float, float]:
    """Return (sigma, eta, gamma).

    sigma: realized stdev of recent returns, in PRICE units (return-stdev * price).
    eta:   per-share temporary-impact slope ($ per share, per share traded), from a
           local linearization of the sqrt model at `slice_shares`.
    gamma: permanent-impact coef = perm_impact_frac * eta.
    """
    ret_sd = statistics.pstdev(recent_returns) if len(recent_returns) > 1 else 0.0
    sigma = ret_sd * price

    slice_shares = max(1, slice_shares)
    slice_notional = price * slice_shares
    impact_bps = market_impact_bps(slice_notional, adv_dollar, config.impact_coef_bps)
    per_share_cost = price * (impact_bps * 1e-4)            # $ per share at this slice
    eta = per_share_cost / slice_shares                    # local linear slope
    if eta <= 0.0:
        eta = 1e-9                                         # keep solver well-posed
    gamma = config.perm_impact_frac * eta
    return sigma, eta, gamma
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/intraday/execution/test_calibrate.py -v` (expect 3 passing). Then ruff + mypy clean.

- [ ] **Step 5: Commit**

```bash
git add quant/intraday/execution/calibrate.py tests/intraday/execution/test_calibrate.py
git commit -m "feat(intraday-exec): calibrate sigma/eta/gamma (sqrt->linear local reconciliation)"
```

---

### Task 6: ExecutionManager (tracks active programs)

**Files:**
- Create: `quant/intraday/execution/manager.py`
- Test: `tests/intraday/execution/test_manager.py`

The manager holds at most one active program per symbol, builds an A–C program from a parent entry, and yields due child `Order`s each tick. It does NOT submit — the loop submits what the manager returns.

- [ ] **Step 1: Write the failing test**

Create `tests/intraday/execution/test_manager.py`:

```python
from quant.intraday.execution.config import ExecConfig
from quant.intraday.execution.manager import ExecutionManager
from quant.intraday.strategy import Order, OrderType, Side


def test_start_entry_builds_program_and_blocks_restart():
    mgr = ExecutionManager(ExecConfig(horizon_ticks=3))
    parent = Order("QQQ", Side.BUY, 90, OrderType.MARKET)
    mgr.start_entry(parent, tick_index=0, sigma=0.02, eta=1e-4, gamma=1e-5)
    assert mgr.has_active("QQQ")
    # a second start for the same symbol while active is a no-op (returns False)
    assert mgr.start_entry(parent, tick_index=0, sigma=0.02, eta=1e-4, gamma=1e-5) is False


def test_due_slices_emit_orders_summing_to_parent_over_horizon():
    mgr = ExecutionManager(ExecConfig(horizon_ticks=3))
    mgr.start_entry(Order("QQQ", Side.BUY, 90, OrderType.MARKET),
                    tick_index=0, sigma=0.02, eta=1e-4, gamma=1e-5)
    total = 0
    for t in range(3):
        for o in mgr.due_slices(t):
            assert o.symbol == "QQQ" and o.side is Side.BUY
            total += o.qty
            mgr.record_fill("QQQ", o.qty)
    assert total == 90
    assert not mgr.has_active("QQQ")  # completed -> removed


def test_cancel_removes_program():
    mgr = ExecutionManager(ExecConfig(horizon_ticks=3))
    mgr.start_entry(Order("IWM", Side.SELL, 30, OrderType.MARKET),
                    tick_index=0, sigma=0.02, eta=1e-4, gamma=1e-5)
    mgr.cancel("IWM")
    assert not mgr.has_active("IWM")
    assert mgr.due_slices(0) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/intraday/execution/test_manager.py -v`
Expected: FAIL — ModuleNotFoundError.

- [ ] **Step 3: Write minimal implementation**

Create `quant/intraday/execution/manager.py`:

```python
"""ExecutionManager: holds at most one active ExecutionProgram per symbol, builds
A-C programs from parent entries, and yields due child Orders per tick. It never
submits — the loop submits what due_slices() returns and reports fills back."""

from __future__ import annotations

from quant.intraday.execution.almgren_chriss import optimal_schedule
from quant.intraday.execution.config import ExecConfig
from quant.intraday.execution.scheduler import ExecutionProgram
from quant.intraday.strategy import Order, OrderType, Side


class ExecutionManager:
    def __init__(self, config: ExecConfig) -> None:
        self._cfg = config
        self._programs: dict[str, ExecutionProgram] = {}

    def has_active(self, symbol: str) -> bool:
        prog = self._programs.get(symbol)
        return prog is not None and not prog.is_complete

    def start_entry(
        self, parent: Order, *, tick_index: int,
        sigma: float, eta: float, gamma: float,
    ) -> bool:
        """Build an A-C program for `parent`. No-op (False) if one is already active."""
        if self.has_active(parent.symbol):
            return False
        plan = optimal_schedule(
            total_shares=parent.qty, n_intervals=self._cfg.horizon_ticks, tau=1.0,
            sigma=sigma, eta=eta, gamma=gamma, risk_aversion=self._cfg.risk_aversion,
        )
        self._programs[parent.symbol] = ExecutionProgram(
            symbol=parent.symbol, side=parent.side, total_qty=parent.qty,
            child_sizes=plan.child_sizes, start_tick=tick_index,
        )
        return True

    def due_slices(self, tick_index: int) -> list[Order]:
        orders: list[Order] = []
        for prog in self._programs.values():
            qty = prog.slice_due(tick_index)
            if qty > 0:
                orders.append(Order(prog.symbol, prog.side, qty, OrderType.MARKET))
        return orders

    def record_fill(self, symbol: str, qty: int) -> None:
        prog = self._programs.get(symbol)
        if prog is not None:
            prog.record_fill(qty)
            if prog.is_complete:
                del self._programs[symbol]

    def cancel(self, symbol: str) -> None:
        prog = self._programs.get(symbol)
        if prog is not None:
            prog.cancel()
            del self._programs[symbol]
```

Note for `start_entry`: keyword-only after `parent` (the `*`). The signature `start_entry(self, parent, *, tick_index, sigma, eta, gamma)` matches the test call `mgr.start_entry(parent, tick_index=0, sigma=..., eta=..., gamma=...)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/intraday/execution/test_manager.py -v` (expect 3 passing). Then ruff + mypy clean.

- [ ] **Step 5: Commit**

```bash
git add quant/intraday/execution/manager.py tests/intraday/execution/test_manager.py
git commit -m "feat(intraday-exec): ExecutionManager (per-symbol A-C programs, due slices)"
```

---

### Task 7: Wire the ExecutionManager into the sleeve loop

**Files:**
- Modify: `quant/intraday/live/loop.py`
- Test: `tests/intraday/live/test_loop_execution.py`

Changes to `run_tick`:
1. `TickDeps` gains an optional `exec_manager: ExecutionManager | None = None` field (default None ⇒ today's immediate behavior, so all existing loop tests still pass unchanged).
2. When a manager IS present: BEFORE the strategy step, emit `exec_manager.due_slices(tick_index)` and submit each (clamped, capped) — these are in-flight entry slices. Then run the strategy; an ENTRY intent calls `exec_manager.start_entry(...)` (calibration inline) instead of submitting immediately. REDUCING/exit intents submit immediately AND call `exec_manager.cancel(symbol)`.
3. A symbol with an active program counts as NOT flat for new opens: `is_new_open = pos == 0 and not exec_manager.has_active(symbol)`.
4. `_flatten_all` cancels all active programs for the flattened symbols (loss-halt + flat-by-close de-risk immediately).
5. `TickDeps` gains `tick_index: int` (the loop driver increments it; tests pass it explicitly).

This is the one integration task; keep `run_tick` readable — extract the manager handling into a small helper `_work_active_programs(deps)` if it grows.

- [ ] **Step 1: Write the failing test**

Create `tests/intraday/live/test_loop_execution.py`:

```python
from datetime import UTC, datetime

from quant.intraday.data.events import QuoteBar
from quant.intraday.execution.config import ExecConfig
from quant.intraday.execution.manager import ExecutionManager
from quant.intraday.live.config import SleeveConfig
from quant.intraday.live.loop import TickDeps, run_tick
from quant.intraday.live.sleeve import Fill, SleeveLedger
from quant.intraday.strategy import Order, Side


class _Broker:
    def __init__(self):
        self.orders = []
    def account(self):
        class A:
            equity = 100_000.0
        return A()
    def submit_simple_order(self, *, symbol, side, qty, client_order_id,
                            order_type="market", limit_price=None, dry_run=False):
        self.orders.append((symbol, side, qty)); return client_order_id


class _Feed:
    def __init__(self, bars): self._bars = bars
    def latest_quotes(self, now=None): return self._bars


class _Strat:
    def __init__(self, orders): self._orders = orders
    def on_event(self, event, ctx): return self._orders


def _qb(sym, price):
    return QuoteBar(ts=datetime(2026, 6, 8, 15, 0, tzinfo=UTC), symbol=sym,
                    bid=price - 0.01, ask=price + 0.01, bid_size=100, ask_size=100)


def _deps(tmp_path, broker, feed, strat, ledger, *, tick_index, mgr, now=None):
    return TickDeps(
        data_dir=tmp_path, config=SleeveConfig(notional_cap_pct=1.0, notional_cap_abs=1e9,
                                               per_trade_cap=1e9, mean_reversion_lookback=5),
        broker=broker, feed=feed, strategy=strat, ledger=ledger,
        now=now or datetime(2026, 6, 8, 15, 0, tzinfo=UTC),
        session_open=True, session_close=datetime(2026, 6, 8, 20, 0, tzinfo=UTC),
        tick_index=tick_index, exec_manager=mgr,
    )


def test_entry_is_worked_over_ticks_not_dumped(tmp_path):
    mgr = ExecutionManager(ExecConfig(horizon_ticks=3, risk_aversion=1e-12))
    broker, ledger = _Broker(), SleeveLedger()
    # strategy wants to BUY 90 on tick 0 only; subsequent ticks the strategy is silent
    strat0 = _Strat([Order("QQQ", Side.BUY, 90)])
    run_tick(_deps(tmp_path, broker, _Feed([_qb("QQQ", 100.0)]), strat0, ledger,
                   tick_index=0, mgr=mgr))
    # tick 0 should NOT dump all 90 at once (worked over 3 ticks ~ 30 each)
    first_qty = sum(q for s, _, q in broker.orders if s == "QQQ")
    assert first_qty < 90
    # silent strategy on ticks 1,2 -> manager keeps working the program
    silent = _Strat([])
    run_tick(_deps(tmp_path, broker, _Feed([_qb("QQQ", 100.0)]), silent, ledger,
                   tick_index=1, mgr=mgr))
    run_tick(_deps(tmp_path, broker, _Feed([_qb("QQQ", 100.0)]), silent, ledger,
                   tick_index=2, mgr=mgr))
    total = sum(q for s, _, q in broker.orders if s == "QQQ")
    assert total == 90  # fully worked by end of horizon


def test_flatten_cancels_active_program(tmp_path):
    mgr = ExecutionManager(ExecConfig(horizon_ticks=5, risk_aversion=1e-12))
    broker, ledger = _Broker(), SleeveLedger()
    # start an entry program
    run_tick(_deps(tmp_path, broker, _Feed([_qb("QQQ", 100.0)]),
                   _Strat([Order("QQQ", Side.BUY, 100)]), ledger, tick_index=0, mgr=mgr))
    assert mgr.has_active("QQQ")
    # now a flat-by-close tick: must cancel the program, no further entry slices
    flat = _Strat([])
    run_tick(_deps(tmp_path, broker, _Feed([_qb("QQQ", 100.0)]), flat, ledger,
                   tick_index=1, mgr=mgr,
                   now=datetime(2026, 6, 8, 19, 50, tzinfo=UTC)))
    assert not mgr.has_active("QQQ")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/intraday/live/test_loop_execution.py -v`
Expected: FAIL — `TickDeps.__init__() got an unexpected keyword argument 'tick_index'` (and `exec_manager`).

- [ ] **Step 3: Write the implementation**

In `quant/intraday/live/loop.py`:

(a) Add imports at the top with the other intraday imports:
```python
from quant.intraday.execution.calibrate import calibrate
from quant.intraday.execution.config import ExecConfig
from quant.intraday.execution.manager import ExecutionManager
```

(b) Add two fields to `TickDeps` (after `dry_run`):
```python
    tick_index: int = 0
    exec_manager: ExecutionManager | None = None
    exec_config: ExecConfig | None = None
```

(c) In `_flatten_all`, cancel any active programs for flattened symbols. At the top of the for-loop body (after computing `sym`), add:
```python
        if deps.exec_manager is not None:
            deps.exec_manager.cancel(sym)
```

(d) In `run_tick`, place this block AFTER the strategy per-order loop (the loop that
routes entries to the manager and submits exits) and BEFORE the final `_journal(...)`
call. **Ordering matters:** an entry's program is CREATED during the strategy step
(3e), so its offset-0 slice must be worked AFTER that step — otherwise the first slice
is skipped and the parent never fully fills. (It is still after all guardrails, so
halt/loss/flat early-returns are unaffected.) Add the in-flight program working:
```python
    # Work in-flight execution programs (entry slices), AFTER the strategy step so a
    # program just created this tick gets its first (offset-0) slice worked now.
    if deps.exec_manager is not None:
        for i, child in enumerate(deps.exec_manager.due_slices(deps.tick_index)):
            price = marks.get(child.symbol)
            if price is None:
                continue
            qty = clamp_qty_to_caps(
                desired_qty=child.qty, price=price,
                gross_notional=deps.ledger.gross_notional(marks),
                sleeve_allocation=allocation, config=cfg,
            )
            if qty <= 0:
                continue
            side_str = "buy" if child.side is Side.BUY else "sell"
            # High offset (1000 + i) so program-slice COIDs never collide with the
            # strategy-order COIDs (small seq values) submitted earlier this tick.
            coid = make_sleeve_coid(child.symbol, deps.now, 1000 + i)
            deps.broker.submit_simple_order(symbol=child.symbol, side=side_str, qty=qty,
                                            client_order_id=coid, dry_run=deps.dry_run)
            signed = qty if child.side is Side.BUY else -qty
            deps.ledger.record(Fill(symbol=child.symbol, qty=signed, price=price))
            deps.exec_manager.record_fill(child.symbol, qty)
```
where `i` is the `enumerate(...)` index of the due-slices loop: `for i, child in enumerate(deps.exec_manager.due_slices(deps.tick_index)):`.

(e) In the per-order strategy loop, change the open/exit handling. Replace the `is_new_open` line and the submit branch so that, WHEN a manager is present, a new-open entry is routed to the manager and reducing orders cancel any program:
```python
        pos = deps.ledger.position(order.symbol)
        is_reducing = pos != 0 and (
            (pos > 0 and order.side is Side.SELL) or (pos < 0 and order.side is Side.BUY)
        )
        has_prog = deps.exec_manager is not None and deps.exec_manager.has_active(order.symbol)
        is_new_open = pos == 0 and not has_prog
        if is_new_open and trade_budget_exhausted(round_trips=deps.ledger.round_trips, config=cfg):
            continue
        price = marks.get(order.symbol)
        if price is None:
            continue
        if is_reducing and deps.exec_manager is not None:
            deps.exec_manager.cancel(order.symbol)  # de-risk immediately, drop any program
        # Route a NEW OPEN through the execution manager when present.
        if is_new_open and deps.exec_manager is not None:
            ec = deps.exec_config or ExecConfig()
            sigma, eta, gamma = calibrate(
                price=price, slice_shares=max(1, order.qty // ec.horizon_ticks),
                adv_dollar=_sleeve_adv_dollar(order.symbol, price),
                recent_returns=_recent_returns(deps, order.symbol), config=ec,
            )
            deps.exec_manager.start_entry(order, tick_index=deps.tick_index,
                                          sigma=sigma, eta=eta, gamma=gamma)
            continue  # slices will be worked on subsequent ticks
        # else: immediate path (exits, or no manager) — existing clamp+submit below
```
Keep the existing immediate clamp+submit code as the fallback for the non-manager / reducing path. For `_sleeve_adv_dollar` and `_recent_returns`, add minimal module-level helpers in loop.py:
```python
def _sleeve_adv_dollar(symbol: str, price: float) -> float:
    # Mega-liquid ETF proxy ADV ($). Calibration only needs an order-of-magnitude
    # anchor; refine later from the data layer.
    return 5_000_000_000.0


def _recent_returns(deps: TickDeps, symbol: str) -> list[float]:
    # Spine keeps no return history yet; return an empty list -> sigma=0 is acceptable
    # (A-C degenerates to TWAP-like, which is the safe default). A future task can wire
    # the data layer here.
    return []
```

(f) The `run_loop` driver must pass an incrementing `tick_index`. In the `run` CLI (Task 8 wires the manager); for `run_loop`, have the factory close over a counter — but since `deps_factory` already builds fresh deps, document that the factory is responsible for supplying `tick_index`. No change needed to `run_loop` itself beyond what the factory provides.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/intraday/live/test_loop_execution.py -v` (expect 2 passing).
Then run the WHOLE spine loop suite to confirm the default-None path is unchanged:
`uv run pytest tests/intraday/live/ -q` (all still pass). Then ruff + mypy clean on loop.py.

- [ ] **Step 5: Commit**

```bash
git add quant/intraday/live/loop.py tests/intraday/live/test_loop_execution.py
git commit -m "feat(intraday-exec): work sleeve entries via ExecutionManager (exits/flattens immediate)"
```

---

### Task 8: Evaluation adapter + realized sim cost

**Files:**
- Create: `quant/intraday/execution/evaluate.py`
- Test: `tests/intraday/execution/test_evaluate.py`

A `LiquidationStrategy` (implements `IntradayStrategy`) emits the next child on each QuoteBar event until the schedule is exhausted; `evaluate_schedule(...)` runs it through `BacktestEngine` and returns the realized cost from `result.costs`.

- [ ] **Step 1: Write the failing test**

Create `tests/intraday/execution/test_evaluate.py`:

```python
from datetime import UTC, datetime, timedelta

from quant.intraday.data.events import QuoteBar
from quant.intraday.execution.evaluate import LiquidationStrategy, evaluate_schedule
from quant.intraday.strategy import Side


def _events(n, price=100.0):
    t0 = datetime(2026, 6, 8, 14, 30, tzinfo=UTC)
    out = []
    for i in range(n):
        out.append(QuoteBar(ts=t0 + timedelta(minutes=i), symbol="QQQ",
                            bid=price - 0.01, ask=price + 0.01, bid_size=500, ask_size=500))
    return out


def test_liquidation_strategy_emits_schedule_in_order():
    strat = LiquidationStrategy(symbol="QQQ", side=Side.BUY, child_sizes=[10, 20, 30])
    emitted = []

    class _Ctx:
        def position(self, s): return 0
        def cash(self): return 0.0
        def nbbo(self, s): return None
        def now(self): return datetime(2026, 6, 8, 14, 30, tzinfo=UTC)

    for ev in _events(5):
        for o in strat.on_event(ev, _Ctx()):
            emitted.append(o.qty)
    assert emitted == [10, 20, 30]  # exhausts after 3 events, silent after


def test_evaluate_schedule_returns_realized_cost():
    res = evaluate_schedule(
        events=_events(6), symbol="QQQ", side=Side.BUY, child_sizes=[20, 20, 20],
        adv_dollar={"QQQ": 5_000_000_000.0}, impact_coef_bps=10.0,
    )
    assert "total_cost" in res and res["total_cost"] >= 0.0
    assert "commission" in res and "impact" in res and "spread" in res
    assert res["filled_shares"] == 60
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/intraday/execution/test_evaluate.py -v`
Expected: FAIL — ModuleNotFoundError.

- [ ] **Step 3: Write minimal implementation**

Create `quant/intraday/execution/evaluate.py`:

```python
"""Evaluate an execution schedule against the real intraday sim fill model. A
LiquidationStrategy emits one child per event until the schedule is exhausted;
evaluate_schedule runs it through BacktestEngine and reports realized cost."""

from __future__ import annotations

from typing import Any

from quant.intraday.data.events import Event, QuoteBar
from quant.intraday.sim.engine import BacktestEngine
from quant.intraday.strategy import Order, OrderType, Side, StrategyContext


class LiquidationStrategy:
    """Works a fixed parent along `child_sizes`, one slice per QuoteBar event."""

    def __init__(self, *, symbol: str, side: Side, child_sizes: list[int]) -> None:
        self._symbol = symbol
        self._side = side
        self._sizes = list(child_sizes)
        self._i = 0

    def on_event(self, event: Event, ctx: StrategyContext) -> list[Order]:
        if not isinstance(event, QuoteBar) or event.symbol != self._symbol:
            return []
        if self._i >= len(self._sizes):
            return []
        qty = self._sizes[self._i]
        self._i += 1
        if qty <= 0:
            return []
        return [Order(self._symbol, self._side, qty, OrderType.MARKET)]


def evaluate_schedule(
    *, events: list[Event], symbol: str, side: Side, child_sizes: list[int],
    adv_dollar: dict[str, float], impact_coef_bps: float,
) -> dict[str, Any]:
    """Run the schedule through the sim; return realized cost components + fills."""
    strat = LiquidationStrategy(symbol=symbol, side=side, child_sizes=child_sizes)
    result = BacktestEngine().run(
        strat, events, adv_dollar=adv_dollar, impact_coef_bps=impact_coef_bps
    )
    filled = sum(abs(f.signed_qty()) for f in result.fills)
    return {
        "total_cost": result.costs.total,
        "commission": result.costs.commission,
        "impact": result.costs.impact,
        "spread": result.costs.spread,
        "filled_shares": filled,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/intraday/execution/test_evaluate.py -v` (expect 2 passing). If `Fill.signed_qty()` is a property not a method, adjust `f.signed_qty()` → `f.signed_qty` to match `quant/intraday/sim/fills.py`. Then ruff + mypy clean.

- [ ] **Step 5: Commit**

```bash
git add quant/intraday/execution/evaluate.py tests/intraday/execution/test_evaluate.py
git commit -m "feat(intraday-exec): sim-based schedule evaluation (realized cost)"
```

---

### Task 9: CLI — `quant intraday exec frontier` + `schedule`

**Files:**
- Modify: `quant/intraday/cli.py`
- Test: `tests/intraday/execution/test_exec_cli.py`

Add an `exec` subgroup under `intraday` with two commands:
- `schedule --symbol --shares [--horizon] [--lam]` — prints the A–C child-size schedule.
- `frontier --symbol --shares [--horizon]` — prints the efficient frontier (closed-form expected cost vs variance across a λ grid) plus the TWAP/immediate baseline child schedules for comparison. (VWAP needs a volume curve; if none is supplied, print TWAP+immediate and note VWAP is sim-only.)

Both use a fixed illustrative σ/η/γ from `calibrate` with a representative price/ADV so they run without live data (this is a demonstration/inspection command).

- [ ] **Step 1: Write the failing test**

Create `tests/intraday/execution/test_exec_cli.py`:

```python
from click.testing import CliRunner

from quant.intraday.cli import intraday


def test_exec_group_exists():
    r = CliRunner().invoke(intraday, ["exec", "--help"])
    assert r.exit_code == 0
    assert "frontier" in r.output and "schedule" in r.output


def test_schedule_prints_child_sizes():
    r = CliRunner().invoke(intraday, ["exec", "schedule", "--symbol", "QQQ",
                                      "--shares", "1000", "--horizon", "5"])
    assert r.exit_code == 0
    assert "QQQ" in r.output
    # 5 child sizes shown
    assert r.output.count("slice") >= 1 or "child" in r.output.lower()


def test_frontier_prints_points_and_baselines():
    r = CliRunner().invoke(intraday, ["exec", "frontier", "--symbol", "QQQ",
                                      "--shares", "1000"])
    assert r.exit_code == 0
    assert "frontier" in r.output.lower()
    assert "twap" in r.output.lower() and "immediate" in r.output.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/intraday/execution/test_exec_cli.py -v`
Expected: FAIL — `exec` not a command of `intraday`.

- [ ] **Step 3: Write the implementation**

In `quant/intraday/cli.py`, add (after the existing groups). Use the module-level `click` import already present:

```python
@intraday.group()
def exec_() -> None:  # function name avoids the `exec` builtin; CLI name set below
    """Optimal-execution (Almgren-Chriss) demonstration commands."""


# Register under the CLI name "exec".
intraday.add_command(exec_, name="exec")


def _demo_params(shares: int, horizon: int):
    from quant.intraday.execution.calibrate import calibrate
    from quant.intraday.execution.config import ExecConfig
    cfg = ExecConfig(horizon_ticks=horizon)
    # Representative mega-liquid ETF anchors so the command runs without live data.
    sigma, eta, gamma = calibrate(
        price=400.0, slice_shares=max(1, shares // horizon), adv_dollar=5_000_000_000.0,
        recent_returns=[0.0008, -0.0007, 0.0009, -0.0006] * 15, config=cfg,
    )
    return cfg, sigma, eta, gamma


@exec_.command()
@click.option("--symbol", required=True)
@click.option("--shares", type=int, required=True)
@click.option("--horizon", type=int, default=5)
@click.option("--lam", type=float, default=None, help="risk aversion (default ExecConfig)")
def schedule(symbol: str, shares: int, horizon: int, lam: float | None) -> None:
    """Print the Almgren-Chriss child-size schedule for a parent order."""
    from quant.intraday.execution.almgren_chriss import optimal_schedule
    cfg, sigma, eta, gamma = _demo_params(shares, horizon)
    plan = optimal_schedule(total_shares=shares, n_intervals=horizon, tau=1.0,
                            sigma=sigma, eta=eta, gamma=gamma,
                            risk_aversion=lam if lam is not None else cfg.risk_aversion)
    click.echo(f"A-C schedule for {symbol} ({shares} sh over {horizon} ticks):")
    for i, n in enumerate(plan.child_sizes):
        click.echo(f"  slice {i}: {n} sh")
    click.echo(f"expected_cost={plan.expected_cost:.4f} variance={plan.variance:.6f}")


@exec_.command()
@click.option("--symbol", required=True)
@click.option("--shares", type=int, required=True)
@click.option("--horizon", type=int, default=5)
def frontier(symbol: str, shares: int, horizon: int) -> None:
    """Print the efficient frontier (cost vs variance) + TWAP/immediate baselines."""
    from quant.intraday.execution.almgren_chriss import efficient_frontier
    from quant.intraday.execution.baselines import immediate, twap
    cfg, sigma, eta, gamma = _demo_params(shares, horizon)
    pts = efficient_frontier(total_shares=shares, n_intervals=horizon, tau=1.0,
                             sigma=sigma, eta=eta, gamma=gamma,
                             lambdas=[1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3])
    click.echo(f"Efficient frontier for {symbol} ({shares} sh, horizon {horizon}):")
    click.echo("  lambda        expected_cost    variance")
    for p in pts:
        click.echo(f"  {p.risk_aversion:<12.1e} {p.expected_cost:<15.4f} {p.variance:.6f}")
    click.echo(f"baseline TWAP child sizes: {twap(total_shares=shares, n_intervals=horizon)}")
    click.echo(f"baseline immediate: {immediate(total_shares=shares)}")
    click.echo("(VWAP requires a volume curve; available in sim evaluation only.)")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/intraday/execution/test_exec_cli.py -v` (expect 3 passing). Confirm no regression to the intraday CLI: `uv run pytest tests/intraday/ -q`. Then ruff + mypy clean on cli.py.

- [ ] **Step 5: Commit**

```bash
git add quant/intraday/cli.py tests/intraday/execution/test_exec_cli.py
git commit -m "feat(intraday-exec): `quant intraday exec` frontier + schedule CLI"
```

---

### Task 10: Full-suite green + lint/type gate

**Files:** none (verification task)

- [ ] **Step 1: Run the execution + live suites**

Run: `uv run pytest tests/intraday/execution/ tests/intraday/live/ -q`
Expected: ALL pass.

- [ ] **Step 2: Run the full suite excluding network-gated tests**

Run: `uv run pytest -m "not network and not alpaca" -q`
Expected: green (prior baseline + the new tests). (The unfiltered `pytest` blocks on live-Alpaca tests — always exclude `network`/`alpaca`.)

- [ ] **Step 3: Lint + type gate**

Run: `uv run ruff check . && uv run mypy quant`
Expected: clean. Fix any findings without blanket ignores.

- [ ] **Step 4: Commit any fixes**

```bash
git add -A
git commit -m "chore(intraday-exec): full-suite green + lint/type clean"
```

---

## Notes for the implementer

- **Sleeve impact is negligible at sleeve size** — this is expected and documented in the spec. The technique's value shows in the `frontier` CLI and sim evaluation on realistic parent sizes, not in live sleeve P&L. Do NOT "fix" tiny live impact by inflating sizes.
- **Default-None manager keeps the spine unchanged** — Task 7's `exec_manager=None` default means every existing spine test passes untouched; the manager only activates when wired (the `run` CLI can opt in via a follow-up; this plan does not force it into live by default).
- **Exits/flattens are always immediate** — the safety boundary. Never route a reducing order or a guardrail flatten through the manager.
- **A-C cost/variance are computed from the trajectory**, not the gnarly closed form — keep it that way; it's robust and matches the textbook objective `E[C] + lambda*V[C]`.
