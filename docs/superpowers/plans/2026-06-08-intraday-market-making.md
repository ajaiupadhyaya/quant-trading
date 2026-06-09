# Intraday Market Making (Avellaneda–Stoikov) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone, sim/research-only Avellaneda–Stoikov market-making showcase: the inventory-skewed quoting model, a Poisson fill-intensity simulator, a seeded price path, and a `quant intraday mm` CLI that prints the spread-capture vs inventory-risk tradeoff.

**Architecture:** A new self-contained `quant/intraday/marketmaking/` subpackage. Pure quoting math (`avellaneda_stoikov.py`) + intensity model (`intensity.py`) + seeded price path (`price_path.py`) feed a deterministic simulator (`simulator.py`); `evaluate.py` sweeps risk-aversion γ; the CLI surfaces it. It imports nothing from the live loop, the execution engine, or the order-execution sim — and touches no live path.

**Tech Stack:** Python 3.11+, `uv` for all commands, dataclasses, `math`, `random.Random` (seeded), Click CLI, pytest.

**Conventions:** Run everything with `uv run`. Keep `uv run ruff check .` and `uv run mypy quant` clean. TDD: failing test → fail → implement → pass → commit. The working tree has uncommitted data artifacts — ONLY `git add` the exact paths each task lists.

**Price process note:** Avellaneda–Stoikov uses **absolute** volatility σ (price units), so the mid follows **arithmetic** Brownian motion `dS = σ·dW` (`s_{t+1} = s_t + σ·√dt·z`, `z~N(0,1)`), NOT geometric. The spec's "GBM" label was loose; ABM is the faithful process and what this plan implements.

**Spec:** `docs/superpowers/specs/2026-06-08-intraday-market-making-design.md`

**File map (`quant/intraday/marketmaking/`):** `config.py` (MMConfig), `avellaneda_stoikov.py` (quoting math), `intensity.py` (fill model), `price_path.py` (ABM), `simulator.py` (MMResult + run_market_making), `evaluate.py` (gamma_sweep). Plus modify `quant/intraday/cli.py` (add the `mm` group).

---

### Task 1: MMConfig

**Files:**
- Create: `quant/intraday/marketmaking/__init__.py` (empty)
- Create: `tests/intraday/marketmaking/__init__.py` (empty)
- Create: `quant/intraday/marketmaking/config.py`
- Test: `tests/intraday/marketmaking/test_config.py`

- [ ] **Step 1: Create the two empty `__init__.py` package markers.**

- [ ] **Step 2: Write the failing test**

Create `tests/intraday/marketmaking/test_config.py`:

```python
import pytest

from quant.intraday.marketmaking.config import MMConfig


def test_defaults():
    c = MMConfig()
    assert c.gamma > 0
    assert c.k > 0
    assert c.fill_rate_a > 0
    assert c.horizon_seconds > 0
    assert c.dt_seconds > 0
    assert c.sigma > 0
    assert c.lot_size >= 1
    assert isinstance(c.seed, int)


def test_n_steps_is_horizon_over_dt():
    c = MMConfig(horizon_seconds=100.0, dt_seconds=2.0)
    assert c.n_steps == 50


def test_rejects_bad_values():
    with pytest.raises(ValueError):
        MMConfig(gamma=0.0)
    with pytest.raises(ValueError):
        MMConfig(k=0.0)
    with pytest.raises(ValueError):
        MMConfig(dt_seconds=0.0)
    with pytest.raises(ValueError):
        MMConfig(lot_size=0)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/intraday/marketmaking/test_config.py -v`
Expected: FAIL — ModuleNotFoundError.

- [ ] **Step 4: Write minimal implementation**

Create `quant/intraday/marketmaking/config.py`:

```python
"""Configuration for the Avellaneda-Stoikov market-making simulator. No magic
numbers per the Charter; all knobs live here. This is a STYLIZED model: A and k are
assumed intensity parameters, not fit to live fills."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MMConfig:
    gamma: float = 0.1          # inventory risk aversion
    k: float = 1.5              # order-book depth / intensity decay
    fill_rate_a: float = 140.0  # base fill intensity at the touch (per unit time)
    horizon_seconds: float = 600.0  # T (one 10-min episode)
    dt_seconds: float = 1.0     # simulation step
    sigma: float = 0.02         # ABSOLUTE vol, price units per sqrt(second)
    lot_size: int = 1           # shares per fill
    seed: int = 7

    def __post_init__(self) -> None:
        for name in ("gamma", "k", "fill_rate_a", "horizon_seconds", "dt_seconds", "sigma"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if self.lot_size < 1:
            raise ValueError("lot_size must be >= 1")

    @property
    def n_steps(self) -> int:
        return int(self.horizon_seconds / self.dt_seconds)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/intraday/marketmaking/test_config.py -v` (expect 3 passing). Then `uv run ruff check quant/intraday/marketmaking/ tests/intraday/marketmaking/` and `uv run mypy quant/intraday/marketmaking/config.py` (clean).

- [ ] **Step 6: Commit**

```bash
git add quant/intraday/marketmaking/__init__.py quant/intraday/marketmaking/config.py tests/intraday/marketmaking/
git commit -m "feat(intraday-mm): MMConfig for Avellaneda-Stoikov simulator"
```

---

### Task 2: Avellaneda–Stoikov quoting math

**Files:**
- Create: `quant/intraday/marketmaking/avellaneda_stoikov.py`
- Test: `tests/intraday/marketmaking/test_avellaneda_stoikov.py`

- [ ] **Step 1: Write the failing test**

Create `tests/intraday/marketmaking/test_avellaneda_stoikov.py`:

```python
import math

from quant.intraday.marketmaking.avellaneda_stoikov import (
    optimal_spread,
    quotes,
    reservation_price,
)


def test_reservation_price_skews_with_inventory():
    mid, gamma, sigma, tau = 100.0, 0.1, 0.02, 300.0
    assert reservation_price(mid, 0, gamma, sigma, tau) == mid          # flat -> at mid
    assert reservation_price(mid, 5, gamma, sigma, tau) < mid           # long -> below mid
    assert reservation_price(mid, -5, gamma, sigma, tau) > mid          # short -> above mid


def test_optimal_spread_increases_with_gamma_sigma_tau():
    base = optimal_spread(gamma=0.1, sigma=0.02, t_remaining=300.0, k=1.5)
    assert optimal_spread(gamma=0.2, sigma=0.02, t_remaining=300.0, k=1.5) > base  # gamma up
    assert optimal_spread(gamma=0.1, sigma=0.04, t_remaining=300.0, k=1.5) > base  # sigma up
    assert optimal_spread(gamma=0.1, sigma=0.02, t_remaining=600.0, k=1.5) > base  # tau up
    assert base > 0


def test_quotes_symmetric_about_reservation_price():
    mid, q, gamma, sigma, tau, k = 100.0, 3, 0.1, 0.02, 300.0, 1.5
    bid, ask = quotes(mid, q, gamma, sigma, tau, k)
    r = reservation_price(mid, q, gamma, sigma, tau)
    spread = optimal_spread(gamma=gamma, sigma=sigma, t_remaining=tau, k=k)
    assert math.isclose((bid + ask) / 2.0, r, rel_tol=1e-9)
    assert math.isclose(ask - bid, spread, rel_tol=1e-9)
    assert bid < ask


def test_long_inventory_pushes_both_quotes_down_vs_flat():
    flat_bid, flat_ask = quotes(100.0, 0, 0.1, 0.02, 300.0, 1.5)
    long_bid, long_ask = quotes(100.0, 5, 0.1, 0.02, 300.0, 1.5)
    assert long_bid < flat_bid and long_ask < flat_ask  # skewed to sell down inventory
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/intraday/marketmaking/test_avellaneda_stoikov.py -v`
Expected: FAIL — ModuleNotFoundError.

- [ ] **Step 3: Write minimal implementation**

Create `quant/intraday/marketmaking/avellaneda_stoikov.py`:

```python
"""Avellaneda-Stoikov (2008) optimal market-making quotes. Absolute volatility
(price units); the mid follows arithmetic Brownian motion. Pure functions."""

from __future__ import annotations

import math


def reservation_price(
    mid: float, inventory: int, gamma: float, sigma: float, t_remaining: float
) -> float:
    """r = s - q*gamma*sigma^2*(T-t). Long inventory skews the quote center down."""
    return mid - inventory * gamma * sigma * sigma * t_remaining


def optimal_spread(*, gamma: float, sigma: float, t_remaining: float, k: float) -> float:
    """Total bid-ask spread: gamma*sigma^2*(T-t) + (2/gamma)*ln(1 + gamma/k)."""
    return gamma * sigma * sigma * t_remaining + (2.0 / gamma) * math.log(1.0 + gamma / k)


def quotes(
    mid: float, inventory: int, gamma: float, sigma: float, t_remaining: float, k: float
) -> tuple[float, float]:
    """Return (bid, ask) centered on the reservation price, spread wide."""
    r = reservation_price(mid, inventory, gamma, sigma, t_remaining)
    half = optimal_spread(gamma=gamma, sigma=sigma, t_remaining=t_remaining, k=k) / 2.0
    return r - half, r + half
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/intraday/marketmaking/test_avellaneda_stoikov.py -v` (expect 4 passing). Then ruff + mypy clean on the new file.

- [ ] **Step 5: Commit**

```bash
git add quant/intraday/marketmaking/avellaneda_stoikov.py tests/intraday/marketmaking/test_avellaneda_stoikov.py
git commit -m "feat(intraday-mm): Avellaneda-Stoikov reservation price + optimal spread + quotes"
```

---

### Task 3: Fill-intensity model

**Files:**
- Create: `quant/intraday/marketmaking/intensity.py`
- Test: `tests/intraday/marketmaking/test_intensity.py`

- [ ] **Step 1: Write the failing test**

Create `tests/intraday/marketmaking/test_intensity.py`:

```python
import random

from quant.intraday.marketmaking.intensity import (
    draws_fill,
    fill_intensity,
    fill_probability,
)


def test_intensity_decays_with_distance():
    near = fill_intensity(delta=0.1, A=140.0, k=1.5)
    far = fill_intensity(delta=1.0, A=140.0, k=1.5)
    assert near > far > 0.0


def test_probability_in_unit_interval_and_monotone():
    p_near = fill_probability(delta=0.05, A=140.0, k=1.5, dt=1.0)
    p_far = fill_probability(delta=2.0, A=140.0, k=1.5, dt=1.0)
    assert 0.0 <= p_far <= p_near <= 1.0


def test_probability_far_quote_approaches_zero():
    assert fill_probability(delta=100.0, A=140.0, k=1.5, dt=1.0) < 1e-6


def test_negative_distance_clamps_to_one():
    # quote inside the mid (delta<0) -> intensity explodes -> prob clamps to <=1
    p = fill_probability(delta=-1.0, A=140.0, k=1.5, dt=1.0)
    assert 0.0 <= p <= 1.0


def test_draws_fill_is_seed_deterministic():
    r1, r2 = random.Random(1), random.Random(1)
    seq1 = [draws_fill(0.5, r1) for _ in range(20)]
    seq2 = [draws_fill(0.5, r2) for _ in range(20)]
    assert seq1 == seq2
    assert any(seq1) and not all(seq1)  # 0.5 prob -> mix of True/False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/intraday/marketmaking/test_intensity.py -v`
Expected: FAIL — ModuleNotFoundError.

- [ ] **Step 3: Write minimal implementation**

Create `quant/intraday/marketmaking/intensity.py`:

```python
"""Poisson fill-intensity model: a quote at distance `delta` from mid fills with
intensity lambda(delta) = A*exp(-k*delta). Over a step dt the fill probability is
1 - exp(-lambda*dt), clamped to [0, 1]."""

from __future__ import annotations

import math
import random


def fill_intensity(*, delta: float, A: float, k: float) -> float:
    """lambda(delta) = A * exp(-k * delta). Larger distance -> lower intensity."""
    return A * math.exp(-k * delta)


def fill_probability(*, delta: float, A: float, k: float, dt: float) -> float:
    """P(>=1 fill in dt) = 1 - exp(-lambda*dt), clamped to [0, 1]."""
    lam = fill_intensity(delta=delta, A=A, k=k)
    p = 1.0 - math.exp(-lam * dt)
    return min(1.0, max(0.0, p))


def draws_fill(prob: float, rng: random.Random) -> bool:
    """Bernoulli draw against `prob` using the supplied seeded RNG."""
    return rng.random() < prob
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/intraday/marketmaking/test_intensity.py -v` (expect 5 passing). Note the test calls use keyword args (`fill_intensity(delta=..., A=..., k=...)`, `fill_probability(delta=..., A=..., k=..., dt=...)`); keep the keyword-only signatures. Then ruff + mypy clean.

- [ ] **Step 5: Commit**

```bash
git add quant/intraday/marketmaking/intensity.py tests/intraday/marketmaking/test_intensity.py
git commit -m "feat(intraday-mm): Poisson fill-intensity model"
```

---

### Task 4: Arithmetic Brownian price path

**Files:**
- Create: `quant/intraday/marketmaking/price_path.py`
- Test: `tests/intraday/marketmaking/test_price_path.py`

- [ ] **Step 1: Write the failing test**

Create `tests/intraday/marketmaking/test_price_path.py`:

```python
import random

from quant.intraday.marketmaking.price_path import abm_path


def test_length_and_start():
    path = abm_path(s0=100.0, sigma=0.02, dt=1.0, n_steps=50, rng=random.Random(1))
    assert len(path) == 51          # s_0 .. s_n
    assert path[0] == 100.0


def test_seed_determinism():
    a = abm_path(s0=100.0, sigma=0.02, dt=1.0, n_steps=50, rng=random.Random(3))
    b = abm_path(s0=100.0, sigma=0.02, dt=1.0, n_steps=50, rng=random.Random(3))
    assert a == b


def test_zero_sigma_is_flat():
    path = abm_path(s0=100.0, sigma=0.0, dt=1.0, n_steps=10, rng=random.Random(1))
    assert all(p == 100.0 for p in path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/intraday/marketmaking/test_price_path.py -v`
Expected: FAIL — ModuleNotFoundError.

- [ ] **Step 3: Write minimal implementation**

Create `quant/intraday/marketmaking/price_path.py`:

```python
"""Arithmetic Brownian motion mid-price path (Avellaneda-Stoikov uses absolute vol):
s_{t+1} = s_t + sigma*sqrt(dt)*z, z~N(0,1). Seeded for reproducibility."""

from __future__ import annotations

import math
import random


def abm_path(*, s0: float, sigma: float, dt: float, n_steps: int, rng: random.Random) -> list[float]:
    """Return [s_0, s_1, ..., s_n] under arithmetic Brownian motion."""
    step_sd = sigma * math.sqrt(dt)
    path = [s0]
    s = s0
    for _ in range(n_steps):
        s = s + step_sd * rng.gauss(0.0, 1.0)
        path.append(s)
    return path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/intraday/marketmaking/test_price_path.py -v` (expect 3 passing). Then ruff + mypy clean.

- [ ] **Step 5: Commit**

```bash
git add quant/intraday/marketmaking/price_path.py tests/intraday/marketmaking/test_price_path.py
git commit -m "feat(intraday-mm): seeded arithmetic Brownian mid-price path"
```

---

### Task 5: The market-making simulator

**Files:**
- Create: `quant/intraday/marketmaking/simulator.py`
- Test: `tests/intraday/marketmaking/test_simulator.py`

- [ ] **Step 1: Write the failing test**

Create `tests/intraday/marketmaking/test_simulator.py`:

```python
import math

from quant.intraday.marketmaking.config import MMConfig
from quant.intraday.marketmaking.price_path import abm_path
from quant.intraday.marketmaking.simulator import MMResult, run_market_making
import random


def _prices(n, seed=11):
    return abm_path(s0=100.0, sigma=0.02, dt=1.0, n_steps=n, rng=random.Random(seed))


def test_result_shape_and_determinism():
    cfg = MMConfig(horizon_seconds=500.0, dt_seconds=1.0, seed=5)
    prices = _prices(cfg.n_steps)
    r1 = run_market_making(prices, cfg)
    r2 = run_market_making(prices, cfg)
    assert isinstance(r1, MMResult)
    assert r1 == r2                                   # deterministic on seed
    assert r1.n_bid_fills >= 0 and r1.n_ask_fills >= 0
    assert math.isfinite(r1.final_pnl)
    assert len(r1.inventory_path) == len(prices)


def test_pnl_conservation():
    cfg = MMConfig(horizon_seconds=400.0, dt_seconds=1.0, seed=2)
    prices = _prices(cfg.n_steps)
    r = run_market_making(prices, cfg)
    # final_pnl == cash + inventory * last_mid, by construction
    assert math.isclose(r.final_pnl, r.cash + r.terminal_inventory * prices[-1], rel_tol=1e-9)


def test_higher_gamma_controls_inventory():
    # The core A-S behavior: more risk aversion -> tighter inventory on the same path.
    prices = _prices(800)
    cfg_lo = MMConfig(gamma=0.01, horizon_seconds=800.0, dt_seconds=1.0, seed=4)
    cfg_hi = MMConfig(gamma=2.0, horizon_seconds=800.0, dt_seconds=1.0, seed=4)
    lo = run_market_making(prices, cfg_lo)
    hi = run_market_making(prices, cfg_hi)
    assert hi.max_abs_inventory <= lo.max_abs_inventory
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/intraday/marketmaking/test_simulator.py -v`
Expected: FAIL — ModuleNotFoundError.

- [ ] **Step 3: Write minimal implementation**

Create `quant/intraday/marketmaking/simulator.py`:

```python
"""Deterministic Avellaneda-Stoikov market-making simulator. Steps a mid-price path,
quotes via the A-S model, draws fills via the intensity model, tracks inventory/cash/
P&L. Fully determined by (prices, config) including config.seed."""

from __future__ import annotations

import random
from dataclasses import dataclass

from quant.intraday.marketmaking.avellaneda_stoikov import quotes
from quant.intraday.marketmaking.config import MMConfig
from quant.intraday.marketmaking.intensity import draws_fill, fill_probability


@dataclass(frozen=True)
class MMResult:
    final_pnl: float
    cash: float
    n_bid_fills: int
    n_ask_fills: int
    inventory_path: list[int]
    mean_abs_inventory: float
    max_abs_inventory: int
    terminal_inventory: int
    spread_captured: float


def run_market_making(prices: list[float], config: MMConfig) -> MMResult:
    rng = random.Random(config.seed)
    inventory = 0
    cash = 0.0
    spread_captured = 0.0
    n_bid = 0
    n_ask = 0
    inv_path = [0]
    T = config.horizon_seconds
    dt = config.dt_seconds
    lot = config.lot_size

    # The last price has no "next step" to fill against; quote on prices[:-1] and
    # mark the book at prices[-1] at the end.
    for i, mid in enumerate(prices[:-1]):
        tau = max(0.0, T - i * dt)
        bid, ask = quotes(mid, inventory, config.gamma, config.sigma, tau, config.k)
        p_bid = fill_probability(delta=mid - bid, a=config.fill_rate_a, k=config.k, dt=dt)
        p_ask = fill_probability(delta=ask - mid, a=config.fill_rate_a, k=config.k, dt=dt)
        if draws_fill(p_bid, rng):           # we BUY at our bid
            inventory += lot
            cash -= bid * lot
            spread_captured += abs(mid - bid) * lot
            n_bid += 1
        if draws_fill(p_ask, rng):           # we SELL at our ask
            inventory -= lot
            cash += ask * lot
            spread_captured += abs(ask - mid) * lot
            n_ask += 1
        inv_path.append(inventory)

    last_mid = prices[-1]
    final_pnl = cash + inventory * last_mid
    abs_inv = [abs(q) for q in inv_path]
    return MMResult(
        final_pnl=final_pnl,
        cash=cash,
        n_bid_fills=n_bid,
        n_ask_fills=n_ask,
        inventory_path=inv_path,
        mean_abs_inventory=sum(abs_inv) / len(abs_inv),
        max_abs_inventory=max(abs_inv),
        terminal_inventory=inventory,
        spread_captured=spread_captured,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/intraday/marketmaking/test_simulator.py -v` (expect 3 passing). If `test_higher_gamma_controls_inventory` is flaky for the chosen seed (the property is statistical), try a longer path / stronger γ contrast or a different seed until it holds ROBUSTLY — do NOT weaken the `<=` assertion away from the real A-S property; if it genuinely fails, report it as a possible bug in the quoting/sim wiring. Then ruff + mypy clean.

- [ ] **Step 5: Commit**

```bash
git add quant/intraday/marketmaking/simulator.py tests/intraday/marketmaking/test_simulator.py
git commit -m "feat(intraday-mm): deterministic A-S market-making simulator"
```

---

### Task 6: Gamma sweep (the tradeoff)

**Files:**
- Create: `quant/intraday/marketmaking/evaluate.py`
- Test: `tests/intraday/marketmaking/test_evaluate.py`

- [ ] **Step 1: Write the failing test**

Create `tests/intraday/marketmaking/test_evaluate.py`:

```python
import random

from quant.intraday.marketmaking.config import MMConfig
from quant.intraday.marketmaking.evaluate import SweepPoint, gamma_sweep
from quant.intraday.marketmaking.price_path import abm_path


def test_sweep_returns_point_per_gamma_in_order():
    cfg = MMConfig(horizon_seconds=600.0, dt_seconds=1.0, seed=9)
    prices = abm_path(s0=100.0, sigma=0.02, dt=1.0, n_steps=cfg.n_steps, rng=random.Random(9))
    gammas = [0.01, 0.1, 1.0, 5.0]
    pts = gamma_sweep(prices, cfg, gammas)
    assert [p.gamma for p in pts] == gammas
    assert all(isinstance(p, SweepPoint) for p in pts)


def test_sweep_shows_inventory_control_tradeoff():
    cfg = MMConfig(horizon_seconds=800.0, dt_seconds=1.0, seed=9)
    prices = abm_path(s0=100.0, sigma=0.02, dt=1.0, n_steps=cfg.n_steps, rng=random.Random(9))
    pts = gamma_sweep(prices, cfg, [0.01, 5.0])
    # Higher gamma -> tighter inventory control (>= because ties are allowed)
    assert pts[1].max_abs_inventory <= pts[0].max_abs_inventory
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/intraday/marketmaking/test_evaluate.py -v`
Expected: FAIL — ModuleNotFoundError.

- [ ] **Step 3: Write minimal implementation**

Create `quant/intraday/marketmaking/evaluate.py`:

```python
"""Sweep risk-aversion gamma to show the A-S tradeoff: low gamma = tight spread, many
fills, higher inventory risk; high gamma = wide spread, fewer fills, controlled
inventory. The market-making analog of the optimal-execution efficient frontier."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from quant.intraday.marketmaking.config import MMConfig
from quant.intraday.marketmaking.simulator import run_market_making


@dataclass(frozen=True)
class SweepPoint:
    gamma: float
    final_pnl: float
    n_fills: int
    mean_abs_inventory: float
    max_abs_inventory: int
    terminal_inventory: int


def gamma_sweep(prices: list[float], config: MMConfig, gammas: list[float]) -> list[SweepPoint]:
    """Run the simulator on the SAME path+seed for each gamma."""
    pts: list[SweepPoint] = []
    for g in gammas:
        cfg = dataclasses.replace(config, gamma=g)
        r = run_market_making(prices, cfg)
        pts.append(SweepPoint(
            gamma=g,
            final_pnl=r.final_pnl,
            n_fills=r.n_bid_fills + r.n_ask_fills,
            mean_abs_inventory=r.mean_abs_inventory,
            max_abs_inventory=r.max_abs_inventory,
            terminal_inventory=r.terminal_inventory,
        ))
    return pts
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/intraday/marketmaking/test_evaluate.py -v` (expect 2 passing). Then ruff + mypy clean.

- [ ] **Step 5: Commit**

```bash
git add quant/intraday/marketmaking/evaluate.py tests/intraday/marketmaking/test_evaluate.py
git commit -m "feat(intraday-mm): gamma sweep (spread-capture vs inventory-risk tradeoff)"
```

---

### Task 7: CLI — `quant intraday mm simulate` + `sweep`

**Files:**
- Modify: `quant/intraday/cli.py`
- Test: `tests/intraday/marketmaking/test_mm_cli.py`

- [ ] **Step 1: Write the failing test**

Create `tests/intraday/marketmaking/test_mm_cli.py`:

```python
from click.testing import CliRunner

from quant.intraday.cli import intraday


def test_mm_group_exists():
    r = CliRunner().invoke(intraday, ["mm", "--help"])
    assert r.exit_code == 0
    assert "simulate" in r.output and "sweep" in r.output


def test_simulate_prints_pnl_and_inventory():
    r = CliRunner().invoke(intraday, ["mm", "simulate", "--symbol", "QQQ", "--seed", "5"])
    assert r.exit_code == 0
    assert "QQQ" in r.output
    assert "pnl" in r.output.lower()
    assert "inventory" in r.output.lower()


def test_sweep_prints_table_with_gamma_and_note():
    r = CliRunner().invoke(intraday, ["mm", "sweep", "--symbol", "QQQ"])
    assert r.exit_code == 0
    assert "gamma" in r.output.lower()
    assert "stylized" in r.output.lower()   # honesty note present
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/intraday/marketmaking/test_mm_cli.py -v`
Expected: FAIL — `mm` not a command of `intraday`.

- [ ] **Step 3: Write the implementation**

In `quant/intraday/cli.py`, add (after the existing groups; reuse the module-level `import click`):

```python
@intraday.group()
def mm() -> None:
    """Avellaneda-Stoikov market-making simulator (sim/research only)."""


def _mm_demo_prices(seed: int, steps: int) -> list[float]:
    import random as _random

    from quant.intraday.marketmaking.price_path import abm_path
    # Representative mega-liquid ETF anchor (price 400, ~0.02 $/sqrt(s)) so it runs
    # without live data.
    return abm_path(s0=400.0, sigma=0.02, dt=1.0, n_steps=steps, rng=_random.Random(seed))


@mm.command()
@click.option("--symbol", required=True)
@click.option("--gamma", type=float, default=None, help="risk aversion (default MMConfig)")
@click.option("--seed", type=int, default=7)
@click.option("--steps", type=int, default=600)
def simulate(symbol: str, gamma: float | None, seed: int, steps: int) -> None:
    """Run one A-S market-making episode and print P&L + inventory stats."""
    import dataclasses

    from quant.intraday.marketmaking.config import MMConfig
    from quant.intraday.marketmaking.simulator import run_market_making

    cfg = MMConfig(horizon_seconds=float(steps), dt_seconds=1.0, seed=seed)
    if gamma is not None:
        cfg = dataclasses.replace(cfg, gamma=gamma)
    prices = _mm_demo_prices(seed, cfg.n_steps)
    r = run_market_making(prices, cfg)
    click.echo(f"A-S market making for {symbol} (gamma={cfg.gamma}, {cfg.n_steps} steps, seed={seed}):")
    click.echo(f"  final pnl:        {r.final_pnl:.2f}")
    click.echo(f"  spread captured:  {r.spread_captured:.2f}")
    click.echo(f"  fills:            {r.n_bid_fills} bid / {r.n_ask_fills} ask")
    click.echo(f"  inventory:        mean|q|={r.mean_abs_inventory:.2f} max|q|={r.max_abs_inventory} terminal={r.terminal_inventory}")
    click.echo("note: stylized A-S model (A, k are assumed parameters, not a live edge).")


@mm.command()
@click.option("--symbol", required=True)
@click.option("--seed", type=int, default=7)
@click.option("--steps", type=int, default=800)
def sweep(symbol: str, seed: int, steps: int) -> None:
    """Print the gamma tradeoff table (spread-capture vs inventory-risk)."""
    from quant.intraday.marketmaking.config import MMConfig
    from quant.intraday.marketmaking.evaluate import gamma_sweep

    cfg = MMConfig(horizon_seconds=float(steps), dt_seconds=1.0, seed=seed)
    prices = _mm_demo_prices(seed, cfg.n_steps)
    pts = gamma_sweep(prices, cfg, [0.01, 0.05, 0.1, 0.5, 1.0, 5.0])
    click.echo(f"A-S gamma sweep for {symbol} ({cfg.n_steps} steps, seed={seed}):")
    click.echo("  gamma     pnl         fills    mean|q|   max|q|   terminal_q")
    for p in pts:
        click.echo(f"  {p.gamma:<8.2f}  {p.final_pnl:<10.2f}  {p.n_fills:<7d}  "
                   f"{p.mean_abs_inventory:<8.2f}  {p.max_abs_inventory:<7d}  {p.terminal_inventory}")
    click.echo("note: stylized A-S model (A, k are assumed parameters, not a live edge).")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/intraday/marketmaking/test_mm_cli.py -v` (expect 3 passing). Confirm no regression to the intraday CLI: `uv run pytest tests/intraday/ -q`. Then `uv run ruff check quant/intraday/cli.py tests/intraday/marketmaking/test_mm_cli.py` and `uv run mypy quant/intraday/cli.py` (clean).

- [ ] **Step 5: Commit**

```bash
git add quant/intraday/cli.py tests/intraday/marketmaking/test_mm_cli.py
git commit -m "feat(intraday-mm): `quant intraday mm` simulate + sweep CLI"
```

---

### Task 8: Full-suite green + lint/type gate

**Files:** none (verification task)

- [ ] **Step 1: Run the market-making suite**

Run: `uv run pytest tests/intraday/marketmaking/ -q`
Expected: ALL pass.

- [ ] **Step 2: Run the full suite excluding network-gated tests**

Run: `uv run pytest -m "not network and not alpaca" -q`
Expected: green (prior baseline + the new tests). (The unfiltered `pytest` blocks on live-Alpaca tests — always exclude `network`/`alpaca`.)

- [ ] **Step 3: Lint + type gate**

Run: `uv run ruff check . && uv run mypy quant`
Expected: clean. Fix any findings without blanket ignores.

- [ ] **Step 4: Manual artifact check**

Run: `uv run quant intraday mm sweep --symbol QQQ`
Expected: a γ table where higher γ shows lower `max|q|` (inventory control) — eyeball that the tradeoff is visible.

- [ ] **Step 5: Commit any fixes**

```bash
git add -A
git commit -m "chore(intraday-mm): full-suite green + lint/type clean"
```

---

## Notes for the implementer

- **Standalone by design** — this package imports nothing from `quant/intraday/live/`, `quant/intraday/execution/`, or `quant/intraday/sim/`. Do not wire it into any live path.
- **ABM not GBM** — absolute volatility; the mid is arithmetic Brownian. This is faithful to A-S and makes the σ² terms in the quoting math dimensionally consistent.
- **Determinism is a hard requirement** — every random draw goes through `random.Random(config.seed)`. Same inputs ⇒ identical `MMResult`. Tests rely on it.
- **The γ-inventory-control property is the real point** — if `test_higher_gamma_controls_inventory` won't hold robustly, that's a signal the quoting/sim wiring is wrong, not a reason to weaken the test.
- **Honesty** — both CLI commands print the "stylized model" note; keep it.
```
