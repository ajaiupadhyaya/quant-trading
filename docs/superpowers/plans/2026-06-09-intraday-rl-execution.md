# Intraday RL Execution Agent (Tabular Q-learning) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone, numpy-only tabular Q-learning agent that learns to liquidate a parent order over a discretized `(time, inventory)` execution MDP, and a CLI that shows it converging to the Almgren–Chriss optimum and beating TWAP.

**Architecture:** A new self-contained `quant/intraday/rl/` subpackage. `env.py` defines the execution MDP (square-root impact + A-C risk penalty + terminal force-liquidation, seeded ABM mid); `qlearning.py` trains a Q-table by ε-greedy Q-learning; `policy.py` extracts the greedy policy + rolls it to a child-size schedule; `evaluate.py` costs the learned / A-C / TWAP schedules through ONE shared env cost function; a CLI surfaces it. Reuses `quant/backtest/impact.py` (impact) and `quant/intraday/execution/` (A-C `optimal_schedule`, `twap`, `calibrate`).

**Tech Stack:** Python 3.11+, `uv`, numpy (Q-table), `random.Random` (seeded sampling), dataclasses, Click, pytest. NO torch.

**Conventions:** Run everything with `uv run`. Keep `uv run ruff check .` and `uv run mypy quant` clean. TDD: failing test → fail → implement → pass → commit. ONLY `git add` the exact paths each task lists.

**Honest framing:** on this low-dim MDP tabular RL rediscovers the DP/A-C optimum; the showcase is the agent *learning* it from rewards and beating TWAP. CLI prints this note.

**Spec:** `docs/superpowers/specs/2026-06-09-intraday-rl-execution-design.md`

**Reused signatures (verified):** `market_impact_bps(trade_notional, adv_dollar, impact_coef_bps) -> float`; `optimal_schedule(*, total_shares, n_intervals, tau, sigma, eta, gamma, risk_aversion) -> ACPlan` (`.child_sizes`); `twap(*, total_shares, n_intervals) -> list[int]`; `calibrate(*, price, slice_shares, adv_dollar, recent_returns, config) -> (sigma, eta, gamma)`.

**File map (`quant/intraday/rl/`):** `config.py` (RLConfig), `env.py` (ExecutionEnv + `step_cost`), `qlearning.py` (train → TrainResult), `policy.py` (greedy_action, rollout_schedule), `evaluate.py` (cost_schedule + compare). Plus modify `quant/intraday/cli.py` (add the `rl` group).

---

### Task 1: RLConfig

**Files:**
- Create: `quant/intraday/rl/__init__.py` (empty)
- Create: `tests/intraday/rl/__init__.py` (empty)
- Create: `quant/intraday/rl/config.py`
- Test: `tests/intraday/rl/test_config.py`

- [ ] **Step 1: Create the two empty `__init__.py` package markers.**

- [ ] **Step 2: Write the failing test**

Create `tests/intraday/rl/test_config.py`:

```python
import pytest

from quant.intraday.rl.config import RLConfig


def test_defaults():
    c = RLConfig()
    assert c.total_shares >= 1
    assert c.n_steps >= 1
    assert c.n_actions >= 2
    assert 0 < c.alpha <= 1
    assert 0 <= c.gamma_discount <= 1
    assert c.epsilon_start >= c.epsilon_end >= 0
    assert c.n_episodes >= 1
    assert c.risk_aversion >= 0
    assert c.sigma > 0 and c.dt > 0 and c.start_price > 0
    assert isinstance(c.seed, int)


def test_rejects_bad_values():
    with pytest.raises(ValueError):
        RLConfig(total_shares=0)
    with pytest.raises(ValueError):
        RLConfig(n_actions=1)            # need >= 2 (do-nothing + something)
    with pytest.raises(ValueError):
        RLConfig(alpha=0.0)
    with pytest.raises(ValueError):
        RLConfig(epsilon_start=0.1, epsilon_end=0.5)   # start < end
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/intraday/rl/test_config.py -v`
Expected: FAIL — ModuleNotFoundError.

- [ ] **Step 4: Write minimal implementation**

Create `quant/intraday/rl/config.py`:

```python
"""Configuration for the tabular Q-learning execution agent. No magic numbers per
the Charter; all knobs here. Stylized: tabular RL rediscovers the DP/A-C optimum."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RLConfig:
    total_shares: int = 20          # parent order to liquidate
    n_steps: int = 10               # horizon (discrete steps)
    n_actions: int = 5              # discrete sell-fraction levels (0 .. 1 of remaining)
    alpha: float = 0.2              # learning rate
    gamma_discount: float = 1.0     # undiscounted episodic objective
    epsilon_start: float = 1.0      # exploration schedule
    epsilon_end: float = 0.02
    n_episodes: int = 20_000
    risk_aversion: float = 0.01     # lambda in the A-C risk penalty
    impact_coef_bps: float = 10.0   # sqrt-impact coefficient at 100% ADV
    adv_dollar: float = 5_000_000_000.0
    sigma: float = 0.02             # absolute vol, price units / sqrt(step)
    dt: float = 1.0                 # step length
    start_price: float = 100.0
    seed: int = 7

    def __post_init__(self) -> None:
        if self.total_shares < 1:
            raise ValueError("total_shares must be >= 1")
        if self.n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        if self.n_actions < 2:
            raise ValueError("n_actions must be >= 2")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if not 0.0 <= self.gamma_discount <= 1.0:
            raise ValueError("gamma_discount must be in [0, 1]")
        if not self.epsilon_start >= self.epsilon_end >= 0.0:
            raise ValueError("require epsilon_start >= epsilon_end >= 0")
        if self.n_episodes < 1:
            raise ValueError("n_episodes must be >= 1")
        if self.risk_aversion < 0:
            raise ValueError("risk_aversion must be >= 0")
        for name in ("sigma", "dt", "start_price", "adv_dollar", "impact_coef_bps"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/intraday/rl/test_config.py -v` (expect 2 passing). Then `uv run ruff check quant/intraday/rl/ tests/intraday/rl/` and `uv run mypy quant/intraday/rl/config.py` (clean).

- [ ] **Step 6: Commit**

```bash
git add quant/intraday/rl/__init__.py quant/intraday/rl/config.py tests/intraday/rl/
git commit -m "feat(intraday-rl): RLConfig for tabular Q-learning execution agent"
```

---

### Task 2: ExecutionEnv (the MDP)

**Files:**
- Create: `quant/intraday/rl/env.py`
- Test: `tests/intraday/rl/test_env.py`

`step_cost` is a module-level pure function (reused by both the env and `evaluate.cost_schedule`) so the cost model lives in ONE place. The env holds a seeded `random.Random` for the ABM mid.

- [ ] **Step 1: Write the failing test**

Create `tests/intraday/rl/test_env.py`:

```python
import random

from quant.intraday.rl.config import RLConfig
from quant.intraday.rl.env import ExecutionEnv, step_cost


def test_step_cost_is_nonnegative_and_zero_when_flat_and_no_trade():
    cfg = RLConfig()
    # no trade, no inventory -> zero cost
    assert step_cost(traded=0, inventory_after=0, price=100.0, config=cfg) == 0.0
    # trading incurs positive impact cost
    assert step_cost(traded=10, inventory_after=0, price=100.0, config=cfg) > 0.0
    # holding inventory incurs positive risk cost
    assert step_cost(traded=0, inventory_after=10, price=100.0, config=cfg) > 0.0


def test_reset_returns_full_inventory_and_full_horizon():
    cfg = RLConfig(total_shares=20, n_steps=10)
    env = ExecutionEnv(cfg, seed=1)
    s = env.reset()
    assert s == (10, 20)   # (steps_remaining, inventory_remaining)


def test_step_sells_fraction_and_decrements_state():
    cfg = RLConfig(total_shares=20, n_steps=10, n_actions=5)
    env = ExecutionEnv(cfg, seed=1)
    env.reset()
    # action index 2 of 5 -> fraction 0.5 -> sell 10 of 20
    (sr, inv), reward, done = env.step(2)
    assert sr == 9 and inv == 10
    assert reward <= 0.0          # reward is negative cost
    assert not done


def test_inventory_never_negative_and_action_clamps():
    cfg = RLConfig(total_shares=20, n_steps=10, n_actions=5)
    env = ExecutionEnv(cfg, seed=1)
    env.reset()
    # max action (sell all) repeatedly -> inventory hits 0 and stays there
    env.step(4)                   # sell all 20
    (sr, inv), reward, done = env.step(4)
    assert inv == 0


def test_terminal_force_liquidates_remaining():
    cfg = RLConfig(total_shares=20, n_steps=2, n_actions=5)
    env = ExecutionEnv(cfg, seed=1)
    env.reset()
    env.step(0)                   # do nothing (still hold 20)
    (sr, inv), reward, done = env.step(0)   # last step, still did nothing
    assert done and sr == 0
    assert inv == 0               # forced flat at the deadline
    assert reward < 0.0           # charged the forced-liquidation impact
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/intraday/rl/test_env.py -v`
Expected: FAIL — ModuleNotFoundError.

- [ ] **Step 3: Write minimal implementation**

Create `quant/intraday/rl/env.py`:

```python
"""Execution MDP for the tabular Q-learning agent. State = (steps_remaining,
inventory_remaining). Action index in [0, n_actions) maps to a sell-fraction of the
remaining inventory. Reward = -(square-root impact + A-C risk penalty); leftover
inventory is force-liquidated at the deadline. The mid follows a seeded ABM path."""

from __future__ import annotations

import math
import random

from quant.backtest.impact import market_impact_bps
from quant.intraday.rl.config import RLConfig

State = tuple[int, int]  # (steps_remaining, inventory_remaining)


def step_cost(*, traded: int, inventory_after: int, price: float, config: RLConfig) -> float:
    """Per-step COST (>= 0): square-root market impact of `traded` shares plus the
    A-C inventory-risk penalty on `inventory_after`. Cost, not reward (reward = -cost)."""
    impact_cost = 0.0
    if traded > 0:
        notional = price * traded
        imp_bps = market_impact_bps(notional, config.adv_dollar, config.impact_coef_bps)
        impact_cost = price * (imp_bps / 1e4) * traded
    risk_cost = config.risk_aversion * config.sigma**2 * inventory_after**2 * config.dt
    return impact_cost + risk_cost


def action_to_shares(action_index: int, inventory: int, n_actions: int) -> int:
    """Map a discrete action to a share count: evenly spaced sell-fractions of the
    remaining inventory, 0 (do nothing) .. 1 (sell all). Clamped to inventory."""
    fraction = action_index / (n_actions - 1)
    return min(inventory, int(round(fraction * inventory)))


class ExecutionEnv:
    def __init__(self, config: RLConfig, *, seed: int) -> None:
        self._cfg = config
        self._rng = random.Random(seed)
        self._steps_remaining = config.n_steps
        self._inventory = config.total_shares
        self._price = config.start_price

    def reset(self) -> State:
        self._steps_remaining = self._cfg.n_steps
        self._inventory = self._cfg.total_shares
        self._price = self._cfg.start_price
        return (self._steps_remaining, self._inventory)

    def step(self, action_index: int) -> tuple[State, float, bool]:
        cfg = self._cfg
        traded = action_to_shares(action_index, self._inventory, cfg.n_actions)
        inventory_after = self._inventory - traded
        cost = step_cost(traded=traded, inventory_after=inventory_after,
                         price=self._price, config=cfg)

        self._steps_remaining -= 1
        done = self._steps_remaining == 0

        # Terminal force-liquidation of any remainder (charged its impact).
        if done and inventory_after > 0:
            notional = self._price * inventory_after
            imp_bps = market_impact_bps(notional, cfg.adv_dollar, cfg.impact_coef_bps)
            cost += self._price * (imp_bps / 1e4) * inventory_after
            inventory_after = 0

        self._inventory = inventory_after
        # advance the mid one ABM step (zero-drift, so the expected-optimal policy is
        # path-independent; the stochasticity makes this a genuine RL problem).
        self._price = self._price + cfg.sigma * math.sqrt(cfg.dt) * self._rng.gauss(0.0, 1.0)
        return (self._steps_remaining, inventory_after), -cost, done
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/intraday/rl/test_env.py -v` (expect 5 passing). Then ruff + mypy clean on the new file.

- [ ] **Step 5: Commit**

```bash
git add quant/intraday/rl/env.py tests/intraday/rl/test_env.py
git commit -m "feat(intraday-rl): execution MDP env (sqrt impact + risk penalty + force-liquidation)"
```

---

### Task 3: Q-learning trainer

**Files:**
- Create: `quant/intraday/rl/qlearning.py`
- Test: `tests/intraday/rl/test_qlearning.py`

- [ ] **Step 1: Write the failing test**

Create `tests/intraday/rl/test_qlearning.py`:

```python
import numpy as np

from quant.intraday.rl.config import RLConfig
from quant.intraday.rl.qlearning import TrainResult, train


def _small():
    return RLConfig(total_shares=20, n_steps=10, n_actions=5, n_episodes=4000, seed=3)


def test_qtable_shape_and_determinism():
    cfg = _small()
    r1 = train(cfg)
    r2 = train(cfg)
    assert isinstance(r1, TrainResult)
    assert r1.qtable.shape == (cfg.n_steps + 1, cfg.total_shares + 1, cfg.n_actions)
    assert np.array_equal(r1.qtable, r2.qtable)          # deterministic on seed
    assert len(r1.training_curve) > 0
    assert np.isfinite(r1.qtable).all()


def test_training_curve_improves():
    # mean episode cost in the last block should be lower than the first block
    cfg = _small()
    r = train(cfg)
    assert r.training_curve[-1] < r.training_curve[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/intraday/rl/test_qlearning.py -v`
Expected: FAIL — ModuleNotFoundError.

- [ ] **Step 3: Write minimal implementation**

Create `quant/intraday/rl/qlearning.py`:

```python
"""Tabular Q-learning for the execution MDP. Epsilon-greedy with a linear decay;
deterministic given config.seed. Returns the Q-table and a per-block mean-episode-cost
training curve (to show convergence)."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from quant.intraday.rl.config import RLConfig
from quant.intraday.rl.env import ExecutionEnv


@dataclass(frozen=True)
class TrainResult:
    qtable: np.ndarray            # [n_steps+1, total_shares+1, n_actions]
    training_curve: list[float]   # mean episode COST per block of episodes


def _epsilon(episode: int, cfg: RLConfig) -> float:
    frac = episode / max(1, cfg.n_episodes - 1)
    return cfg.epsilon_start + (cfg.epsilon_end - cfg.epsilon_start) * frac


def train(config: RLConfig) -> TrainResult:
    cfg = config
    q = np.zeros((cfg.n_steps + 1, cfg.total_shares + 1, cfg.n_actions), dtype=float)
    explore_rng = random.Random(cfg.seed)
    n_blocks = 20
    block_size = max(1, cfg.n_episodes // n_blocks)
    curve: list[float] = []
    block_cost = 0.0
    block_count = 0

    for ep in range(cfg.n_episodes):
        env = ExecutionEnv(cfg, seed=cfg.seed + 1 + ep)  # fresh seeded path per episode
        sr, inv = env.reset()
        eps = _epsilon(ep, cfg)
        ep_cost = 0.0
        done = False
        while not done:
            if explore_rng.random() < eps:
                action = explore_rng.randrange(cfg.n_actions)
            else:
                action = int(np.argmax(q[sr, inv]))
            (nsr, ninv), reward, done = env.step(action)
            ep_cost += -reward
            # Q-update (undiscounted by default; bootstrap on next-state max, 0 if done).
            future = 0.0 if done else float(np.max(q[nsr, ninv]))
            target = reward + cfg.gamma_discount * future
            q[sr, inv, action] += cfg.alpha * (target - q[sr, inv, action])
            sr, inv = nsr, ninv
        block_cost += ep_cost
        block_count += 1
        if block_count == block_size:
            curve.append(block_cost / block_count)
            block_cost = 0.0
            block_count = 0
    if block_count:
        curve.append(block_cost / block_count)
    return TrainResult(qtable=q, training_curve=curve)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/intraday/rl/test_qlearning.py -v` (expect 2 passing). If `test_training_curve_improves` is marginal, raise `n_episodes` in `_small()` until the last block is clearly below the first (the property is real — early ε≈1 is random, late ε≈0 is greedy-optimal — so it should hold strongly). Then ruff + mypy clean. mypy note: `np.argmax` returns `np.intp`; wrap in `int(...)` (done). For `q[sr, inv]` indexing returning `np.ndarray`, `np.max`/`np.argmax` are fine.

- [ ] **Step 5: Commit**

```bash
git add quant/intraday/rl/qlearning.py tests/intraday/rl/test_qlearning.py
git commit -m "feat(intraday-rl): tabular Q-learning trainer (epsilon-greedy, deterministic)"
```

---

### Task 4: Greedy policy + schedule rollout

**Files:**
- Create: `quant/intraday/rl/policy.py`
- Test: `tests/intraday/rl/test_policy.py`

- [ ] **Step 1: Write the failing test**

Create `tests/intraday/rl/test_policy.py`:

```python
import numpy as np

from quant.intraday.rl.config import RLConfig
from quant.intraday.rl.policy import greedy_action, rollout_schedule
from quant.intraday.rl.qlearning import train


def test_greedy_action_picks_argmax():
    q = np.zeros((3, 5, 4))
    q[1, 2, 3] = 5.0   # best action at state (1,2) is index 3
    assert greedy_action(q, (1, 2)) == 3


def test_rollout_schedule_liquidates_full_parent():
    cfg = RLConfig(total_shares=20, n_steps=10, n_actions=5, n_episodes=4000, seed=3)
    r = train(cfg)
    sched = rollout_schedule(r.qtable, cfg)
    assert sum(sched) == cfg.total_shares      # fully liquidates
    assert all(n >= 0 for n in sched)
    assert len(sched) == cfg.n_steps
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/intraday/rl/test_policy.py -v`
Expected: FAIL — ModuleNotFoundError.

- [ ] **Step 3: Write minimal implementation**

Create `quant/intraday/rl/policy.py`:

```python
"""Greedy policy extraction from a trained Q-table, and a noiseless rollout of that
policy into a child-size schedule (for comparison against A-C / TWAP)."""

from __future__ import annotations

import numpy as np

from quant.intraday.rl.config import RLConfig
from quant.intraday.rl.env import action_to_shares

State = tuple[int, int]


def greedy_action(qtable: np.ndarray, state: State) -> int:
    sr, inv = state
    return int(np.argmax(qtable[sr, inv]))


def rollout_schedule(qtable: np.ndarray, config: RLConfig) -> list[int]:
    """Greedily roll the policy forward (no price noise needed — the schedule is the
    sequence of share counts the greedy policy trades). Force-liquidates any remainder
    on the final step so the schedule sums to total_shares."""
    cfg = config
    inv = cfg.total_shares
    sched: list[int] = []
    for step in range(cfg.n_steps):
        steps_remaining = cfg.n_steps - step
        action = greedy_action(qtable, (steps_remaining, inv))
        traded = action_to_shares(action, inv, cfg.n_actions)
        if step == cfg.n_steps - 1:        # final step: force-liquidate the remainder
            traded = inv
        sched.append(traded)
        inv -= traded
    return sched
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/intraday/rl/test_policy.py -v` (expect 2 passing). Then ruff + mypy clean.

- [ ] **Step 5: Commit**

```bash
git add quant/intraday/rl/policy.py tests/intraday/rl/test_policy.py
git commit -m "feat(intraday-rl): greedy policy + schedule rollout"
```

---

### Task 5: Evaluation — learned vs A-C vs TWAP (shared cost)

**Files:**
- Create: `quant/intraday/rl/evaluate.py`
- Test: `tests/intraday/rl/test_evaluate.py`

`cost_schedule` replays a FIXED child-size schedule through the same `step_cost` model over a seeded ABM path (with terminal force-liquidation), so learned / A-C / TWAP are costed identically. `compare` averages over `n_eval_paths`.

- [ ] **Step 1: Write the failing test**

Create `tests/intraday/rl/test_evaluate.py`:

```python
from quant.intraday.rl.config import RLConfig
from quant.intraday.rl.evaluate import compare, cost_schedule


def test_cost_schedule_positive_and_seed_deterministic():
    cfg = RLConfig(total_shares=20, n_steps=10)
    c1 = cost_schedule([2] * 10, cfg, seed=5)
    c2 = cost_schedule([2] * 10, cfg, seed=5)
    assert c1 == c2 and c1 > 0.0


def test_compare_returns_three_costs_and_learned_beats_twap():
    cfg = RLConfig(total_shares=20, n_steps=10, n_actions=5, n_episodes=6000, seed=3)
    res = compare(cfg, n_eval_paths=200)
    for key in ("learned", "almgren_chriss", "twap"):
        assert key in res and res[key] > 0.0
    # The learned policy should be no worse than naive TWAP (within tiny noise),
    # and close to the A-C optimum.
    assert res["learned"] <= res["twap"] * 1.05
    assert res["learned"] <= res["almgren_chriss"] * 1.25   # near the optimum
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/intraday/rl/test_evaluate.py -v`
Expected: FAIL — ModuleNotFoundError.

- [ ] **Step 3: Write minimal implementation**

Create `quant/intraday/rl/evaluate.py`:

```python
"""Evaluate the learned policy against the Almgren-Chriss optimum and TWAP, costing all
three schedules through ONE shared env cost model (step_cost) over seeded ABM paths."""

from __future__ import annotations

import math
import random
from typing import Any

from quant.intraday.execution.almgren_chriss import optimal_schedule
from quant.intraday.execution.baselines import twap
from quant.intraday.execution.calibrate import calibrate
from quant.intraday.execution.config import ExecConfig
from quant.intraday.rl.config import RLConfig
from quant.intraday.rl.env import step_cost
from quant.intraday.rl.policy import rollout_schedule
from quant.intraday.rl.qlearning import train


def cost_schedule(schedule: list[int], config: RLConfig, *, seed: int) -> float:
    """Replay a fixed child-size schedule through the env cost model on one seeded ABM
    path; force-liquidate any remainder on the last step. Returns total COST (>= 0)."""
    cfg = config
    rng = random.Random(seed)
    price = cfg.start_price
    inv = cfg.total_shares
    total = 0.0
    n = len(schedule)
    for i, raw in enumerate(schedule):
        traded = min(inv, raw)
        if i == n - 1:                     # final step force-liquidates the remainder
            traded = inv
        inv_after = inv - traded
        total += step_cost(traded=traded, inventory_after=inv_after, price=price, config=cfg)
        inv = inv_after
        price = price + cfg.sigma * math.sqrt(cfg.dt) * rng.gauss(0.0, 1.0)
    return total


def _ac_schedule(config: RLConfig) -> list[int]:
    """Almgren-Chriss optimal child sizes using the SAME risk aversion as the RL reward;
    eta is calibrated from the same sqrt-impact model (local linearization, as in A)."""
    cfg = config
    per_slice = max(1, cfg.total_shares // cfg.n_steps)
    _, eta, gamma = calibrate(
        price=cfg.start_price, slice_shares=per_slice, adv_dollar=cfg.adv_dollar,
        recent_returns=[0.0], config=ExecConfig(impact_coef_bps=cfg.impact_coef_bps),
    )
    plan = optimal_schedule(
        total_shares=cfg.total_shares, n_intervals=cfg.n_steps, tau=cfg.dt,
        sigma=cfg.sigma, eta=eta, gamma=gamma, risk_aversion=cfg.risk_aversion,
    )
    return plan.child_sizes


def _mean_cost(schedule: list[int], config: RLConfig, n_eval_paths: int) -> float:
    return sum(
        cost_schedule(schedule, config, seed=config.seed + 10_000 + j)
        for j in range(n_eval_paths)
    ) / n_eval_paths


def compare(config: RLConfig, n_eval_paths: int = 200) -> dict[str, Any]:
    """Train the agent and compare mean execution cost: learned vs A-C vs TWAP."""
    result = train(config)
    learned = rollout_schedule(result.qtable, config)
    ac = _ac_schedule(config)
    tw = twap(total_shares=config.total_shares, n_intervals=config.n_steps)
    return {
        "learned": _mean_cost(learned, config, n_eval_paths),
        "almgren_chriss": _mean_cost(ac, config, n_eval_paths),
        "twap": _mean_cost(tw, config, n_eval_paths),
        "learned_schedule": learned,
        "training_curve": result.training_curve,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/intraday/rl/test_evaluate.py -v` (expect 2 passing). The `learned <= twap*1.05` and `learned <= ac*1.25` tolerances are deliberately loose to absorb sampling noise on a small training budget; if the learned policy genuinely can't beat TWAP or approach A-C, that's a real signal — investigate the reward/MDP wiring (e.g. risk_aversion scaling) rather than loosening the bounds further. You may raise `n_episodes`/`n_eval_paths` so it holds robustly; report the actual three costs. Then ruff + mypy clean on `evaluate.py` (the `calibrate`/`optimal_schedule` reuse must import cleanly).

- [ ] **Step 5: Commit**

```bash
git add quant/intraday/rl/evaluate.py tests/intraday/rl/test_evaluate.py
git commit -m "feat(intraday-rl): evaluate learned vs A-C vs TWAP through shared cost model"
```

---

### Task 6: CLI — `quant intraday rl train` + `compare`

**Files:**
- Modify: `quant/intraday/cli.py`
- Test: `tests/intraday/rl/test_rl_cli.py`

- [ ] **Step 1: Write the failing test**

Create `tests/intraday/rl/test_rl_cli.py`:

```python
from click.testing import CliRunner

from quant.intraday.cli import intraday


def test_rl_group_exists():
    r = CliRunner().invoke(intraday, ["rl", "--help"])
    assert r.exit_code == 0
    assert "train" in r.output and "compare" in r.output


def test_train_prints_convergence():
    r = CliRunner().invoke(intraday, ["rl", "train", "--shares", "20", "--episodes", "3000"])
    assert r.exit_code == 0
    assert "converg" in r.output.lower() or "cost" in r.output.lower()


def test_compare_prints_three_policies_and_note():
    r = CliRunner().invoke(intraday, ["rl", "compare", "--shares", "20", "--episodes", "3000"])
    assert r.exit_code == 0
    out = r.output.lower()
    assert "learned" in out and "twap" in out and ("almgren" in out or "a-c" in out)
    assert "rediscover" in out or "stylized" in out   # honesty note
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/intraday/rl/test_rl_cli.py -v`
Expected: FAIL — `rl` not a command of `intraday`.

- [ ] **Step 3: Write the implementation**

In `quant/intraday/cli.py`, add (after the existing groups; reuse the module-level `import click`):

```python
@intraday.group()
def rl() -> None:
    """Tabular Q-learning execution agent (sim/research only)."""


@rl.command()
@click.option("--shares", type=int, default=20)
@click.option("--steps", type=int, default=10)
@click.option("--episodes", type=int, default=20000)
@click.option("--seed", type=int, default=7)
def train(shares: int, steps: int, episodes: int, seed: int) -> None:
    """Train the agent and print the convergence curve (mean episode cost per block)."""
    from quant.intraday.rl.config import RLConfig
    from quant.intraday.rl.qlearning import train as train_agent

    cfg = RLConfig(total_shares=shares, n_steps=steps, n_episodes=episodes, seed=seed)
    result = train_agent(cfg)
    click.echo(f"RL execution training ({shares} sh, {steps} steps, {episodes} episodes):")
    click.echo("  convergence (mean episode cost per block, first -> last):")
    curve = result.training_curve
    for i, c in enumerate(curve):
        if i == 0 or i == len(curve) - 1 or i % 5 == 0:
            click.echo(f"    block {i:>2}: {c:.4f}")
    click.echo(f"  improved from {curve[0]:.4f} to {curve[-1]:.4f}")
    click.echo("note: tabular RL rediscovers the DP/A-C optimum; the point is it LEARNS it.")


@rl.command()
@click.option("--shares", type=int, default=20)
@click.option("--steps", type=int, default=10)
@click.option("--episodes", type=int, default=20000)
@click.option("--seed", type=int, default=7)
@click.option("--eval-paths", type=int, default=300)
def compare(shares: int, steps: int, episodes: int, seed: int, eval_paths: int) -> None:
    """Compare learned policy vs Almgren-Chriss optimal vs TWAP (mean execution cost)."""
    from quant.intraday.rl.config import RLConfig
    from quant.intraday.rl.evaluate import compare as compare_policies

    cfg = RLConfig(total_shares=shares, n_steps=steps, n_episodes=episodes, seed=seed)
    res = compare_policies(cfg, n_eval_paths=eval_paths)
    click.echo(f"RL execution comparison ({shares} sh, {steps} steps, {eval_paths} eval paths):")
    click.echo(f"  learned (RL):     mean cost {res['learned']:.4f}")
    click.echo(f"  Almgren-Chriss:   mean cost {res['almgren_chriss']:.4f}")
    click.echo(f"  TWAP:             mean cost {res['twap']:.4f}")
    click.echo(f"  learned schedule: {res['learned_schedule']}")
    click.echo("note: tabular RL rediscovers the DP/A-C optimum; the point is it LEARNS it from rewards.")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/intraday/rl/test_rl_cli.py -v` (3 passing). Then `uv run pytest tests/intraday/ -q` (no regression to data/live/exec/mm groups). Then `uv run ruff check quant/intraday/cli.py tests/intraday/rl/test_rl_cli.py` and `uv run mypy quant/intraday/cli.py` (clean).

- [ ] **Step 5: Commit**

```bash
git add quant/intraday/cli.py tests/intraday/rl/test_rl_cli.py
git commit -m "feat(intraday-rl): `quant intraday rl` train + compare CLI"
```

---

### Task 7: Full-suite green + lint/type gate

**Files:** none (verification task)

- [ ] **Step 1: Run the RL suite**

Run: `uv run pytest tests/intraday/rl/ -q`
Expected: ALL pass.

- [ ] **Step 2: Run the full suite excluding network-gated tests**

Run: `uv run pytest -m "not network and not alpaca" -q`
Expected: green (prior baseline + new tests). (The unfiltered `pytest` blocks on live-Alpaca tests — always exclude `network`/`alpaca`.)

- [ ] **Step 3: Lint + type gate**

Run: `uv run ruff check . && uv run mypy quant`
Expected: clean. Fix any findings without blanket ignores.

- [ ] **Step 4: Manual artifact check**

Run: `uv run quant intraday rl compare --shares 20 --episodes 8000`
Expected: a table where `learned` mean cost is ≤ TWAP and close to Almgren-Chriss — eyeball that the learned agent matches the optimum and beats the naive baseline.

- [ ] **Step 5: Commit any fixes**

```bash
git add -A
git commit -m "chore(intraday-rl): full-suite green + lint/type clean"
```

---

## Notes for the implementer

- **Standalone, numpy-only** — no torch, no live path. Imports `quant/backtest/impact.py` and `quant/intraday/execution/` (A-C/TWAP/calibrate) for cost + baselines only.
- **One shared cost model** — `step_cost` lives in `env.py` and is used by BOTH the env and `evaluate.cost_schedule`, so learned/A-C/TWAP are costed identically (parity). Do not write a second cost model.
- **Determinism is required** — Q-table is deterministic given `config.seed`; tests rely on it. The exploration RNG and the per-episode env seeds both derive from `config.seed`.
- **The "it learns" properties are the real point** — training curve falls (early random ε≈1 → late greedy), learned ≤ TWAP, learned ≈ A-C. If these don't hold, fix the reward/MDP wiring (esp. `risk_aversion` scaling and the terminal force-liquidation), don't weaken the assertions.
- **Honesty** — both CLI commands print that tabular RL rediscovers the DP/A-C optimum; keep it.
```
