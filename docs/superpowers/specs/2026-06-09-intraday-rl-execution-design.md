# Intraday RL Execution Agent (Tabular Q-learning) — Design Spec

**Date:** 2026-06-09
**Status:** Approved design, pending implementation plan
**Scope:** Sub-project C of the intraday/60s showcase track. Independent of the spine (0) and market-making (B); reuses the optimal-execution baselines from Sub-project A.

---

## Context

The intraday showcase track has shipped Sub-project 0 (live 60s sleeve loop), A (Almgren–Chriss
optimal execution), and B (Avellaneda–Stoikov market making). This sub-project adds a
**reinforcement-learning execution agent**.

**Goal = portfolio/learning showcase** (see [[project-quant-trading-intraday]]). Decisions locked
in brainstorming:
1. **Tabular Q-learning** on a discretized execution MDP. The execution problem's state is
   low-dimensional (`time-remaining × inventory-remaining`), so tabular value-based RL is the
   right, faithful, dependency-free tool. This is the **Nevmyvaka–Feng–Kearns (2006)** "RL for
   optimal execution" formulation. Deep PPO/SAC was rejected: it needs PyTorch (a heavy dep the
   repo avoids) and is overkill for a 2-D state.
2. **Task = order liquidation; evaluation vs A-C optimal + TWAP** (reusing
   `quant/intraday/execution/`). **Sim/research only — no live wiring, numpy-only.**

**Honest framing (recorded here, surfaced in CLI):** on this low-dim, near-deterministic MDP,
tabular RL essentially *rediscovers* the dynamic-programming / Almgren–Chriss optimum. The
showcase value is demonstrating that a learning agent — given only per-step rewards — **converges
to the known-optimal execution behavior and beats the naive TWAP baseline**, not beating A-C.

---

## 1. Architecture

A new standalone `quant/intraday/rl/` subpackage: an execution-MDP environment, a tabular
Q-learning trainer, a greedy policy + schedule rollout, and an evaluation/CLI. It reuses
`quant/backtest/impact.py` (square-root impact) and `quant/intraday/execution/` (the A-C
`optimal_schedule` and `twap` baselines from Sub-project A) for cost modeling and comparison. It
imports nothing from the live loop and touches no live path.

---

## 2. The execution MDP

- **Episode:** liquidate a parent of `X` shares (sell-to-flat) over `N` discrete steps.
- **State:** `(steps_remaining, inventory_remaining)`, discretized — time `0..N`, inventory
  `0..X` (integer shares; X kept small, e.g. ≤ ~50, so the Q-table is small).
- **Action:** a discrete trade-size level for this step. The action index `0..n_actions-1` maps
  to a child quantity (clamped to the remaining inventory) — e.g. an evenly spaced grid from 0 up
  to a per-step max. Selling more than remaining is clamped, not an error.
- **Reward** (per step, to MINIMIZE cost ⇒ reward is the negative cost): `reward = −impact_cost −
  λ·σ²·(inventory_after)²·dt`. The first term is the square-root market impact of trading `n`
  shares (reusing `market_impact_bps`); the second is the A-C inventory-risk penalty. This is
  EXACTLY the Almgren–Chriss mean-variance objective (impact + risk, no separate spread term), so
  the RL agent and the A-C closed form optimize the identical cost — the comparison is apples-to-
  apples. The mid follows a seeded ABM path, so holding inventory carries real risk.
- **One shared cost function:** `evaluate.py` costs ALL THREE policies (learned, A-C, TWAP) through
  the SAME env cost model on the same seeded paths, so the comparison is parity-clean (it does NOT
  re-cost A-C via Sub-project A's separate sim — only A's *schedule* is reused, then costed here).
- **Terminal:** at step `N`, any remaining inventory is **force-liquidated** and charged its
  impact cost. This deadline is what drives the agent to schedule trades over the horizon.
- **Transition:** deterministic inventory bookkeeping (`inventory -= traded`); the price advances
  one ABM step (stochastic, seeded). `done` when `steps_remaining == 0`.

---

## 3. Components

`quant/intraday/rl/` (new subpackage):

| File | Responsibility |
|---|---|
| `config.py` | `RLConfig`: `total_shares`, `n_steps`, `n_actions`, `alpha` (learning rate), `gamma_discount`, `epsilon_start`/`epsilon_end`, `n_episodes`, `risk_aversion` (λ), `impact_coef_bps`, `adv_dollar`, `sigma`, `dt`, `start_price`, `seed`. Validated; no magic numbers. |
| `env.py` | `ExecutionEnv`: `reset() -> State`, `step(action_index) -> (next_state, reward, done)`. State = `(steps_remaining, inventory_remaining)`. Square-root impact + spread + risk penalty reward; seeded ABM mid; terminal force-liquidation. |
| `qlearning.py` | `train(config) -> TrainResult` (a Q-table `np.ndarray[n_steps+1, total_shares+1, n_actions]` + a training curve = avg episode cost per block). ε-greedy with a decay schedule; standard Q-update `Q[s,a] += α(r + γ·max_a' Q[s',a'] − Q[s,a])`. Deterministic given `config.seed`. |
| `policy.py` | `greedy_action(qtable, state) -> int`; `rollout_schedule(qtable, config) -> list[int]` (greedy child-size schedule, summing to `X`, directly comparable to A's schedules). |
| `evaluate.py` | `compare(config, n_eval_paths) -> dict`: mean execution cost of the learned policy vs the A-C `optimal_schedule` vs `twap`, over `n_eval_paths` seeded evaluation episodes through the same env cost model. |
| (CLI) | `quant intraday rl train` + `quant intraday rl compare`, added to `quant/intraday/cli.py`. |

---

## 4. Training & reproducibility

ε-greedy Q-learning over `n_episodes`; ε decays linearly from `epsilon_start` to `epsilon_end`.
Each episode draws a fresh seeded ABM path (derived deterministically from `config.seed` + episode
index). All randomness flows through a single seeded RNG, so identical config ⇒ identical Q-table
and identical training curve. The training curve (mean episode cost per block of episodes)
demonstrates convergence toward the optimum.

---

## 5. Evaluation (the showcase artifact)

- `quant intraday rl compare --shares X [--episodes E]`: trains the agent, rolls its greedy policy
  out to a child-size schedule, and prints **mean execution cost: learned vs A-C optimal vs TWAP**
  over many seeded evaluation paths — showing the learned policy lands at/near A-C and beats TWAP.
- `quant intraday rl train --shares X`: prints the convergence curve (mean episode cost per block
  falling toward the optimum).
- Both print the honesty note: tabular RL rediscovers the DP/A-C optimum; the point is the agent
  *learns* it from rewards.

Defaults: a small parent (e.g. X≈20–50 shares), short horizon (N≈10), representative σ/impact, so
both commands run quickly without live data.

---

## 6. Charter compliance

- **Reproducibility:** all randomness via a single seeded RNG; identical config ⇒ identical
  Q-table, schedule, and metrics. Config-driven; no magic numbers.
- **No lookahead:** the policy at step t conditions only on the state at t; training never peeks at
  future price moves beyond the realized step reward.
- **No duplicate models:** reuses `quant/backtest/impact.py` for impact and Sub-project A's
  `optimal_schedule`/`twap` for the baselines.
- **Honesty:** spec + CLI state plainly that tabular RL rediscovers the DP/A-C optimum on this
  low-dim MDP; the value is the learning demonstration, not alpha.

---

## 7. Success criteria

- `ExecutionEnv` is a correct MDP: inventory decrements and never goes negative; over-selling is
  clamped; the episode ends at `N` with remaining inventory force-liquidated; reward is the
  negative per-step cost (incl. the λ risk penalty).
- Q-learning is deterministic (same seed ⇒ identical Q-table) and **learns**: the trained greedy
  policy's mean episode cost is strictly lower than a random policy's (the core "it learned"
  property).
- `rollout_schedule` produces a valid schedule (non-negative child sizes summing to `X`).
- In `compare`, the learned policy's mean cost is **≤ TWAP** and **within a stated tolerance of the
  A-C optimum** over the evaluation paths (the headline property).
- `quant intraday rl compare` and `rl train` produce their artifacts on a seeded run.
- No existing test changes; full suite (excluding network/alpaca) stays green; numpy-only (no new
  heavy deps).

---

## 8. Out of scope / deferred

Deep RL (PPO/SAC, neural nets, PyTorch), continuous state/action spaces, function approximation,
live wiring, multi-asset execution, RL market-making (a possible separate sub-step), and any
transfer to real intraday data beyond the seeded sim. Each is its own future spec.
