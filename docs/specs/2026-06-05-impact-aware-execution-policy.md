# Impact-aware live execution policy

**Date:** 2026-06-05
**Status:** Design — implementation on `feat/exec-impact-policy`
**Charter tie-in:** §2 "realistic execution"; closes the audit gap "live execution is naive market orders despite the backtest modeling square-root impact."

## Problem

The live path (`quant/live/rebalance.py`) nets per-strategy intent into one order
per symbol and submits each as a **plain market order** (`OrderTemplate` defaults:
MARKET / DAY / no limit). The backtest, by contrast, charges a size-dependent
square-root impact cost (`quant/backtest/impact.py`) on every fill. So the
backtest believes a large rebalance is expensive, but the live executor ignores
size entirely — it will dump an order of any participation rate into the open as
an unpriced market order. For the defensive-ETF baseline (liquid, small notional)
this is harmless; for a scaled book or a thinly-traded name it is the single
largest uncontrolled real-money risk in the system.

The system is a **daily batch** (one cron fire per session), not an intraday
loop. Classic intraday TWAP/VWAP-over-the-day does not fit: there is no process
alive through the session to work an order. The architecturally-honest execution
lever for a daily rebalancer is therefore **participation control at submission
time**, with oversized orders sliced *across daily sessions* by the existing
target-vs-current reconcile loop (tomorrow's rebalance re-derives the residual
delta and submits the next tranche — no new scheduler required).

## Design

A pure, config-driven policy applied to the **netted** orders, after
`net_orders(...)` and before the submit loop. It never opens or flips a
position; it only **caps quantity** and **optionally re-prices** what reconcile
already decided to trade.

### `ExecutionPolicyConfig` (frozen dataclass, `quant/execution/policy.py`)

| field | default | meaning |
|---|---|---|
| `enabled` | `False` | master switch. **Disabled ⇒ orders pass through untouched, byte-for-byte identical to today.** |
| `max_participation` | `0.10` | cap order notional at this fraction of trailing dollar-ADV; excess deferred to next session |
| `adv_window` | `21` | trailing sessions for dollar-ADV (matches backtest `impact.py`) |
| `marketable_limit_bps` | `None` | if set, high-participation orders become marketable LIMITs at `ref_price·(1 ± bps)` (buy=+, sell=−) capping adverse slippage; `None` ⇒ stay MARKET |
| `marketable_threshold` | `0.05` | participation above which the marketable-limit re-price applies |

All numbers live here — no magic numbers in the rebalance path. The config is
constructed at the call site like `PortfolioRiskLimits()` is for Guard 5.

### Pure functions

```
participation(notional, dollar_adv) -> float | None        # None when ADV unknown
cap_qty_to_participation(qty, ref_price, dollar_adv, cfg) -> (capped_qty, deferred_qty)
marketable_limit_price(side, ref_price, cfg) -> float | None
apply_execution_policy(orders, *, dollar_adv, reference_prices, cfg)
    -> (adjusted_orders: list[OrderTemplate], plan_rows: list[dict])
```

`apply_execution_policy` is the only function the rebalance path calls. It returns
the adjusted order list **plus** a list of plan rows (symbol, original_qty,
capped_qty, deferred_qty, participation, order_type, limit_price, reason) for the
execution-plan artifact.

### Fail-open semantics (charter-mandatory)

- `dollar_adv` missing / non-finite / ≤ 0 for a symbol ⇒ **cannot estimate** ⇒ that
  order is passed through unchanged (today's behavior). Never block a trade
  because ADV is unknown.
- `cfg.enabled is False` ⇒ identity transform; `plan_rows == []`.
- A capped qty of 0 ⇒ the order is **dropped from this session** (not submitted)
  and recorded as fully deferred; reconcile re-proposes it next session. Dropping
  is safe because per-strategy bookkeeping already recorded the *target* (intent),
  so the book is not left inconsistent — the residual is simply carried.

### Data threading

Dollar-ADV per symbol is computed inside the existing per-strategy loop from that
strategy's `bars` (reusing `trailing_dollar_adv` from `quant/backtest/impact.py`,
PIT-correct: strictly-prior rows only) and accumulated into a
`combined_dollar_adv: dict[str, float]` exactly the way `combined_reference_prices`
already is. No extra bar fetches.

### Integration point

```
netted = net_orders(intended)
netted = _apply_execution_policy(netted, combined_dollar_adv,
                                 combined_reference_prices, exec_policy_cfg)
# ... existing Guard-5 portfolio risk gate runs on the post-policy netted ...
# ... existing submit loop unchanged: it already honors OrderTemplate exec fields ...
```

Because the submit loop and `AlpacaClient.submit_order` already honor
`order_type`/`limit_price`/`time_in_force` (PR #13), **no execution-layer code
changes** — the policy only constructs different `OrderTemplate`s. The
execution-plan artifact is written next to the existing portfolio-risk-gate
artifact under `data/live/`.

## Why this is correct, not just safe

- **Backtest/live parity:** live now respects the same ADV/participation notion the
  backtest already charges for, instead of two divergent execution models.
- **No lookahead:** ADV uses strictly-prior bars (the existing impact helper guarantees this).
- **No new failure surface by default:** disabled ⇒ provably identical to today; the
  feature is enabled only after its own evidence (below).
- **Slicing without a scheduler:** participation cap + daily reconcile = multi-session
  TWAP-by-construction, fitting the cron architecture honestly.

## Out of scope (explicitly deferred)

- Intraday working/child-order scheduling (no live intraday loop exists yet).
- Almgren–Chriss optimal *trajectory* (risk-aversion-tuned schedule) — this ships the
  participation/impact control; the trajectory optimizer is a later slice.
- Touching the backtest engine — backtest impact accounting is already correct.

## Test plan (TDD, written first)

1. `enabled=False` ⇒ output orders **identical** (same objects' fields) to input; no plan rows.
2. ADV unknown (0 / NaN / missing) ⇒ order passes through unchanged even when enabled.
3. Order under cap ⇒ unchanged qty, participation recorded.
4. Order over cap ⇒ qty reduced to exactly `floor(max_participation·ADV/ref_price)`,
   deferred remainder = original − capped.
5. Cap reduces to 0 ⇒ order dropped, recorded fully-deferred.
6. `marketable_limit_bps` set + participation ≥ threshold ⇒ LIMIT with correct
   signed price; below threshold ⇒ stays MARKET.
7. Property: capped_qty + deferred_qty == original_qty; capped_qty ≥ 0; sign/side preserved.
8. Integration: a synthetic rebalance with one oversized + one small order produces a
   capped batch + artifact, and with `enabled=False` is byte-for-byte the current path.

## Acceptance

- New module `quant/execution/policy.py` + `tests/execution/test_policy.py`, all green.
- `ruff` + `mypy --strict` clean.
- Full existing suite unchanged/green (default-off ⇒ no behavior change).
- `quant doctor` still passes.
