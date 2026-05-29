# Per-Symbol Order Netting Implementation Plan

> REQUIRED SUB-SKILL: superpowers:subagent-driven-development. TDD, checkbox steps.

**Goal:** Net all intended orders per symbol before submission so opposing
live-vs-orphan (and any cross-strategy) orders never hit Alpaca.

Spec: docs/superpowers/specs/2026-05-29-order-netting-design.md

## Task 1 — pure `net_orders` (quant/execution/netting.py) + tests/execution/test_netting.py
Collapse list[OrderTemplate] -> one net OrderTemplate per symbol (BUY=+qty, SELL=-qty;
net>0 BUY, net<0 SELL, net==0 drop). strategy_slug = largest-|qty| contributor
(ties alphabetical). Sorted by symbol (deterministic). Tests: opposing pair nets/zeros,
multi-strategy signed sum, passthrough non-overlap, attribution rule, empty->[].

## Task 2 — refactor run_rebalance to collect -> net -> submit (quant/live/rebalance.py)
- Live loop: append reconcile orders to `intended: list[OrderTemplate]`; KEEP snapshot
  write (target, live-only) + outcome; REMOVE inline submit loop.
- Orphan loop: append winddown_orders orders to `intended`; KEEP remaining-snapshot write
  (live-only) + WindDownOutcome; REMOVE inline submit loop. (winddown_orders helper unchanged.)
- After both loops, before append_trades: `from quant.execution.netting import net_orders`;
  `for order in net_orders(intended): submit (dry_run passthrough); append net trade row` on success.
- Integration tests: live BUY + orphan SELL same symbol -> exactly ONE net order to the stub;
  existing wind-down convergence + dry-run tests adapted to netted submission; partial net-fail
  leaves snapshot optimistic (recon halts next run — fail-safe).

## Task 3 — full verification + live dry-run + one real run to clear the DBC excess.
