# Portfolio leverage — explicit, bounded, gated deployment knob

**Date:** 2026-06-11
**Status:** built; actuated at **1.0x** (deploy idle cash, no leverage)

## Why

The live book (trend + defensive-etf) ran at ~0.8x gross / ~8.5% vol against a
−25% drawdown budget — ~⅓ of its risk capacity, needlessly parking ~20% in cash.
The validated edge is real (both strategies passed the honest gates), so deploying
more of it is legitimate "more return from the edge we have" — a risk-appetite
choice, not new alpha. Three free-data alpha screens (pre-FOMC drift, macro
credit-rotation, cross-sectional ETF momentum) all came back NULL first, which is
why scaling the existing edge is the honest lever.

## Leverage scaling (combined book, 2020–2025, 5% borrow cost on the levered part)

| Leverage | CAGR | maxDD (backtest) | Sharpe | GFC-stress worst-loss |
|----------|------|------------------|--------|-----------------------|
| 0.80 (was) | +14.0% | −10.5% | 1.32 | ~37% |
| **1.00** | **+17.6%** | −13.1% | **1.32** | **~46% (gate GREEN)** |
| 1.25 | +20.6% | −16.5% | 1.25 | 57.6% (WARN) |
| 1.50 | +23.6% | −20.7% | 1.19 | **69.1% (WARN)** |

The Sharpe holds 0.8x→1.0x (only borrow cost degrades it above), so scaling is
genuinely the same edge — but **the backtest maxDD hides the tail**: the 2020–2025
sample has no 2008-style systemic crash, while the pre-trade stress test does. At
1.5x a GFC-scenario worst-loss is ~69% (risk gate WARNs > 55%); only **≤~1.2x stays
inside the stress budget**. Decision: **1.0x** — meaningful uplift, Sharpe
unchanged, stress gate green, no catastrophic-crash tail.

## Implementation (default-OFF, byte-identical)

- `run_rebalance(target_leverage=)` + CLI `quant rebalance --leverage X`. `None` =
  today's behaviour exactly. Hard-capped at **2.0x**. Deploys the NORMALIZED
  allocation at the target gross.
- Composes one-way with the de-risk overlay (`--leverage 1.0 --derisk-actuate` =
  1.0x base, cut to 0.75x on today's risk-off day). The pre-trade Guard-5 sees the
  levered orders and fails closed if they breach the hard envelope (the 1.5x case
  only WARNed, did not block).
- Reversible: a later rebalance at a different `--leverage` restores any level.

## Honesty notes

- This raises *returns AND risk* proportionally (same Sharpe) — not alpha.
- Forward returns will be below the favorable-sample backtest Sharpe (1.32).
- The 1.0x choice was a deliberate down-shift from an initial 1.5x once the
  stress test surfaced the ~69% GFC tail the backtest maxDD concealed.
