# Intraday Market Making (Avellaneda–Stoikov) — Design Spec

**Date:** 2026-06-08
**Status:** Approved design, pending implementation plan
**Scope:** Sub-project B of the intraday/60s showcase track. Independent of the spine (0) and optimal-execution (A); shares no live path.

---

## Context

The intraday showcase track has shipped Sub-project 0 (the live 60s sleeve loop) and
Sub-project A (Almgren–Chriss optimal execution). This sub-project adds a **market-making**
showcase based on the Avellaneda–Stoikov (2008) model.

**Goal = portfolio/learning showcase** (see [[project-quant-trading-intraday]]). Decisions
locked in brainstorming:
1. **Sim / research only** — NO live wiring. No change to the spine, Sub-project A, or the
   daily system. (Live quoting is deferred to a possible later sub-step; real market making
   lives on sub-second queue dynamics a 60s loop cannot represent, and live quoting would
   need limit-order + cancel/replace machinery the loop does not have.)
2. **A-S fill-intensity model** `λ(δ) = A·exp(−k·δ)` — the canonical A-S setup; the only way
   to demonstrate the spread/inventory/fill tradeoff. A standalone simulator, deliberately
   NOT the trade-through `BacktestEngine` (intensity fills are a different paradigm).

Honest framing (recorded here and surfaced in CLI output): this is a **stylized** A-S model.
The intensity parameters `A` and `k` are assumptions, not fit to live fills; the value is the
technique and the spread-capture-vs-inventory-risk analysis, not a live edge.

---

## 1. Architecture

A new subpackage `quant/intraday/marketmaking/` holding the A-S quoting math and fill-intensity
model as PURE functions, a standalone deterministic simulator, and a CLI. It does not import or
modify the live loop, the execution engine, or the order-execution sim. It MAY reuse the data
layer to obtain a real mid-price series, but its default path is a seeded ABM so tests are
self-contained.

---

## 2. Components

`quant/intraday/marketmaking/` (new subpackage):

| File | Responsibility |
|---|---|
| `config.py` | `MMConfig`: `gamma` (risk aversion), `k` and `A` (intensity), `horizon_seconds` (T), `dt_seconds` (step), `sigma` (vol, price units/√time), `lot_size`, `seed`. Validated; no magic numbers. |
| `avellaneda_stoikov.py` | Pure quoting math: `reservation_price(mid, inventory, gamma, sigma, t_remaining)`, `optimal_spread(gamma, sigma, t_remaining, k)`, `quotes(mid, inventory, gamma, sigma, t_remaining, k) -> (bid, ask)`. |
| `intensity.py` | `fill_intensity(delta, A, k) -> float` (= A·exp(−k·δ)); `fill_probability(delta, A, k, dt) -> float` (= 1 − exp(−λ·dt), clamped to [0,1]); `draws_fill(prob, rng) -> bool` (seeded Bernoulli). |
| `price_path.py` | `abm_path(s0, sigma, dt, n_steps, rng) -> list[float]` (seeded **arithmetic** Brownian motion mid path: `s_{t+1}=s_t+σ√dt·z`). A-S uses ABSOLUTE volatility (price units), so the faithful process is arithmetic, NOT geometric — this keeps the σ² terms in the quoting math dimensionally consistent. Returns a plain price list so a real replayed mid can substitute. |
| `simulator.py` | `MMResult` dataclass + `run_market_making(prices, config) -> MMResult`: steps the path, computes A-S quotes each step, draws bid/ask fills via the intensity model, updates inventory + cash, marks P&L. Deterministic given `config.seed`. |
| `evaluate.py` | `gamma_sweep(prices, config, gammas) -> list[SweepPoint]` — runs the sim across a γ grid; each point reports P&L, # fills, mean/max |inventory|, terminal inventory. |
| (CLI) | `quant intraday mm simulate` + the γ-sweep, added to `quant/intraday/cli.py`. |

---

## 3. The Avellaneda–Stoikov model (the math)

For a mid price `s`, inventory `q` (signed, lot units), risk aversion `γ`, volatility `σ`, and
time remaining `τ = T − t`:

- **Reservation price** (inventory-skewed): `r = s − q·γ·σ²·τ`. Long inventory (q>0) ⇒ r < s
  (quotes shifted down, keener to sell); short ⇒ r > s.
- **Optimal total spread:** `δ = γ·σ²·τ + (2/γ)·ln(1 + γ/k)`.
- **Quotes:** `bid = r − δ/2`, `ask = r + δ/2`. (Skew comes from r; the spread is symmetric
  about r. Note bid/ask are about the *reservation* price, so they are asymmetric about the
  *mid* when inventory ≠ 0 — the inventory-management mechanism.)
- **Quote distances from mid (for intensity):** `δ_bid = s − bid`, `δ_ask = ask − s`.
- **Fill intensity:** `λ(δ) = A·exp(−k·δ)`; fill probability over `dt` is
  `p = 1 − exp(−λ·dt)`, clamped to [0,1].

**Test anchors:** `∂δ/∂σ > 0`, `∂δ/∂τ > 0` (spread widens with vol and horizon). NOTE: the
spread is **NOT** monotonic in γ — the `(2/γ)ln(1+γ/k)` term *decreases* with γ and can dominate
`γσ²τ` at small γ, so δ can fall as γ rises. The inventory-control behavior instead comes from
the **reservation-price skew** `q·γ·σ²·τ`, which IS linear in γ (higher γ ⇒ stronger pull back
toward flat). Reservation skew sign tracks inventory; as `τ → 0` the inventory term vanishes.

---

## 4. The simulator (`run_market_making`)

Steps `n = horizon/dt` intervals over a mid-price path:

1. `τ = T − t` for the current step.
2. Compute `(bid, ask)` from `quotes(...)` using current inventory.
3. Compute `δ_bid, δ_ask` from mid; convert to fill probabilities via `intensity`.
4. Draw a bid fill and an ask fill independently (seeded RNG). On a **bid fill**: inventory
   `+= lot`, cash `−= bid·lot`. On an **ask fill**: inventory `−= lot`, cash `+= ask·lot`.
5. Advance.

`MMResult`: `final_pnl` (= cash + inventory·last_mid), `n_bid_fills`, `n_ask_fills`,
`inventory_path: list[int]`, `mean_abs_inventory`, `max_abs_inventory`, `terminal_inventory`,
`spread_captured` — defined concretely as the sum over every fill of the half-spread edge
captured versus the contemporaneous mid: `Σ |quote_price − mid_at_fill| · lot` (each fill earns
the distance of its quote from mid; this is the gross edge the MM extracts, before inventory
P&L). Fully determined by `(prices, config)` incl. `config.seed`.

**No lookahead:** step t uses only `prices[t]` and inventory through t.

---

## 5. Evaluation (the showcase artifact)

- `gamma_sweep` runs the sim across a γ grid on the same price path + seed, producing the A-S
  analog of A's efficient frontier: **low γ ⇒ tight spread, many fills, higher inventory risk;
  high γ ⇒ wide spread, fewer fills, controlled inventory**, with P&L for each.
- CLI `quant intraday mm simulate --symbol QQQ [--gamma G] [--seed S] [--steps N]`: runs one
  A-S simulation and prints final P&L, spread captured, fill counts, and inventory stats.
- CLI `quant intraday mm sweep --symbol QQQ`: prints the γ-sweep table (the headline artifact),
  with a one-line note that the model is stylized (A, k are assumptions).

Both default to a seeded ABM path anchored to a representative mega-liquid-ETF price/σ so they
run without live data; `--real` (optional) may replay a real intraday mid from the data layer.

---

## 6. Charter compliance

- **Reproducibility:** all randomness via a seeded `random.Random(config.seed)`; same inputs ⇒
  identical `MMResult`. Config-driven; no magic numbers.
- **No lookahead:** quotes at step t use only information through t.
- **Honesty:** the simulator is a clearly-labelled stylized A-S model; the CLI states that A/k
  are assumed parameters, not a live edge. Consistent with the spec's framing and the Charter's
  "flag results that look too good."
- **No overfitting:** closed-form quoting model with interpretable knobs (γ, k, A); nothing fit.

---

## 7. Success criteria

- The quoting math reproduces the A-S properties (spread monotone ↑ in σ and τ — NOT
  necessarily in γ, see §3; reservation skew sign tracks inventory; quotes symmetric about r).
- The intensity model returns valid probabilities (∈[0,1]; closer ⇒ higher; δ→∞ ⇒ 0).
- The simulator is deterministic (same seed ⇒ identical result) and conserves P&L
  (cash + inventory·mid).
- Higher γ produces lower max |inventory| (the core inventory-control behavior) on the same path.
- `quant intraday mm sweep` produces the γ-tradeoff table on a seeded path.
- No existing test changes; full suite (excluding network/alpaca) stays green.

---

## 8. Out of scope / deferred

Live quoting (limit-order posting + cancel/replace on the sleeve), multi-asset / correlated
inventory, order-book / queue-position modeling, adverse-selection and latency modeling,
calibration of A/k from real fills, and the closed-form finite-horizon HJB solution beyond the
standard A-S approximation. Each is a possible later sub-step with its own spec.
