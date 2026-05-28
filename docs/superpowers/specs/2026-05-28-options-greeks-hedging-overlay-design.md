# Options/Greeks Engine + Protective Hedging Overlay (Pillar 3)

**Date:** 2026-05-28
**Status:** Design — approved direction (brainstormed), pending written-spec review
**Pillar:** 3 of 4 in the autonomous-system vision (regime ✓, sizing ✓, monitoring ✓, **options/hedging** ← this)

## 1. Purpose

Add an analytic options-pricing/Greeks core and build a **protective hedging overlay** on
top of it: estimate the equity book's net beta exposure, construct an index-level (SPY)
options hedge — protective put, collar, or put-spread — and run a point-in-time
returns-overlay backtest that honestly surfaces the **tail-protection vs. cost-drag
tradeoff** of insuring the book.

Like the regime detector (Pillar 1) and the sizing overlay (Pillar 4), this is an
**OBSERVED, comparison-only** signal. It does **not** touch live allocation or place
option orders. It produces a baseline-vs-hedged comparison and logs a `kind="research"`
experiment to the registry. Wiring a hedge into the live book is a deliberate follow-on
spec, gated on its own evidence — same discipline as the prior pillars.

## 2. Why this shape

The other three pillars are overlays/observers on the equity book (regime labels it,
sizing scales gross exposure, the daemon guards it). A protective hedging overlay is the
natural fourth: it reduces left-tail risk on the same book. Pricing hedges **analytically**
(Black-Scholes off cached underlying bars) keeps the backtest fully deterministic and
offline-reproducible — no options-data vendor, no new dependencies (scipy/numpy already
present). The regime-conditional hedge intensity reuses Pillar 1's validated label, so the
pillars compose rather than sit side by side.

## 3. Architecture

New package `quant/options/`, mirroring the `quant/sizing/` layout (pure components → a
composing policy → a PIT returns-overlay backtest → a comparison dataclass → one CLI group):

```
quant/options/
  __init__.py      # public API exports
  pricing.py       # Black-Scholes-Merton price + Greeks + implied-vol solver (pure)
  models.py        # OptionLeg, HedgeStructure, HedgeConfig, HedgeDecision dataclasses
  structures.py    # build protective-put / collar / put-spread leg sets from spot+moneyness
  beta.py          # PIT rolling beta of book returns vs SPY returns
  policy.py        # compose: pick structure, size contracts, scale by regime intensity
  overlay.py       # apply_hedge / compare_hedge PIT returns overlay + tail metrics
```

CLI: one new group `quant hedge` with `price` (engine demo) and `compare` (the overlay).

### 3.1 Pricing core — `pricing.py`

Pure functions, no I/O, no state. Black-Scholes-Merton with continuous dividend yield `q`.

- `bs_price(spot, strike, t_years, vol, r, q, right) -> float` — `right` ∈ {"call","put"}.
- `bs_greeks(spot, strike, t_years, vol, r, q, right) -> Greeks` — delta, gamma, vega,
  theta (per-year), rho. `Greeks` is a frozen dataclass.
- `implied_vol(price, spot, strike, t_years, r, q, right) -> float` — Brent solve
  (`scipy.optimize.brentq`) bracketed on `[1e-4, 5.0]`; returns `nan` if the target price
  is outside the no-arbitrage bounds.

Degradation contract (matches `sizing/components.py`): never raise on finite inputs;
at-expiry (`t_years <= 0`) returns intrinsic value and step-function/zero Greeks;
non-finite or non-positive `vol`/`spot`/`strike` returns `nan` (callers guard).

### 3.2 Models — `models.py`

- `OptionLeg(right: str, strike: float, quantity: float)` — `quantity` signed
  (+long / −short), in index units. Frozen.
- `HedgeStructure(legs: tuple[OptionLeg, ...], spot_at_open: float, expiry_index: int)` —
  a built structure with the spot it was struck against and the bar index it expires on.
  `value(spot, t_years, vol, r, q)` sums signed-leg BS values (pure method delegating to
  `pricing`).
- `HedgeConfig` — all knobs, frozen, intentional defaults:
  - `structure: str = "put"` ∈ {"put", "collar", "put_spread"}
  - `put_moneyness: float = 0.05` (5% OTM put: strike = spot·(1−0.05))
  - `call_moneyness: float = 0.05` (collar financing call, OTM above spot)
  - `spread_width: float = 0.10` (put-spread lower leg this much further OTM)
  - `coverage: float = 1.0` (fraction of net beta-dollar exposure to hedge)
  - `tenor_days: int = 30` (calendar days to expiry at each roll)
  - `roll_days: int = 21` (trading days between rolls)
  - `vol_lookback_days: int = 21`, `risk_free: float = 0.03`, `div_yield: float = 0.015`
  - `beta_lookback_days: int = 63`
  - `use_regime: bool = True`,
    `regime_intensity: Mapping[str,float] = {"calm-bull":0.25,"choppy":0.6,"crisis":1.0}`
- `HedgeDecision` — one roll's record: structure, contracts, premium, net beta, regime
  label, regime intensity applied. Frozen, for introspection/serialization.

### 3.3 Structures — `structures.py`

Pure builders mapping `(spot, config)` → `HedgeStructure` legs (quantity=1 unit; the
policy scales by contract count):

- `protective_put`: `[+1 put @ spot·(1−put_moneyness)]`
- `collar`: `[+1 put @ spot·(1−put_moneyness), −1 call @ spot·(1+call_moneyness)]`
- `put_spread`: `[+1 put @ spot·(1−put_moneyness), −1 put @ spot·(1−put_moneyness−spread_width)]`

### 3.4 Beta — `beta.py`

`rolling_beta(book_returns: np.ndarray, spy_returns: np.ndarray) -> float` — OLS slope of
book on SPY over the supplied (already-trailing, PIT) window. `cov/var`; returns `1.0`
(neutral) on degenerate input (`var<=0`, `<2` points, non-finite). Long-book betas are
typically ~0.8–1.1; clamped to `[0, 3]` defensively.

### 3.5 Policy — `policy.py`

`build_hedge(spot, book_returns_hist, spy_returns_hist, regime_label, config, expiry_index)
-> HedgeDecision`:

1. `beta = rolling_beta(book_hist[-beta_lookback:], spy_hist[-beta_lookback:])`
2. `intensity = regime_intensity[label]` if `use_regime` else `1.0` (unknown label → 1.0)
3. `structure = structures.<config.structure>(spot, config)`
4. `contracts = coverage · intensity · beta · book_value / spot`
   (book_value carried by the overlay; index-unit notional = contracts·spot)
5. premium = `contracts · structure.value(spot, tenor_years, vol, r, q)` priced at open.

All PIT: every input is strictly trailing or as-of yesterday.

### 3.6 Overlay — `overlay.py`

The heart, mirroring `sizing/backtest.py`'s `apply_sizing`/`compare_sizing`:

`apply_hedge(returns, spy_close, config, regime_labels=None) -> (hedged_returns, HedgeLedger)`

Walk the daily index. Maintain `baseline_equity` (compounds `returns`) and a hedge
position. Mechanics:

- **Open/roll:** every `roll_days` (and at t=0), close any open structure at its current
  value, then build a new one via `policy.build_hedge` using **only data through t−1** for
  vol/beta and the regime label as-of t−1. Strikes are set from **today's** spot
  `spy_close[t]` (the price you transact at); the structure expires `tenor_days` calendar
  days out (mapped to the nearest trading-bar index).
- **Daily mark:** reprice the held structure at `spy_close[t]` with shrinking
  time-to-expiry and trailing vol. Hedge P&L for the day = `contracts · (V_t − V_{t−1})`.
  At a roll, the realized close P&L plus the new premium paid are folded in.
- **Equity:** `hedged_equity_t = baseline_equity_t + cumulative_hedge_pnl_t` (same unit
  base, book starts at 1.0). `hedged_returns_t = hedged_equity_t / hedged_equity_{t−1} − 1`.

The long-put theta decay is the visible cost drag; the convex put payoff in a crash is the
visible benefit; a collar's short call caps upside (negative P&L in rallies) but lowers
premium — all emerge naturally from signed-leg BS repricing. No special-casing.

`compare_hedge(returns, spy_close, config, regime_labels=None) -> HedgeComparison`:
computes baseline and hedged metric dicts (the standard 7 from `sizing` **plus tail
metrics** `cvar_5` and `worst_day`), plus a hedge summary (total premium spent, premium as
annualized % drag, number of rolls, mean contracts, structure used). `HedgeComparison` is a
frozen dataclass.

Tail metrics added to `overlay.py` (small, pure): `cvar(returns, alpha=0.05)` (mean of the
worst `alpha` tail of daily returns) and `worst_day(returns)`.

### 3.7 PIT guarantee

Same truncation-invariance proof as sizing: a hedge decision/mark at day `t` depends only
on `returns[:t]`, `spy_close[:t+1]` (today's spot is transactable), and the regime label
as-of `t−1`. A property test asserts that truncating the inputs at `T` reproduces the first
`T` hedged returns **bit-for-bit** (`atol=0`).

## 4. CLI — `quant hedge` group

- `quant hedge price --spot 500 --strike 480 --days 30 --vol 0.2 [--right put] [--rate .03]
  [--div .015]` — prints price + all five Greeks + (if `--mark PRICE`) implied vol. A thin
  demo of the engine; no registry write.
- `quant hedge compare <strategy> [--start --end] [--structure put|collar|put_spread]
  [--put-moneyness --call-moneyness --spread-width --coverage --tenor-days --roll-days]
  [--no-regime]` — runs the strategy's default-param backtest (reusing
  `_run_single_backtest`), loads cached SPY closes over the same window, loads the regime
  series (reusing `_load_regime_labels`), runs `compare_hedge`, prints a baseline-vs-hedged
  Rich table (incl. CVaR / worst-day / max-dd) + a hedge-cost summary line, and appends a
  `kind="research"` `ExperimentRecord` (run_id `hedge-…`) with gates:
  - `gate_maxdd_improved`: hedged max_drawdown ≥ baseline (less negative)
  - `gate_cvar_improved`: hedged cvar_5 ≥ baseline (tail less negative)

  Both gates are **honest tradeoff reporting**, not pass/fail blockers — hedging is
  *expected* to cost Sharpe/CAGR in calm markets, exactly as the sizing overlay honestly
  showed it hurt trend's Sharpe. The point is to quantify the insurance premium.

## 5. Testing strategy (TDD)

- **`pricing.py`:** known-value checks (textbook BS examples), **put-call parity**
  (`C − P == spot·e^{−qT} − strike·e^{−rT}`), Greeks vs **central finite-difference** of
  `bs_price` (delta/gamma/vega/theta/rho), implied-vol round-trip (`iv(price(σ)) == σ`),
  at-expiry intrinsic, and degradation on non-finite inputs.
- **`structures.py`:** correct strikes/signs/quantities per structure; collar net premium
  < bare put premium; put-spread cheaper than bare put.
- **`beta.py`:** recovers a known slope on synthetic `y = β·x + noise`; neutral on
  degenerate input.
- **`policy.py`:** regime intensity scales contracts monotonically (crisis > choppy >
  calm-bull); `use_regime=False` → intensity 1.0; PIT (no peek at `t`).
- **`overlay.py`:** truncation-invariance property test (Hypothesis, `atol=0`); a
  hand-constructed crash scenario where the protective put **reduces** max drawdown; a calm
  uptrend where the hedge **drags** CAGR (cost is real); CVaR/worst-day correctness on
  known arrays; collar caps upside vs bare put in a rally.
- **CLI:** `hedge price` prints finite Greeks; `hedge compare` on a tiny fixture writes a
  well-formed registry record with both gates present.

Target: full suite stays green (currently 502), `mypy --strict` clean, `ruff` +
`ruff format --check` clean.

## 6. Explicitly out of scope (deferred follow-ons)

1. **Live hedge recommendation from Alpaca option snapshots** — a `quant hedge recommend`
   that reads the live paper book, estimates beta, and prices the structure against
   *live* SPY option chains/IV/Greeks via Alpaca. Deferred to keep this slice deterministic
   and offline-testable; the analytic core is the foundation it would build on.
2. **Wiring a hedge into the live book** — actually buying protective structures in the
   paper account. Requires its own evidence gate and governance integration, exactly like
   the regime/sizing live-wiring follow-ons.
3. **Vol-surface skew** — pricing puts at a skew premium over realized vol so backtest
   hedge costs aren't unrealistically cheap. A realism refinement on the analytic core.
4. **American exercise / early-exercise premium** — SPY options are American; we price
   European (Merton). Immaterial for short-tenor index puts but noted.
5. **Per-name hedging** — rejected in brainstorming in favor of the index beta hedge.

## 7. Success criteria

- `quant hedge price` returns Greeks matching finite-difference to tight tolerance.
- `quant hedge compare trend` (and other strategies) runs end-to-end, prints the
  baseline-vs-hedged table + cost summary, and logs a research experiment.
- Over a window containing a sharp drawdown, the protective-put overlay measurably reduces
  max drawdown and CVaR; over a calm window it measurably drags CAGR — the tradeoff is
  surfaced honestly, not hidden.
- Nothing in this pillar touches live allocation or places orders (observed-only).
- Suite green, mypy strict clean, ruff clean.
