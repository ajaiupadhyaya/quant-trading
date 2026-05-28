# Borrow & financing costs — design spec

**Date:** 2026-05-28
**Charter gap:** #2 (realistic execution: borrow + market-impact), slice 2a of 3.
**Status:** approved (architecture + mechanics), pending implementation plan.

## Motivation

`docs/CHARTER.md` principle 2 requires modeling transaction costs, slippage,
market impact, **and borrow/financing costs** — "a strategy that ignores these
is not a result." The backtest engine currently models only flat per-fill
`slippage_bps` + `commission_bps`. It charges **nothing** to hold a short or to
run levered, yet several strategies (pairs, multi-factor, cross-sectional
momentum, risk-parity) hold shorts. This slice closes the borrow/financing
omission.

Market impact (slice 2b) and capacity (slice 2c) are separate follow-on specs;
2a is independent of both.

## Scope

Charge, as daily costs deducted from cash:

1. **Short borrow fee** — on short-position notional, at an annual borrow rate.
2. **Margin-debit financing** — on a negative cash balance (leverage), at an
   annual financing rate.

**No interest credits** (no short rebate, no idle-cash interest) — costs only,
never optimistic, per the charter's "flag results that look too good."

Financing is **on by default** with sensible rates (this is a model change that
*should* shift results; existing governance/validation evidence is refreshed by
re-running `quant validate` — expected, not a regression).

Accrual is **calendar-day, actual/365**: each step bills `rate × (calendar days
since prior bar)/365`, so a Fri→Mon step bills 3 days and holiday gaps are
handled.

## Architecture

### New unit — `quant/backtest/financing.py`

Pure, isolated (keeps the large `engine.py` loop thin), mirroring the role
`activity.py` plays for turnover.

```python
@dataclass(frozen=True)
class FinancingCharge:
    borrow_cost: float
    margin_financing_cost: float

    @property
    def total(self) -> float:
        return self.borrow_cost + self.margin_financing_cost


def financing_charge(
    positions: Mapping[str, int],
    prior_close: Mapping[str, float],
    cash: float,
    days_elapsed: int,
    annual_borrow_bps: float,
    annual_financing_bps: float,
) -> FinancingCharge: ...
```

It takes the two rates as plain floats (not the `BacktestConfig` object) so it
stays fully standalone and testable without constructing a config — and, since
`engine.py` imports `financing_charge`, this avoids a circular import on
`BacktestConfig`.

Logic:
- `short_notional = Σ |qty| · prior_close[sym]` over positions with `qty < 0`.
  A symbol missing from `prior_close`, or with a non-finite price, contributes
  0 (defensive — mirrors the engine's tolerance for sparse bars).
- `borrow_cost = short_notional · (annual_borrow_bps / 1e4) · days_elapsed / 365`
- `margin_debit = max(0.0, -cash)`
- `margin_financing_cost = margin_debit · (annual_financing_bps / 1e4) · days_elapsed / 365`
- Degenerate inputs (no shorts, non-negative cash, `days_elapsed ≤ 0`, empty
  positions) yield `0.0` components; the function never raises.

### Config — `BacktestConfig` (engine.py)

Add two fields (defaults active because financing is on by default):

```python
    annual_borrow_bps: float = 50.0       # short borrow, general-collateral, liquid universe
    annual_financing_bps: float = 200.0   # margin-debit rate — flat approximation; only bites under >1x gross
```

`annual_financing_bps` is a deliberate flat approximation (the true broker call
rate is regime-dependent); it is documented as such and only affects levered
books. A rate-curve-linked financing rate and per-symbol borrow rates are
explicit non-goals here (see Out of scope).

### Engine wiring — `run_backtest` (engine.py)

At the **top of each bar after the first**, before executing today's fills:

1. `days_elapsed = (ts - prev_ts).days` (calendar days).
2. Build `prior_close = {sym: bars[(sym, "close")].loc[prev_ts]}` for held
   symbols — strictly the **prior bar's** close, so the charge uses only
   information available before today (no lookahead).
3. `charge = financing_charge(positions, prior_close, cash, days_elapsed,
   config.annual_borrow_bps, config.annual_financing_bps)` on the positions/cash
   **carried overnight from the prior bar** (before any of today's fills mutate
   them).
4. `cash -= charge.total`; accumulate `borrow_total += charge.borrow_cost` and
   `financing_total += charge.margin_financing_cost`.
5. Track `prev_ts = ts` at the end of each iteration.

The first bar has no prior period → no charge. The existing fill / MTM /
rebalance steps are unchanged; financing is purely an additional cash
deduction, so it flows naturally into the existing daily equity mark.

### Reporting

- `BacktestResult.metadata` gains `borrow_cost`, `margin_financing_cost`, and
  `financing_cost_total` (cumulative dollars). `metadata` already exists on the
  dataclass and is currently unpopulated by `run_backtest`.
- The cost is already reflected in `equity_curve` / `returns` / every derived
  metric (it is deducted from cash), so no metric needs to change to *account*
  for it.
- Explicit line-item: the `quant backtest` combined-book CLI table gains a
  "Financing $" column. Per-strategy rows read
  `sub.metadata["financing_cost_total"]`; the COMBINED row sums those across
  `result.per_strategy` (so `CombinedResult` needs **no** new field —
  per-strategy `BacktestResult`s are in hand there).

### Combined book — `combined.py`

`run_combined_book` already threads `slippage_bps`/`commission_bps` into each
sub-`BacktestConfig`; add the two new fields the same way so borrow/financing
flow to every sub-portfolio. No `CombinedResult` change is needed — the COMBINED
financing total is summed from `per_strategy[*].metadata` at the CLI reporting
site.

## Testing

**`tests/backtest/test_financing.py` (pure function):**
- Single short, `days_elapsed=1`: exact fee = `notional · bps/1e4 · 1/365`.
- Long-only with positive cash → `borrow_cost == 0` and `margin_financing_cost == 0`.
- Weekend gap (`days_elapsed=3`) → exactly 3× the one-day borrow.
- Negative cash (leverage) → `margin_financing_cost` on `|cash|`; positive cash → 0.
- `days_elapsed == 0` (and `< 0`) → all components 0.
- Combined short + margin debit → `total` equals the sum of the two.
- Missing/non-finite `prior_close` for a held short → that symbol contributes 0.

**`tests/backtest/test_engine_financing.py` (integration):**
- A strategy holding a fixed short for N trading days: equity under
  `annual_borrow_bps=0` minus equity under `=50` equals the expected cumulative
  calendar-day borrow (within float tolerance), and
  `metadata["financing_cost_total"]` matches.
- PIT / no-lookahead: a large price move on day *t* does not change day *t*'s
  financing charge (the charge depends only on `prev_ts` close + carried
  positions). Assert by comparing charges with today's bar perturbed.
- A long-only, unlevered run has `financing_cost_total == 0` and is byte-for-byte
  unchanged vs the pre-feature equity curve when rates are set to 0.

## How it could fail

- **Default-rate realism:** 50 bps borrow is right for a liquid large-cap/ETF
  universe but far too low for hard-to-borrow names; `200 bps` financing is a
  flat stand-in for a regime-varying rate. Both are documented approximations
  and exposed in config; per-symbol borrow and a rate curve are deferred.
- **Calendar gaps at data boundaries:** a long stale-data gap (e.g. a missing
  month) would bill borrow across the whole gap. Acceptable — it reflects a real
  holding cost — but worth noting for sparse histories.
- **Double-counting with leverage:** a short *and* a margin debit are distinct
  charges (you pay to borrow the stock and to borrow the cash); the model
  charges both intentionally. For a cash-neutral L/S book the debit is ~0, so
  only borrow applies — the intended behavior.

## Out of scope (this slice)

- Market-impact model (slice 2b) and capacity (slice 2c).
- Per-symbol / hard-to-borrow borrow rates; rate-curve-linked financing.
- Interest credits (short rebate, idle-cash interest).
- Sweeping borrow/financing in the cost-sensitivity validation sweep (it varies
  `slippage_bps` only; financing stays at default).
- A financing line in the walk-forward tear-sheet's multi-window aggregation
  (the cost already shows in every tear-sheet metric).
