# Deferred follow-ups — 2026-05-26 session

This session shipped the bars NaN fix, recon-interpretation correction,
and the evidence-gated paper trading governance layer. The items below
were either out of scope, surfaced as side-effects, or are downstream of
work that needs more data first. None are blocking; each has enough
context here to pick up cold.

## 1. True execution-cost recon metric

**Why deferred:** The current `quant/live/recon.py` measures
`(fill_price − signal_price)` which conflates signal-to-fill price drift
with actual execution cost. The cost-model recalibration the user asked
for cannot run on this metric — see
`docs/live-recon/cost-model-interpretation.md`.

**What to build:** An `execution_cost_bps` column on
`data/live/trades.parquet` populated by joining each fill against the
**fill-time mid** (NBBO midpoint, or 1-minute bar mid at the fill
timestamp). Requires an intraday data source — Alpaca minute bars are
already accessible via the existing `AlpacaClient`; an `alpaca-py`
`StockBarsRequest` with `TimeFrame.Minute` is the minimal path.

**Acceptance:** ≥30 fills per strategy with both `signal_drift_bps`
(existing signed delta) and `execution_cost_bps` (one-way unsigned
cost). At that point a cost-model bump from 5 bps to whatever
execution_cost_bps shows is data-driven, not anecdotal.

## 2. Finnhub earnings-calendar refinement on pairs OU half-life filter

**Why deferred:** Explicitly out-of-scope in the 2026-05-25 SOTA push.
The OU half-life filter rejects pairs whose mean-reversion takes too
long; an earnings event on either leg can decorrelate the pair
mid-trade. A Finnhub earnings-calendar gate would let pairs trading
sidestep names with earnings inside the next half-life window.

**What to build:** `quant/data/earnings.py` wrapper around Finnhub
`/calendar/earnings` (free tier is sufficient for this universe).
Cache to `data/earnings/<symbol>.parquet`. In
`quant/strategies/pairs_trading.py`, in the pair-selection step,
drop any pair where `min(half_life_legA, half_life_legB) > days_to_earnings`
for either leg.

**Acceptance:** Pairs strategy backtest with the gate enabled does not
regress Sharpe vs the baseline (decorrelation events should be
neutral-to-positive net of forgone trades).

## 3. Frozen tear-sheet PDF diff harness

**Why deferred:** Out-of-scope in 2026-05-25 push. The HTML tear-sheets
already exist; what's missing is a way to detect *unintended* visual
regressions when validation logic changes.

**What to build:** A `scripts/tearsheet_diff.py` that renders each
strategy's tear-sheet to a PDF (via Playwright or `weasyprint`), stores
a baseline under `tests/baselines/tearsheets/`, and a pytest test that
fails when the rendered PDF byte-hash drifts. Update-baseline command
behind an explicit flag.

**Acceptance:** Running validation with no logic change produces
byte-identical PDFs; a deliberate tweak to a tear-sheet section flips
the test red.

## 4. Per-strategy cost-model overrides

**Why deferred:** Premature without item 1's execution-cost metric.
Currently `BacktestConfig.slippage_bps=5.0` is global. Once per-strategy
execution-cost data exists, less-liquid strategies (e.g. pairs trading
on small-caps) might justify a higher modeled cost than ETF-only
strategies (trend, risk-parity).

**What to build:** A new optional `slippage_bps` key inside
`data/backtests/<slug>/chosen_params.json["latest"]` that, when
present, overrides `BacktestConfig.slippage_bps` during
`run_walkforward` and `run_validation` for that slug. Default of 5 bps
remains for backwards compatibility.

**Acceptance:** Pairs strategy can be validated with `slippage_bps=15`
and the cost-sensitivity sweep still runs against the override as the
"base" config.

## 5. Governance v2 — capital allocation by evidence strength

From the governance spec's "Open Follow-Up" section:

- Strategy-level capital allocation weighted by evidence strength
  (e.g. DSR-weighted vs equal split).
- Paper-P&L drift monitor: alert when realized strategy P&L diverges
  from backtest expectations by >2σ over a rolling window.
- Scheduled automatic validation refresh (weekly cron) so freshness
  doesn't depend on operator memory.
- Separate promotion workflow for real-money eligibility (stricter
  gates than paper-eligibility).

All blocked on the governance v1 layer being deployed and accumulating
a few weeks of paper P&L for drift baselines.

## 6. Trend strategy bootstrap-gate regression investigation

**Surfaced this session:** Re-running `quant validate trend` on
2026-05-26 (immediately after the bars fix) gave
`bootstrap_total_return_p05 = -4.24%` vs the prior `+12.3%`. Holdout
still passes at +21%, but the OOS-bootstrap regression is unexplained.

**To investigate:** (a) was the prior `+12.3%` run on a different
walk-forward window? (b) Is the regime count change (3→4 tested) a
seed/data effect or a real regime-definition change? (c) Does the
bootstrap settle if `bootstrap_resamples` is bumped from 1000 to 5000?

Until investigated, governance will quarantine trend on this run.
That's fail-closed-correct behavior, but the underlying signal warrants
a look before the next code change to trend.
