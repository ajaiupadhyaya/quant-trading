# Go-Live Decisions Log — 2026-05-25 → 2026-05-26

Per-strategy decisions during the all-strategies-live tuning pass and the
2026-05-26 paper-trading deployment.

**Validation gate thresholds (per `quant/backtest/validation.py`):**
- DSR ≥ 0.30
- PSR ≥ 0.70
- Block-bootstrap lower-5% ≥ 0%
- ≥ 50% of TESTED regimes positive (regimes with < 30 days OOS data don't count as "tested")
- Holdout total return ≥ 0%

---

## Baseline (pre-tuning)

| Strategy     | DSR    | PSR    | Bootstrap-5% | Regimes pos | Holdout    | Gates | Status        |
| ------------ | ------ | ------ | ------------ | ----------- | ---------- | ----- | ------------- |
| trend        | 0.54   | 0.99   | +12.3%       | 3/3         | +20.2%     | 5/5   | ENABLED LIVE  |
| momentum     | 0.836  | 0.991  | +8.79%       | 1/3         | +18.19%    | 4/5   | disabled      |
| multi-factor | (~+)   | (~+)   | (+)          | 1/3         | +11%       | 4/5   | disabled      |
| risk-parity  | < 0.3  | < 0.7  | < 0          | < 50%       | (varies)   | < 5   | disabled      |
| pairs        | 0.000  | 0.073  | -49.41%      | 1/5         | -2.94%     | 0/5   | disabled      |

---

## Iteration 1 — code changes shipped (2026-05-25 → 2026-05-26)

### trend (re-confirmed on fresh data through 2026-05-22)

- No code changes (Phase 1.6).
- **DSR:** 0.544 ✓ — **PSR:** 0.990 ✓ — **Bootstrap-5%:** +12.31% ✓ — **Regime:** 3/3 ✓ — **Holdout:** +21.11% ✓
- **Overall: PASS 5/5.** Chosen latest: `vol_target_annual=0.12, allow_short=False, lookbacks_months=[1,3,6,12]`.

### momentum (commit `743ff80`)

- Wired shared `RegimeOverlay` (SPY 200dma cap=0.5 + VIX>threshold cap=0.25) on top of existing Daniel-Moskowitz drawdown control.
- Added `regime_overlay_vix_threshold ∈ {25, 30, 35}` to walk-forward grid.
- VIX series auto-loaded from FRED cache via `_load_vix_safe()` (degrades cleanly when cache missing).
- **DSR:** 0.747 ✓ — **PSR:** 0.970 ✓ — **Bootstrap-5%:** +2.44% ✓ — **Regime:** 1/3 ✗ — **Holdout:** +16.23% ✓
- **Overall: 4/5.** Chosen latest: `lookback_months=9, top_pct=0.3, trend_filter_days=150, regime_overlay_vix_threshold=35.0`.
- **Decision: ENABLE LIVE.** Regime gate failure is structural — long-biased momentum cannot flip crash-regime returns positive without an active inverse sleeve; the overlay attenuates losses but doesn't make crashes profitable (cash returns 0%; gate requires strictly positive). DSR/PSR/bootstrap/holdout very strong, cost-robust at 30bps (+35.13%). Re-evaluate after 2-week paper P&L.

### multi-factor (commit `1ed0bf5`)

- Wired shared `RegimeOverlay`. SPY bars loaded separately (megacap universe has no SPY) via `_load_spy_bars_safe()`.
- Kept existing `dollar_neutral ∈ {True, False}` grid.
- Added `regime_overlay_vix_threshold ∈ {25, 30, 35}`.
- **Validation: in flight** at paper-trading go-live (heavy walk-forward + fundamentals fetch per window). Will be appended below when complete.
- Chosen latest: `quintile_pct=0.25, dollar_neutral=False, vol_lookback=60, regime_overlay_vix_threshold=25.0`.
- **Decision: ENABLE LIVE** alongside momentum on the same structural rationale.

### risk-parity (commit `41eaa4c`)

- Added `shrinkage_floor` parameter that bounds Ledoit-Wolf intensity from below.
- Widened grid: `vol_target_annual ∈ {0.06..0.12}`, `lookback_days ∈ {63, 126, 252, 504}`, `shrinkage_floor ∈ {0.0, 0.20, 0.40}`.
- **DSR:** 0.003 ✗ — **PSR:** 0.165 ✗ — **Bootstrap-5%:** -44.57% ✗ — **Regime:** 1/3 ✗ — **Holdout:** +11.67% ✓
- **Overall: 1/5.** Cost-sensitivity sweep shows essentially zero edge (Sharpe ≈ 0.01 at 0bps, max DD -20.80%). The walk-forward optimizer landed at `vol_target=0.12, lookback=504, shrinkage_floor=0.0` — the HIGHEST vol target with NO shrinkage floor (most aggressive corner of the grid), which suggests the optimizer is over-fitting to the (favourable) post-2022 recovery and getting hammered in 2020/2022. Even widened grid couldn't recover the strategy's poor walk-forward sharpe.
- **Decision: ENABLE LIVE PER USER DIRECTION ("no disabling features").** HRP is a long-only diversified portfolio across SPY/TLT/IEF/GLD/DBC/VNQ/EFA/EEM; the holdout being positive (+11.67%) is meaningful for the next-12-month forward window. Acknowledged risk: this strategy may lose money on paper. Equal $200K allocation means max exposure is bounded. Critical follow-up: revisit by 2026-06-09 — if paper P&L is materially negative, disable and revisit grid (likely needs a regime filter for 2022-style stagflationary periods that took down 60/40 portfolios).

### pairs (commit `126f911`)

- VIX gate at top of `target_positions`: returns `{}` when latest VIX > `vix_max`.
- Per-pair stop-loss: forces flat when `|z| > stop_loss_z`.
- Tightened defaults: `max_active_pairs=3`, half-life `[2, 20]`, `adf_p_max=0.01`.
- Widened grid: `entry_z ∈ {2, 2.5, 3}`, `exit_z ∈ {0, 0.25, 0.5}`, `lookback_days ∈ {30, 45, 60, 90}`, `stop_loss_z ∈ {3.5, 4.5, 6}`, `vix_max ∈ {20, 25, 30}`.
- **Validation: in flight** (slowest — 324-combo grid × 11 windows + per-window PCA discovery).
- Chosen latest: `entry_z=3.0, exit_z=0.0, lookback_days=30, stop_loss_z=3.5, vix_max=30.0`.
- **Decision: ENABLE LIVE WITH DOCUMENTED CAVEAT.** Pairs alpha is structurally weak post-2010 per literature + baseline gate results. Iteration-1 tightening (high z-thresholds, stop-loss, low max-active) is the highest-confidence configuration we can ship. Currently sits at 0 positions in the dry-run because VIX=16.76 and no spreads at z=3.0 entry — this is correct behavior, not a bug. Re-evaluate after 2-week observation window.

---

## Iteration 2 (deferred)

A future iteration could add an active crisis-positive sleeve (rotate to TLT + GLD when overlay factor < 1.0) to flip the regime gate for momentum / multi-factor / risk-parity. That would be a non-trivial strategy redesign and is deferred — current deployment ships with the iteration-1 changes only.

---

## Final live status (post-deployment 2026-05-26)

| Strategy     | `enabled_live` | Gates passed | Notes |
| ------------ | -------------- | ------------ | ----- |
| trend        | True           | 5/5          | re-confirmed on fresh data |
| momentum     | True           | 4/5          | regime gate failure is structural; cost-robust; strong DSR/PSR/holdout |
| multi-factor | True           | TBD          | validate in flight; expected ~4/5 (same shape as momentum) |
| risk-parity  | True           | 1/5          | holdout positive (+11.67%); walk-forward sharpe essentially zero. Monitor closely. |
| pairs        | True           | TBD          | validate in flight; tightest configuration we can ship; observe live |

---

## Dry-run rebalance (2026-05-26 11:07 ET, post fix at `a6e9080`)

```
Account equity:     $1,000,069.92
Enabled strategies: 5  (equal $200K split)
Total orders:       20

Per-strategy outcomes:
  momentum     Target=2  Orders=2   (DBC, EEM via overlay-trimmed top picks)
  multi-factor Target=5  Orders=5   (BAC, GOOGL, JPM, WMT, XOM)
  pairs        Target=0  Orders=0   (no z>3.0 spreads at VIX=16.76)
  risk-parity  Target=8  Orders=8   (full HRP allocation across ETF universe)
  trend        Target=5  Orders=5   (DBC, EEM, EFA, SPY, VNQ)
```

`quant doctor` reports 6/6 PASS prior to dry-run.

---

## Deployment

- **Commits on origin/main:** `5897453..051891e` (RegimeOverlay + 4 strategies + enable + history fix + tear-sheets)
- **Cron schedule (`daily-rebalance`):** `55 19 * * 1-5` UTC (15:55 ET, weekdays).
- **First auto-rebalance fire:** Tue 2026-05-26 19:55 UTC (~3.5 hours after market open).
- **Paper account:** $1,000,069.92 equity, Alpaca paper, $1M starter.
- **CI:** `ci` + `smoke-test` workflows passing on `40bc8e7`.

## Evidence-gated refresh — 2026-05-27

Evidence-gated paper trading supersedes the earlier "enable per user
direction" notes above. `StrategySpec.enabled_live=True` still means a
strategy can be considered for paper trading, but governance now blocks
capital unless fresh validation evidence passes every required gate.

| Strategy     | Command class | DSR   | PSR   | Bootstrap-5% | Regimes pos | Holdout | Governance |
| ------------ | ------------- | ----- | ----- | ------------ | ----------- | ------- | ---------- |
| trend        | full, 5000 bootstrap resamples | 0.520 ✓ | 0.954 ✓ | -2.62% ✗ | 4/4 ✓ | +21.05% ✓ | quarantined: bootstrap_lower |
| momentum     | full, 5000 bootstrap resamples | 0.640 ✓ | 0.928 ✓ | -12.81% ✗ | 2/4 ✓ | +15.67% ✓ | quarantined: bootstrap_lower |
| risk-parity  | full, 1000 bootstrap resamples | 0.082 ✗ | 0.595 ✗ | -37.15% ✗ | 2/4 ✓ | +9.48% ✓ | quarantined: DSR, PSR, bootstrap_lower |
| multi-factor | quick/default params, 1000 bootstrap resamples | 0.007 ✗ | 0.320 ✗ | -63.75% ✗ | 1/4 ✗ | +1.58% ✓ | quarantined: DSR, PSR, bootstrap_lower, regime |
| pairs        | quick/default params, 1000 bootstrap resamples | 0.021 ✗ | 0.260 ✗ | -42.26% ✗ | 0/4 ✗ | +5.76% ✓ | quarantined: DSR, PSR, bootstrap_lower, regime |

Notes:

- `multi-factor` full-grid validation was attempted on 2026-05-27 but was
  stopped after it became clear it would take multiple hours interactively.
  The quick/default-params report is conservative evidence that removes the
  missing-validation state; full-grid evidence belongs in the weekly
  timeout-bounded workflow.
- `pairs` was run in the conservative quick/default-params mode first, per the
  completion plan. Full-grid pairs discovery should also run only under the
  scheduled workflow or a dedicated long-running session.
- The quick `multi-factor` and `pairs` runs were executed in the local sandbox
  after `uv` approval timed out, so Alpaca/yfinance network fetches failed and
  the runs relied on local cached data. Treat these as fail-closed governance
  artifacts, not as a final research-grade optimization result.

## Follow-ups

- Build weekly matrix validation with explicit per-strategy timeouts and network-capable execution.
- 2026-06-09 (2 weeks post-launch): review paper P&L per strategy if any strategy becomes live.
- Iteration-2 design (crisis-positive sleeve via TLT/GLD rotation) deferred to a future plan.
