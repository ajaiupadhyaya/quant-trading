# Go-Live Decisions Log — 2026-05-25

Per-strategy decisions during the all-strategies-live tuning pass.

**Validation gate thresholds (per `quant/backtest/validation.py`):**
- DSR ≥ 0.30
- PSR ≥ 0.70
- Block-bootstrap lower-5% ≥ 0%
- ≥ 50% of TESTED regimes positive (regimes with < 30 days OOS data don't count as "tested")
- Holdout total return ≥ 0%

**Baseline (pre-tuning, from project memory + committed tear-sheets in `data/backtests/<slug>/tearsheet.html`):**

| Strategy     | DSR    | PSR    | Bootstrap-5% | Regimes pos | Holdout    | Gates | Status        |
| ------------ | ------ | ------ | ------------ | ----------- | ---------- | ----- | ------------- |
| trend        | 0.54   | 0.99   | +12.3%       | 3/3         | +20.2%     | 5/5   | ENABLED LIVE  |
| momentum     | 0.836  | 0.991  | +8.79%       | 1/3         | +18.19%    | 4/5   | disabled      |
| multi-factor | (~+)   | (~+)   | (+)          | 1/3         | +11%       | 4/5   | disabled      |
| risk-parity  | < 0.3  | < 0.7  | < 0          | < 50%       | (varies)   | < 5   | disabled      |
| pairs        | 0.000  | 0.073  | -49.41%      | 1/5         | -2.94%     | 0/5   | disabled      |

Baselines transcribed from project memory (project_quant_trading.md). Actual
2026-05-25 morning snapshot lives in the committed tear-sheets per strategy.

---

## Iteration 1 (after Task 1.1–1.5 — regime overlay + per-strategy fixes)

Filled in after each strategy completes Phase 1.

### trend
- Re-confirmed on data through 2026-05-22: <result>

### momentum
- Overlay: SPY 200dma halve, VIX>30 quarter, existing DD-control
- Grid additions: `regime_overlay_vix_threshold ∈ {25, 30, 35}`, `regime_overlay_enabled=[True]`
- Result: <result>
- Decision: <ENABLE LIVE | iterate | escape>

### multi-factor
- Overlay: same as momentum
- Grid additions: `dollar_neutral ∈ {False, True}`
- Result: <result>
- Decision: <ENABLE LIVE | iterate | escape>

### risk-parity
- Tuning: `shrinkage_floor ∈ {0.0, 0.20, 0.40}`, wider `vol_target_annual ∈ {0.06, 0.08, 0.10, 0.12}`
- Result: <result>
- Decision: <ENABLE LIVE | iterate | escape>

### pairs
- VIX gate: skip trading when VIX > vix_max (default 25)
- Stop-loss: exit at |z| > stop_loss_z (default 4.5)
- ADF p-value: tightened to 0.01 (from 0.05)
- Half-life: tightened to [2, 20] (from [1, 30])
- Max active pairs: 3 (from 5)
- Result: <result>
- Decision: <ENABLE LIVE | iterate | escape>

---

## Iteration 2 (conditional)

Only run for strategies that still fail one or more gates after Iteration 1.

(empty until needed)

---

## Final live status

| Strategy     | enabled_live | Gates passed | Notes |
| ------------ | ------------ | ------------ | ----- |
| trend        | TBD          | TBD          |       |
| momentum     | TBD          | TBD          |       |
| multi-factor | TBD          | TBD          |       |
| risk-parity  | TBD          | TBD          |       |
| pairs        | TBD          | TBD          |       |

---

## Dry-run output (2026-05-25)

(filled in after Phase 4)
