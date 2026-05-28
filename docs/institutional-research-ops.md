# Institutional Research And Paper-Ops Roadmap

This repo is an Alpaca paper-trading research platform. The institutional bar is
process discipline: reproducible evidence, data provenance, fail-closed
governance, risk reports, and observable operations. It is not HFT market making
or real-money deployment.

## Current Production Path

- `defensive-etf-allocation` is the current governance-live paper baseline.
- Weak research strategies remain quarantined until fresh evidence passes all
  gates.
- `quant validate` now records an experiment row and immutable data snapshot id.
- `quant governance halt` blocks all non-dry-run order submission.
- `quant data quality`, `quant data snapshot`, and `quant risk pretrade` produce
  auditable artifacts under `data/`.

## Remediation Tracks

- `trend`: add crisis-positive sleeve, volatility targeting, and drawdown-aware
  cash filter experiments.
- `momentum`: test market-regime filter, crash protection, and defensive ETF
  overlay.
- `risk-parity`: add adaptive covariance, stricter risk budgets, and stress
  deleveraging.
- `multi-factor`: keep full-grid PIT validation in weekly automation and add
  factor diagnostics before live eligibility.
- `pairs`: improve pair discovery quality, include transaction-cost realism, and
  add borrow/shortability proxies before live eligibility.

## ML/RL Governance

ML/RL is research-only until it beats simple baselines under the same evidence
gates. First candidates are regime classification, volatility/turnover
forecasting, and ensemble weighting. Offline RL/contextual bandits may be tested
only against frozen snapshots; live policy updates are out of scope.

## Operator Commands

```bash
uv run quant data quality
uv run quant data snapshot --symbols SPY,TLT,IEF,GLD,DBC,VNQ,EFA,EEM --start 2010-01-01 --end 2026-05-28
uv run quant research leaderboard --metric dsr
uv run quant risk pretrade
uv run quant governance halt --reason "operator stop"
uv run quant governance resume --reason "verified healthy"
```
