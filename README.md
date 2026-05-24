# quant-trading

Systematic trading project — 5 strategies modeled on what AQR / Bridgewater / Citadel / JPM Quant publish about, paper-traded live on Alpaca via GitHub Actions, terminal-first CLI + TUI for navigation.

**Status:** Brainstorm complete; implementation pending.

**Design spec:** [`docs/specs/2026-05-23-quant-trading-design.md`](docs/specs/2026-05-23-quant-trading-design.md)

## What this is

Five strategies running daily paper trades on Alpaca:

1. **Cross-sectional equity momentum** — Jegadeesh-Titman 12-1 + residual momentum + trend filter + vol scaling
2. **Multi-factor equity portfolio** — Hou-Xue-Zhang 6 factors + HRP weights + factor timing + industry-neutral L/S
3. **Statistical arbitrage / pairs trading** — PCA clustering for pair discovery + Kalman hedge ratios + Ornstein-Uhlenbeck half-life filter
4. **Trend-following multi-asset ETFs** — Moskowitz-Ooi-Pedersen TSMOM ensemble + vol targeting + drawdown control
5. **Hierarchical Risk Parity all-weather** — López de Prado HRP + Ledoit-Wolf shrinkage + constant-vol targeting

All five trade against a single Alpaca paper account; per-strategy attribution via `client_order_id`. Daily equity snapshots + trade logs committed back to this repo by the daily Actions runner — git history IS the audit trail.

## Validation rigor

Every strategy passes this battery before going paper-live:

- Walk-forward analysis (5y train / 1y test / 6mo step)
- Combinatorial purged cross-validation (López de Prado 2018)
- Deflated Sharpe Ratio (Bailey & López de Prado 2014)
- Probabilistic Sharpe Ratio
- Monte Carlo trade-level bootstrap (1000 resamples)
- Regime stress tests (2008 / 2015-16 / 2020 / 2022 / 2024)
- Transaction-cost sensitivity (0 / 5 / 15 / 30 bps slippage)

Pass criteria: deflated Sharpe ≥ 0.3, probabilistic Sharpe ≥ 0.7, positive in ≥3 regimes, positive 5th-percentile bootstrap return.

## CLI

The `quant` command-group is installed when you run `uv sync --all-extras`. From the repo root:

```bash
uv run quant --help                  # top-level help
uv run quant strategies              # list registered strategies (empty until Plan 4)
uv run quant status                  # Alpaca account + open positions (needs .env)
uv run quant data inventory          # show what's on disk under data/

# Stubs landing in later plans:
uv run quant backtest <strategy>     # Plan 2 — backtest engine
uv run quant validate <strategy>     # Plan 3 — validation harness
uv run quant rebalance --dry-run     # Plan 6 — live execution
uv run quant tearsheet <strategy>    # Plan 2 — tear-sheet viewer
uv run quant journal                 # Plan 6 — trade log
uv run quant monitor                 # Plan 6 — Textual TUI
```

## Local setup

```bash
git clone <repo>
cd quant-trading
cp .env.example .env                 # fill in Alpaca paper + FRED keys
uv venv && uv sync --all-extras
uv run pytest                        # run the unit tests
```

## License & disclaimer

Personal research project. Not investment advice. Past performance does not guarantee future results. Paper trading does not guarantee real-money behavior. Real-money deployment is explicitly out of scope of the current design.
