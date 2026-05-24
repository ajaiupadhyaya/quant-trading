# quant-trading

Systematic trading project — 5 strategies modeled on what AQR / Bridgewater / Citadel / JPM Quant publish about, paper-traded live on Alpaca via GitHub Actions, terminal-first CLI + TUI for navigation.

**Status:** All 6 plans landed. Five strategies registered, walk-forward + validation battery green, live rebalance + journal wired to Alpaca paper, GitHub Actions cron + Textual TUI shipped.

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
uv run quant strategies              # list the 5 registered strategies
uv run quant status                  # Alpaca account + open positions (needs .env)
uv run quant data inventory          # show what's on disk under data/
uv run quant data refresh --start 2010-01-01  # refresh bar cache for all registered universes
uv run quant backtest <strategy>     # walk-forward backtest + tear-sheet
uv run quant tearsheet <strategy>    # open the rendered tear-sheet
uv run quant validate <strategy>     # full validation battery (DSR/PSR/CPCV/bootstrap/regimes)
uv run quant rebalance --dry-run     # daily rebalance pass against Alpaca paper (dry-run prints orders)
uv run quant rebalance               # daily rebalance — submits orders, snapshots equity + per-strategy positions
uv run quant journal --since 2026-05-01     # structured trade log
uv run quant monitor                 # Textual TUI dashboard
```

## Running a backtest

Once at least one strategy is registered (Plans 4-5), run the full walk-forward
pipeline:

    uv run quant backtest <slug>

This:

1. Fetches daily bars for the strategy's universe (Alpaca primary, yfinance backup).
2. Runs walk-forward analysis: rolling 5-year train / 1-year test / 6-month step.
3. For each train window, grid-searches the strategy's parameter space and picks
   the best by in-sample Sharpe.
4. Stitches the OOS test segments into one continuous equity curve.
5. Writes the HTML tear-sheet + sidecar parquet + JSON to
   `data/backtests/<slug>/`.

Open the tear-sheet:

    uv run quant tearsheet <slug>

Refresh the bar cache for the union of all registered universes + ETFs +
S&P 500 (run this before a fresh backtest if the cache is stale):

    uv run quant data refresh --start 2010-01-01

The tear-sheet renders: OOS equity curve, drawdown, monthly returns heatmap,
returns distribution histogram, plus the per-window chosen-parameters table.

### Validation

After a backtest, run the full §4 validation battery:

```bash
uv run quant validate momentum
```

The report includes:
- **Deflated Sharpe Ratio** — multiple-testing-corrected Sharpe (Bailey & Lopez de Prado).
- **Probabilistic Sharpe Ratio** — Pr(true Sharpe > 0).
- **Stationary-block bootstrap** — 5/50/95 percentile CIs for total return, Sharpe, max DD.
- **Regime stress tests** — per-regime metrics across GFC, China '15, COVID, '22 bear, '24 bull.
- **Combinatorial Purged CV** — path-Sharpe distribution.

Exit code `0` = passes the live gate (DSR ≥ 0.30, PSR ≥ 0.70, bootstrap lower-5% > 0, ≥3 positive regimes). Exit code `2` = fails one or more gates.

## Live paper trading

Daily flow once `ALPACA_API_KEY` + `ALPACA_SECRET_KEY` are set:

```bash
uv run quant rebalance --dry-run   # see what would happen
uv run quant rebalance             # for real (paper); snapshots data/live/*.parquet
uv run quant journal --since 2026-05-01
uv run quant monitor               # full-screen Textual dashboard
```

Per-strategy attribution is carried via `client_order_id` (prefix = strategy slug)
and an append-only `data/live/strategy_positions.parquet` snapshot — so the
combined Alpaca account is shared but each strategy keeps its own books.

## GitHub Actions

Two production workflows ship in `.github/workflows/`:

- **`daily-rebalance.yml`** — Mon–Fri 19:55 UTC. Refreshes bar caches, runs the
  rebalance, commits `data/live/*.parquet` + new bars back to the repo. Manual
  dispatch supports a `dry_run` switch.
- **`nightly-backtest.yml`** — Tue–Sat 02:00 UTC. Walk-forward backtest in a
  5-way matrix, each strategy in parallel; commits refreshed tear-sheets under
  `data/backtests/<slug>/`.

Both need repository secrets `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `FRED_API_KEY`.

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
