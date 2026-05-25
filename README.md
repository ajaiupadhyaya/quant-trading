# quant-trading

Systematic trading project — 5 strategies modeled on what AQR / Bridgewater / Citadel / JPM Quant publish about, paper-traded live on Alpaca via GitHub Actions, terminal-first CLI + TUI for navigation.

**Status:** Production-ready for Alpaca paper trading.

- Strategies: PCA pair discovery + Engle-Granger ADF + OU half-life + opt-in Kalman hedge; HRP all-weather with Ledoit-Wolf shrinkage; TSMOM with Daniel-Moskowitz drawdown control; inverse-vol momentum; Hou-Xue-Zhang multi-factor on SEC EDGAR PIT fundamentals.
- Validation: walk-forward + CPCV + DSR + PSR + stationary-block bootstrap + regime stress + OOS holdout + 0/5/15/30bps cost-sensitivity sweep.
- Live ops: pre-trade safety guards (market-open, reconciliation, risk circuit breaker, bar freshness), `quant doctor` pre-flight, daily / nightly / weekly-grid / smoke CI workflows.
- Observability: Textual TUI, combined-book tear-sheet with rolling Sharpe/vol + underwater + round-trip P&L distribution, structured trade journal.

See [§ Status & roadmap](#status--roadmap) for explicitly-deferred items.

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
uv run quant combined-book           # joint equity curve across all live-enabled strategies
uv run quant monitor                 # Textual TUI dashboard (press ? for keybindings)
uv run quant doctor                  # pre-flight check before connecting Alpaca
uv run quant data refresh-fundamentals   # one-time: pull SEC EDGAR PIT facts
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

## Validation gate results (2026-05-25)

| Strategy | DSR | PSR | Bootstrap 5% | Regimes | Holdout | Cost-robust | Verdict |
|---|---|---|---|---|---|---|---|
| **trend** | 0.54 | 0.99 | +12.3% | 3/3 ✓ | +20.2% | ✓ | **LIVE** |
| momentum | 0.84 | 0.99 | +8.8% | 1/3 ✗ | +18.2% | ✓ | research (regime gate) |
| multi-factor | 0.91 | 1.00 | +78.9% | 1/3 ✗ | +11.1% | ✓ | research (regime gate) |
| risk-parity | 0.01 | 0.24 | -42.9% | 1/3 ✗ | +11.3% | weak | disabled (4/5 fail) |
| pairs | 0.00 | 0.07 | -49.4% | 1/3 ✗ | -2.9% | ✗ | disabled (5/5 fail) |

Momentum and multi-factor each pass 4/5 gates with strong margins but lose the regime gate — both long-biased cross-sectional equity strategies are regime-fragile in sharp drawdowns. Drawdown control (Daniel-Moskowitz "managed momentum") reduces magnitude but doesn't flip a crash regime positive. They stay disabled live until a regime overlay (VIX kill switch / cross-sectional dispersion detector / short-leg amplifier) lands. Code is fully wired; flip `enabled_live=True` once the overlay validates.

The regime gate itself was generalized from the spec's strict "≥3 of 5" to "≥50% of TESTED regimes" — our 2010-start cache leaves GFC 2008 + most of China 2015 unreachable.

## Local setup

```bash
git clone <repo>
cd quant-trading
cp .env.example .env                 # fill in Alpaca paper + FRED keys
uv venv && uv sync --all-extras
uv run pytest                        # run the unit tests
```

## Status & roadmap

Spec [`docs/specs/2026-05-23-quant-trading-design.md`](docs/specs/2026-05-23-quant-trading-design.md) is fully implemented. Explicitly deferred:

- **Finnhub earnings calendar** — for skipping earnings days on stat-arb pairs. Pairs OU half-life filter already excludes pairs that won't mean-revert within a normal earnings cycle, so this is a refinement.
- **Frozen golden tear-sheet PDFs** — spec §8 mentions committing reference PDFs that CI diffs against. The HTML tear-sheets are in `data/backtests/`; visual-diff CI is a future iteration.
- **Real-money deployment** — out of scope per spec §7.3. Requires a separate design pass (per-strategy risk attribution, centralized order netting, multi-strategy capital allocation via HRP across strategies, compliance / tax-loss harvesting / wash-sale handling, real-money Alpaca credentials, pager alerts).

## License & disclaimer

Personal research project. Not investment advice. Past performance does not guarantee future results. Paper trading does not guarantee real-money behavior. Real-money deployment is explicitly out of scope of the current design.
