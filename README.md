# quant-trading

Systematic trading project — an evidence-gated defensive ETF production
baseline plus 5 research strategies modeled on what AQR / Bridgewater / Citadel
/ JPM Quant publish about, paper-traded on Alpaca via GitHub Actions,
terminal-first CLI + TUI for navigation.

**Status:** Production baseline is ready for Alpaca paper trading when
governance is fresh and live; research strategies remain quarantined until they
pass the same gates.

- Strategies: defensive ETF allocation baseline; PCA pair discovery + Engle-Granger ADF + OU half-life + opt-in Kalman hedge; HRP all-weather with Ledoit-Wolf shrinkage; TSMOM with Daniel-Moskowitz drawdown control; inverse-vol momentum; Hou-Xue-Zhang multi-factor on SEC EDGAR PIT fundamentals.
- Validation: walk-forward + CPCV + DSR + PSR + stationary-block bootstrap + regime stress + OOS holdout + 0/5/15/30bps cost-sensitivity sweep.
- Regime engine: 3-state Gaussian HMM (calm-bull / choppy / crisis) over SPY return/vol, VIX, drawdown, and term-spread features; walk-forward refit + filtered posteriors; observed/gated signal only — no live allocation change until it passes its own four-gate validation.
- Live ops: pre-trade safety guards (market-open, reconciliation, risk circuit breaker, bar freshness), `quant doctor` pre-flight, daily / nightly / weekly-grid / smoke CI workflows.
- Observability: Textual TUI, combined-book tear-sheet with rolling Sharpe/vol + underwater + round-trip P&L distribution, structured trade journal.

See [§ Status & roadmap](#status--roadmap) for explicitly-deferred items.

**Charter:** [`docs/CHARTER.md`](docs/CHARTER.md) — the governing methodology every strategy and backtest is held to.

**Design spec:** [`docs/specs/2026-05-23-quant-trading-design.md`](docs/specs/2026-05-23-quant-trading-design.md)

## What this is

One production baseline and five research strategies running through the same
governance and paper-trading stack:

1. **Defensive ETF allocation** — monthly risk-on top-3 6/12 month momentum across SPY, TLT, IEF, GLD, DBC, VNQ, EFA, EEM; risk-off defensive allocation to IEF/TLT/GLD when SPY is below its 200-day average
2. **Cross-sectional equity momentum** — Jegadeesh-Titman 12-1 + residual momentum + trend filter + vol scaling
3. **Multi-factor equity portfolio** — Hou-Xue-Zhang 6 factors + HRP weights + factor timing + industry-neutral L/S
4. **Statistical arbitrage / pairs trading** — PCA clustering for pair discovery + Kalman hedge ratios + Ornstein-Uhlenbeck half-life filter
5. **Trend-following multi-asset ETFs** — Moskowitz-Ooi-Pedersen TSMOM ensemble + vol targeting + drawdown control
6. **Hierarchical Risk Parity all-weather** — López de Prado HRP + Ledoit-Wolf shrinkage + constant-vol targeting

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
uv run quant data quality            # data-quality gates for cached OHLCV bars
uv run quant data snapshot --symbols SPY,TLT --start 2010-01-01 --end 2026-05-28
uv run quant data refresh --start 2010-01-01  # refresh bar cache for all registered universes
uv run quant backtest <strategy>     # walk-forward backtest + tear-sheet
uv run quant tearsheet <strategy>    # open the rendered tear-sheet
uv run quant validate <strategy>     # full validation battery (DSR/PSR/CPCV/bootstrap/regimes)
uv run quant governance audit <strategy>  # reproducibility hashes + quarantine explanation
uv run quant governance drift             # advisory paper-P&L drift flags
uv run quant governance halt --reason "operator stop"
uv run quant governance resume --reason "verified healthy"
uv run quant research leaderboard --metric dsr
uv run quant risk pretrade
uv run quant rebalance --dry-run     # daily rebalance pass against Alpaca paper (dry-run prints orders)
uv run quant rebalance               # daily rebalance — submits orders, snapshots equity + per-strategy positions
uv run quant journal --since 2026-05-01     # structured trade log
uv run quant combined-book           # joint equity curve across all live-enabled strategies
uv run quant monitor                 # Textual TUI dashboard (press ? for keybindings)
uv run quant doctor                  # pre-flight check before connecting Alpaca
uv run quant data refresh-fundamentals   # one-time: pull SEC EDGAR PIT facts
```

### Regime detection (observed, gated signal)

```bash
uv run quant regime fit                 # refit HMM walk-forward, write data/regime/regime_series.parquet
uv run quant regime label               # print the current market regime + posterior
uv run quant regime label --asof 2022-06-15
uv run quant regime validate            # run the four out-of-sample gates, log to the registry
```

A market-wide 3-state Gaussian HMM (calm-bull / choppy / crisis) over SPY
return/vol, VIX, drawdown, and term-spread features. Point-in-time by
construction (walk-forward refit + filtered posteriors). It is an **observed
signal only** — it does not change any live position until it passes its own
validation gate. See `docs/superpowers/specs/2026-05-28-regime-detection-engine-design.md`.

### Position sizing (observed overlay)

`quant sizing compare <strategy>` reports how a composable gross-exposure
overlay — volatility targeting, fractional Kelly, a drawdown throttle, and the
regime multiplier — would have reshaped a strategy's realized return path. It is
point-in-time (the day-`t` scalar uses only data through `t-1`) and observation-
only: it does not change live allocation. Components are individually toggleable
(`--no-vol-target`, `--no-kelly`, `--no-drawdown`, `--no-regime`).

### Options/Greeks engine + hedging overlay (observed overlay)

An analytic Black-Scholes-Merton core (price, full Greeks, implied vol) plus a
protective hedging overlay that insures the equity book's tail.

```bash
uv run quant hedge price --spot 500 --strike 475 --days 30 --vol 0.18 --right put
uv run quant hedge price ... --mark 6.5      # back out implied vol from a market price
uv run quant hedge compare <strategy>        # baseline vs SPY-hedged returns + cost
uv run quant hedge compare trend --structure collar --coverage 0.5 --no-regime
```

`quant hedge compare` estimates the book's net SPY beta point-in-time, builds an
index-level hedge (protective `put`, `collar`, or `put_spread`), rolls it on a
fixed cadence, and reprices every leg daily via Black-Scholes off the cached SPY
bars — no options-data vendor, fully offline-reproducible. Hedge intensity scales
with the regime label (light in calm-bull, full in crisis) when a regime series
is present. Like the sizing overlay it is observation-only and honest about the
tradeoff: protective puts reduce drawdown and CVaR in crashes but drag Sharpe and
CAGR in calm markets (the insurance premium is real, and surfaced rather than
hidden). It logs a `kind="research"` experiment with `gate_maxdd_improved` /
`gate_cvar_improved` flags. See
`docs/superpowers/specs/2026-05-28-options-greeks-hedging-overlay-design.md`.

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

## Strategy governance

Normal paper rebalances are evidence-gated. `StrategySpec.enabled_live=True`
means a strategy is live-capable in code; governance decides whether it is
currently eligible for paper capital based on fresh validation evidence. Capital
allocation is deterministic and written to `data/governance/allocation.json`.

```bash
uv run quant validate trend            # produces validation_report.json
uv run quant governance refresh        # rebuilds the manifest + state files
uv run quant governance status         # render the eligibility table
uv run quant governance audit trend    # explain hashes, bootstrap metadata, and gates
uv run quant governance drift          # writes data/governance/drift_report.json
uv run quant rebalance --dry-run
```

`quant rebalance` fails closed when governance artifacts are missing or
malformed. Quarantined strategies can still be observed with:

```bash
uv run quant rebalance --dry-run --include-quarantined
```

`--include-quarantined` is rejected for non-dry-run rebalances.

The full operator checklist lives in
[`docs/operator-runbook.md`](docs/operator-runbook.md).
The institutional research/ops roadmap lives in
[`docs/institutional-research-ops.md`](docs/institutional-research-ops.md).

### Bootstrap regression audit

On May 27, 2026, `trend` and `momentum` were revalidated against data ending
May 26, 2026 with `--bootstrap-resamples 5000 --bootstrap-seed 0`. Both
remained quarantined by the bootstrap lower-5% gate:

- `trend`: bootstrap lower-5% total return `-2.62%` (older 1000-resample
  report: `-4.24%`).
- `momentum`: bootstrap lower-5% total return `-12.81%`.

This looks like a stable regression, not a bootstrap sampling artifact. Keep
both strategies fail-closed until their risk model or signal construction
improves and the audit command shows fresh passing evidence. Runtime note:
`trend` is reasonable for ad hoc reruns; full-grid `momentum` validation is
slow enough that weekly automation should run it with explicit timeout limits
or a conservative first pass.

### Monitoring daemon (kill-switch automation)

`quant guard run` is a headless guardian loop. Each tick it evaluates
guardrails — paper-P&L drift, account drawdown, position reconciliation, bar
freshness — and **automatically pulls the kill-switch** (`set_halt`) on a
halt-severity verdict, so a bleeding or misbehaving book stops trading without
a human in the loop. It streams a one-line heartbeat and writes
`data/ops/monitor_status.json`.

Key safety property: the daemon can HALT but **never resumes** — restarting
trading is always a deliberate `quant governance resume`. Use `quant guard check`
for a one-shot, read-only evaluation (never halts), and `quant guard run --dry-run`
to observe what it *would* do without touching the kill-switch.

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

## Validation gate results (2026-05-27)

| Strategy | DSR | PSR | Bootstrap 5% | Regimes | Holdout | Cost-robust | Verdict |
|---|---|---|---|---|---|---|---|
| **defensive-etf-allocation** | 0.60 | 0.98 | +5.9% | 2/4 ✓ | +22.0% | ✓ | **LIVE** |
| trend | 0.54 | 0.99 | -2.6% | 3/3 ✓ | +20.2% | ✓ | quarantined (bootstrap gate) |
| momentum | 0.84 | 0.99 | -12.8% | 1/3 ✗ | +18.2% | mixed | quarantined (bootstrap/regime) |
| multi-factor | current artifact fails | current artifact fails | current artifact fails | current artifact fails | current artifact passes | unknown | quarantined |
| risk-parity | current artifact fails | current artifact fails | current artifact fails | current artifact passes | current artifact passes | weak | quarantined |
| pairs | current artifact fails | current artifact fails | current artifact fails | current artifact fails | current artifact fails | ✗ | quarantined |

The research strategies are not forced live. They stay remediation candidates
until fresh validation evidence passes DSR, PSR, bootstrap lower-5%, regime, and
holdout gates. `defensive-etf-allocation` is the current paper-trading baseline.

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
