# Quant Trading — Design Spec

> **Implementation status:** Plan 1 ✅ · Plan 2 ✅ · Plan 3 ✅ (this doc) · Plans 4–6 pending.

**Date:** 2026-05-23
**Status:** Brainstorm complete, ready for writing-plans → implementation
**Repo:** `~/Documents/quant-trading` (new standalone)
**Author context:** Brainstormed in conversation alongside completion of [`news-dashboard`](../../../news-dashboard) Quant Lab v1; this project takes the lessons from v1 (engineering rigor) and adds what v1 lacks (state-of-the-art quant research methodology + serious live-trading discipline).

---

## 0. Problem statement

Build a **fully-fleshed-out, industry-quality systematic trading project** that:

1. Implements **5 named strategies** modeled on what major quant firms (AQR, Bridgewater, Citadel, Jane Street factor strategies, JPM Quant) actually publish about
2. **Trains/backtests each on as much historical data as is freely available** — equities back to 2002+, multi-asset back to ETF inception, fundamentals from SEC EDGAR
3. Uses **state-of-the-art validation methodology** — walk-forward + combinatorial purged cross-validation + deflated Sharpe + regime stress tests + Monte Carlo bootstrap
4. **Trades paper-live on Alpaca** via GitHub Actions daily cron — accumulates a real, public, reproducible track record
5. Exposes a **rich CLI + Textual TUI** for navigating the project, observing live runs, opening tear-sheets, drilling into per-strategy state — terminal-first, no web frontend
6. Is **commit-state-back-to-repo** for transparency — daily equity and trades are append-only parquet files committed by the Actions runner; git history IS the audit trail
7. Lays foundation for **future real-money deployment** as a separate follow-on decision (NOT in scope of this spec; gate is "paper trading shows clear edge over buy-and-hold after 3-6 months")

**Non-goal:** beating the market. The honest expectation is risk-managed exposure with Sharpe 0.5–1.0 after costs, annualized returns 3–8%/yr, with meaningfully different behavior from buy-and-hold (positive in regime drawdowns, negative or flat in strong bull markets). The point is *defensible methodology*, not magic returns.

---

## 1. Decisions locked in during brainstorm

| # | Decision | Rationale |
|---|---|---|
| 1 | **New standalone repo** at `~/Documents/quant-trading` | Single-purpose identity, no clutter from news-dashboard's other domains |
| 2 | **5 strategies**, 3 refined-from-Quant-Lab-v1 + 2 net-new | Spans equity factors, market-neutral, trend, macro-allocation |
| 3 | **GitHub Actions** cron at 15:55 ET for daily rebalance | Free, reliable, no server to maintain, Alpaca API keys as secrets |
| 4 | **CLI-first**, no web frontend | Alpaca's own dashboard shows positions/P&L; CLI + TUI handle observability |
| 5 | **Industry-level rigor** — walk-forward + combinatorial purged CV + deflated Sharpe + regime stress + Monte Carlo | Distinguishes a real research process from a hobbyist backtest |
| 6 | **Free data sources only for Q1** | Alpaca free tier (IEX), FRED, SEC EDGAR, yfinance fallback; SIP paid feed only when intraday is added later |
| 7 | **Commit state back to repo** as audit trail | Beautifully transparent; ~10 KB/day = ~3 MB/year is trivial |
| 8 | **Strategies run independently** — own attribution via `client_order_id` prefix; combined book held in single Alpaca account | Simplest mental model; netting added only if/when real money deployed |
| 9 | **Real money is OUT OF SCOPE** for this spec | Separate later decision gated on 3-6 months of paper-live results |

---

## 2. Strategy specifications

Each strategy gets the state-of-the-art version, not the textbook one. Effort estimate per strategy assumes the Quant Lab v1 base class + engine already exist and can be ported.

### 2.1 Cross-sectional equity momentum
**Base:** Jegadeesh-Titman 12-1 top-decile, monthly rebalance, S&P 500 universe.
**SOTA enhancements:**
- Trend filter — long only when name's price > 200-day MA (Faber 2007)
- Residual momentum — strip market beta via rolling regression; rank on residual (Blitz-Huij-Martens 2011)
- Volatility scaling — equal risk contribution per name (Asness-Frazzini-Pedersen)
- Cross-asset overlay — separate momentum signal on SPY/EFA/EEM/AGG/GLD; combine
- Weekly partial rebalances to reduce turnover impact (Novy-Marx)

**Parameters to optimize via walk-forward:** lookback (6, 9, 12 months), skip (0, 1), top decile size (5-15%), trend-filter MA period (150, 200, 250).
**Origin:** refine Quant Lab v1's `cross_sectional_momentum.py`.

### 2.2 Multi-factor equity portfolio (long/short)
**Base:** 4-factor (momentum + value + quality + low-vol) equal-weighted.
**SOTA enhancements:**
- Factor zoo curation — start with Hou-Xue-Zhang 6: momentum, value (B/M), quality (gross profitability), investment (asset growth, inverted), low-volatility, size. Drop any that fail validation.
- Hierarchical risk parity weights between factors (Lopez de Prado 2016)
- Factor timing — combine cross-sectional rank with factor-momentum
- Industry-neutral construction — equal sector weights to avoid mechanical tilts
- Long-short dollar-neutral variant — long top quintile, short bottom quintile

**Parameters to optimize:** which factors to include (subset selection), quintile size, sector neutrality on/off, dollar-neutral on/off, lookback periods per factor.
**Origin:** refine Quant Lab v1's `multi_factor_combo.py`.

### 2.3 Statistical arbitrage / pairs trading
**Base:** OLS hedge ratio + z-score threshold on 5 hand-picked pairs.
**SOTA enhancements:**
- Pair discovery via PCA-on-returns clustering (Avellaneda-Lee 2008) — ~50-100 statistical pairs discovered, not hand-picked
- Multi-test cointegration screen — Engle-Granger + Johansen + Phillips-Ouliaris; require ≥2 pass
- Kalman filter for time-varying hedge ratios (Elliott et al. 2005)
- Ornstein-Uhlenbeck spread modeling — fit half-life; only trade pairs with HL ∈ [1, 30] days
- Risk parity within each pair — equal vol-weighted legs
- Portfolio-level overlay — gross-exposure cap, per-pair concentration limit, correlation-regime de-risking

**Parameters:** discovery window (1y, 2y), cointegration alpha (0.01, 0.05), entry z (1.5, 2.0, 2.5), exit z (0, 0.5), max concurrent pairs (20, 50, 100), half-life range bounds.
**Origin:** refine Quant Lab v1's `pairs_trading.py` (currently only 5 hand-picked pairs).

### 2.4 Trend-following multi-asset ETFs (Time-Series Momentum)
**Universe:** SPY (US equities), TLT (LT bonds), IEF (intermediate bonds), GLD (gold), DBC (commodities), VNQ (REITs), EFA (international developed), EEM (emerging).
**SOTA enhancements:**
- Multiple lookback ensemble (Moskowitz-Ooi-Pedersen 2012) — combine 1m, 3m, 6m, 12m signals
- Volatility-scaled positions — each asset sized to contribute equal portfolio vol (target 10% annual)
- Dynamic vol target — scale gross exposure down in high-vol regimes (Hurst-Ooi-Pedersen 2017)
- Drawdown control — reduce leverage as drawdown deepens (Daniel-Moskowitz "managed momentum")
- Long/short — short the negative-trend assets too, not just no-position

**Parameters:** lookback set (which combos), vol target (8-12%), drawdown-leverage curve, short on/off, regime filter on/off.
**Origin:** **net-new**.

### 2.5 Hierarchical Risk Parity all-weather portfolio
**Universe:** same as 2.4, plus optional crypto sleeve (BTC ETF + ETH ETF) toggleable.
**SOTA enhancements:**
- Hierarchical Risk Parity (Lopez de Prado 2016) — hierarchical clustering then recursive bisection; avoids naive equal-risk pitfalls in correlated assets
- Ledoit-Wolf shrinkage covariance estimator
- Constant vol targeting — leverage adjusted monthly to keep realized vol at 10%
- Risk overlay — VIX-based or MA-based regime filter scales gross exposure down in stress periods
- Drift-band rebalancing — rebalance when weights deviate >5% from target

**Parameters:** vol target (8, 10, 12%), drift threshold (3, 5, 7%), regime filter on/off, crypto sleeve on/off.
**Origin:** **net-new**.

---

## 3. Data sources & storage

### 3.1 Data inventory (all free tier where possible)

| Data | Primary | Backup | Cost | Use |
|---|---|---|---|---|
| Equity OHLCV daily, 20+ yr | Alpaca IEX feed | yfinance | Free | All strategies |
| Equity OHLCV intraday | Alpaca SIP | Polygon | $99/mo Algo+ (defer) | Future intraday |
| Fundamentals (P/B, ROE, etc.) | SEC EDGAR + simfin | yfinance Ticker.info | Free | 2.2 |
| Economic / macro | FRED API | Treasury Direct | Free | 2.4/2.5 regime filters |
| Earnings calendar | Finnhub | Polygon | Free | Avoid earnings on stat arb pairs |
| ETF holdings | iShares/Vanguard/SPDR | ETF.com | Free | 2.4/2.5 |
| Crypto OHLCV | Alpaca crypto | CoinGecko | Free | 2.5 (optional) |
| News sentiment | Alpha Vantage news | reuse news-dashboard pipeline | Free | Future overlays |

### 3.2 Storage layout (everything committed to repo unless noted)

```
data/
├── raw/<source>/<symbol>.parquet         # market data, refreshed nightly
├── universe/sp500.csv, etf_universe.csv  # static reference lists
├── fundamentals/<symbol>.parquet         # point-in-time fundamentals
├── macro/<series_id>.parquet             # FRED data
├── features/<strategy>/*.parquet         # computed signals, residuals, factor scores
├── backtests/<strategy>/
│   ├── tearsheet.html
│   ├── chosen_params.json
│   ├── walkforward.parquet
│   ├── monte_carlo.parquet
│   └── regime_breakdown.parquet
└── live/
    ├── equity.parquet                    # appended each daily run, all strategies
    ├── trades.parquet                    # all paper-trade fills
    └── positions_snapshot.parquet        # daily end-of-day snapshot
```

- **Git LFS** for parquet files >5 MB.
- Bar cache: persistent parquet, refreshed nightly via Action; not re-downloaded per backtest.

---

## 4. Validation methodology (the "industry level" part)

Every strategy passes through the full battery before going paper-live.

1. **Walk-forward analysis** — 5-year train / 1-year test / 6-month step over 2002-2024.
2. **Combinatorial purged cross-validation** (Lopez de Prado 2018) — prevents look-ahead leakage from overlapping windows. Gold standard for time-series ML.
3. **Deflated Sharpe Ratio** (Bailey & Lopez de Prado 2014) — corrects in-sample best-of-grid Sharpe for multiple-testing bias. A naive 1.0 deflates to ~0.5 after correction.
4. **Probabilistic Sharpe Ratio** — confidence interval that true Sharpe is positive.
5. **Monte Carlo bootstrap on trade-level returns** — 1000-resample CIs on total return, Sharpe, max DD.
6. **Regime stress tests** — separate metrics per regime: 2008 GFC, 2015-16 China selloff, 2020 COVID, 2022 bear, 2024 bull. Tear-sheet shows the breakdown.
7. **Out-of-sample after parameter selection** — explicit 2024+ holdout never used during walk-forward. Final "is this real" test.
8. **Transaction cost sensitivity** — backtest under 0 / 5 / 15 / 30 bps slippage curves.
9. **Per-strategy tear-sheet** built with **quantstats** + custom additions for the above.

**Pass criteria for going paper-live:**
- Deflated Sharpe ≥ 0.3
- Positive Monte Carlo lower 5th-percentile total return
- Positive return in ≥3 of the 5 regime stress tests
- Probabilistic Sharpe ≥ 0.7

If a strategy fails any criterion, it stays in `data/backtests/` for transparency but is NOT enabled in the live rebalance Action.

---

## 5. CLI + TUI interface

### 5.1 `quant` CLI (Click + Rich)

One-shot commands:

```
quant backtest <strategy>              # run full walk-forward, write tear-sheet, open it
quant backtest <strategy> --quick       # skip combinatorial CV + bootstrap, fast iteration
quant validate <strategy>              # full pass/fail validation report
quant rebalance [--dry-run]            # daily live rebalance; --dry-run prints orders only
quant status                           # Rich-formatted account state + per-strategy snapshot
quant tearsheet <strategy>             # opens HTML tear-sheet in default browser
quant journal [--since YYYY-MM-DD]     # structured trade log, optionally filtered
quant data refresh                     # nightly: refresh all bar caches
quant data inventory                   # show what's in data/, sizes, last-updated dates
```

### 5.2 `quant monitor` TUI (Textual)

Full-screen multi-pane live monitor:

```
┌─ ACCOUNT ────────────────────┬─ STRATEGIES ────────────────────────┐
│ Equity      $103,247.18      │  momentum    +2.3% ▲ Sharpe 0.7      │
│ Today P&L   +$412.50         │  multi-factor +0.1% ─ Sharpe 0.4      │
│ Buying Pwr  $98,124.00       │  pairs       +0.8% ▲ Sharpe 1.1      │
│ Margin %    0%               │  trend       -0.3% ▼ Sharpe 0.5      │
│ Pattern Day Trader: no       │  risk-parity +0.4% ▲ Sharpe 0.6      │
├─ POSITIONS (12) ─────────────┴──────────────────────────────────────┤
│ AAPL  long  120 @ 184.32   $22,118   +1.2%   [momentum]            │
│ KO    long  450 @  61.20   $27,540   +0.4%   [pairs]               │
│ PEP   short -150 @ 178.20  -$26,730  -0.4%   [pairs]               │
│ TLT   long  280 @  92.10   $25,788   -0.8%   [risk-parity]         │
│ ...                                                                 │
├─ TRADES (today) ────────────────────────────────────────────────────┤
│ 09:32 BUY  AAPL 50 @ 184.20  [momentum]                            │
│ 09:33 SELL TSLA 10 @ 245.10  [pairs]                               │
│ 09:35 BUY  SPY  20 @ 510.40  [trend]                               │
├─ EQUITY CURVE (30d, normalized to $100k) ──────────────────────────┤
│ 110k                                                                │
│      ▁▂▃▃▂▃▄▅▆▆▇▇▆▇█▇▇▇▆▇█▇▇▇█▇▇▇▇█                                │
│ 100k                                                                │
└─────────────────────────────────────────────────────────────────────┘
```

**Behaviour:**
- Updates every 60s from Alpaca account API
- Keyboard: ↑↓ in lists, Enter on strategy → drill into per-strategy panel (positions + equity + recent trades for that strategy only)
- `b <slug>` → open tear-sheet in browser
- `r` → force refresh
- `q` → quit
- `?` → help overlay

Implementation: `textual` library, ~500-800 LOC. Drill-down panels swap content within the same layout; navigation is keyboard-driven, no mouse required.

---

## 6. Repository layout

```
quant-trading/
├── README.md                              ← intro + live equity badge + status table
├── pyproject.toml                         ← uv/poetry-managed, no requirements.txt
├── .github/workflows/
│   ├── daily-rebalance.yml                ← 15:55 ET weekdays, runs quant rebalance
│   ├── monthly-rebalance.yml              ← last weekday of month 15:30 ET, momentum/multi-factor full rebalance
│   ├── nightly-backtest.yml               ← 22:00 ET, refreshes tear-sheets
│   └── ci.yml                             ← pytest + ruff on every push
├── quant/
│   ├── __init__.py
│   ├── cli.py                             ← Click entrypoint for all subcommands
│   ├── tui.py                             ← Textual app
│   ├── strategies/
│   │   ├── base.py                        ← Strategy ABC + StrategySpec dataclass
│   │   ├── cross_sectional_momentum.py
│   │   ├── multi_factor.py
│   │   ├── pairs_trading.py               ← Kalman + OU + clustering
│   │   ├── trend_following.py             ← TSMOM ensemble
│   │   └── risk_parity.py                 ← HRP + vol targeting
│   ├── data/
│   │   ├── bars.py                        ← Alpaca + yfinance fetchers, parquet cache
│   │   ├── fundamentals.py                ← SEC EDGAR + simfin
│   │   ├── macro.py                       ← FRED
│   │   └── universe.py                    ← S&P 500 snapshot, ETF list
│   ├── backtest/
│   │   ├── engine.py                      ← vectorbt-based, ported from Quant Lab v1
│   │   ├── walkforward.py                 ← windows + OOS stitching
│   │   ├── combinatorial.py               ← purged CV (Lopez de Prado)
│   │   ├── deflated_sharpe.py             ← Bailey & LdP correction
│   │   ├── bootstrap.py                   ← Monte Carlo trade-level resample
│   │   ├── regime.py                      ← regime stress test runner
│   │   └── tearsheet.py                   ← quantstats wrapper + custom panels
│   ├── execution/
│   │   ├── alpaca.py                      ← StockHistoricalDataClient + TradingClient
│   │   ├── orders.py                      ← per-strategy attribution via client_order_id
│   │   └── reconciler.py                  ← compute deltas vs Alpaca live state
│   ├── reports/
│   │   ├── tearsheet.py                   ← HTML generator, per-strategy
│   │   └── live_equity.py                 ← combined live equity + attribution
│   └── util/
│       ├── logging.py                     ← structured logging (loguru)
│       └── config.py                      ← env-based config
├── data/                                  ← committed: see §3.2
├── tests/                                 ← pytest, mirrors quant/ structure
└── docs/specs/2026-05-23-quant-trading-design.md   ← this file
```

---

## 7. Operational lifecycle

### 7.1 Initial development (~6 weeks of focused work)

| Week | Milestone |
|---|---|
| 1 | Repo skeleton, CLI scaffolding, data layer (bars + universe + fundamentals + macro), Alpaca client integration |
| 2 | Backtest engine ported from Quant Lab v1 + walk-forward + tear-sheet pipeline |
| 3 | Combinatorial purged CV + deflated Sharpe + bootstrap + regime stress |
| 4 | Strategies 1, 2, 3 (refined ports from Quant Lab v1 with SOTA enhancements) |
| 5 | Strategies 4, 5 (net-new) |
| 6 | TUI, Alpaca paper execution, GitHub Actions wiring, end-to-end smoke |

### 7.2 Steady state (per day)

- **15:30 ET**: nightly Action wakes up (the day before, technically). Refreshes bar cache, fundamentals, macro.
- **15:55 ET weekday**: `daily-rebalance.yml` fires. For each enabled strategy: load chosen params, generate today's signals, compute target positions, submit deltas to Alpaca via `client_order_id` = `<slug>-<date>-<symbol>`. Commit appended `equity.parquet` row + new trades.
- **22:00 ET**: nightly `nightly-backtest.yml` refreshes tear-sheets so the committed HTML stays in sync with the latest data.
- **End of week**: developer runs `quant monitor` to review the week's behavior; runs `quant validate <strategy>` for any strategy whose live behavior is drifting from backtest.

### 7.3 The "go real money" gate (out of scope of this spec)

After 3-6 months of paper-live with at least 3 strategies showing positive deflated Sharpe in BOTH backtest AND live periods, plus regime coverage that includes at least one meaningful drawdown survived intact, write a NEW spec for the real-money phase. That spec will need to add:
- Live-only risk limits (per-strategy max drawdown circuit breakers, per-position size caps)
- Centralized order netting (don't waste spread on offsetting orders)
- Multi-strategy capital allocation (HRP across strategies, not just equal)
- Compliance / tax-loss harvesting / wash-sale handling
- Real-money Alpaca account credentials (separate from paper)
- Pager / on-call alerts (Discord webhook, PagerDuty, etc.)

That is a meaningfully different design and should NOT be confused with paper trading.

---

## 8. Testing & quality bar

- **pytest** with ≥80% coverage on `quant/strategies/`, `quant/backtest/`, `quant/execution/`
- **Type hints throughout**, **ruff** for linting, **mypy** strict mode for the public API surface
- **Hypothesis** property-based tests for the backtest engine and OMS (e.g., simulate_fills invariants)
- **Frozen golden tear-sheets** — for each strategy, commit a reference tear-sheet PDF; CI diffs new tear-sheets against it to catch silent regressions
- **End-to-end smoke** — a separate CI workflow runs `quant rebalance --dry-run` against a snapshot bar cache to confirm the daily rebalance code path doesn't break
- **Paper trade itself acts as the integration test** — over time the live equity should track the OOS portion of the walk-forward backtest within a few percent

---

## 9. Open questions deferred to implementation

These don't need answers to start, but the implementer should flag any that block:

1. **Should the `data/` directory really be committed?** Parquet files grow over time; might exceed GitHub's 1 GB soft limit after 5+ years. Mitigation: Git LFS, or split history with `git rebase --root` annually.
2. **Should the regime stress test "regimes" be hard-coded or detected algorithmically?** Hard-coded is simpler and more interpretable. Detection (HMM, BCPM, etc.) is fancier but adds another layer to validate.
3. **Should we adopt `uv` or stick with `pip`?** `uv` is dramatically faster for repeatable env setup; pyproject.toml is the right shape either way.
4. **Crypto sleeve in risk-parity — default on or off?** Alpaca paper supports crypto. Adds tail risk + 24/7 trading complications.
5. **Backtest data start date — 2002 (best you can get on free tier) or 2010 (post-GFC)?** Longer is better for regime coverage; shorter is faster compute.

---

## 10. Reference / inspiration

Papers and books worth keeping at hand during implementation:
- López de Prado, *Advances in Financial Machine Learning* (2018) — combinatorial purged CV, deflated Sharpe
- López de Prado, *Building Diversified Portfolios that Outperform Out of Sample* (2016) — HRP
- Asness, Moskowitz, Pedersen, *Value and Momentum Everywhere* (2013)
- Moskowitz, Ooi, Pedersen, *Time Series Momentum* (2012)
- Jegadeesh, Titman, *Returns to Buying Winners and Selling Losers* (1993)
- Avellaneda, Lee, *Statistical Arbitrage in the US Equities Market* (2010)
- Bailey, López de Prado, *The Deflated Sharpe Ratio* (2014)
- Hurst, Ooi, Pedersen, *A Century of Evidence on Trend-Following Investing* (2017)
- Hou, Xue, Zhang, *Replicating Anomalies* (2020) — factor zoo curation

Industry-quality open-source references:
- `vectorbt` (engine used by Quant Lab v1) — keep using
- `quantstats` — keep using for tear-sheets
- `mlfinlab` (paid, but the open-source predecessor is `mlfinpy`) — has combinatorial purged CV reference implementation
- `vnpy`, `lean` — TUI patterns
- `riskfolio-lib` — HRP reference implementation
