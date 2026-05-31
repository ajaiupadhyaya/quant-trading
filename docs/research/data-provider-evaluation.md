# Data-Provider Intake Evaluation

_Last updated: 2026-05-31_

## Purpose

This document is the **data-provider arm of the Track A.2 intake rubric** for the quant-trading platform. It records the disposition of every external-data API key found in the project's `.env` and judges each against the gaps in the data layer, under the governing **dependency bar**:

> "Add no runtime dependency unless it has a concrete task and measurable acceptance criteria."
> — `docs/superpowers/plans/2026-05-27-quant-platform-completion.md`, Task 6 ("Awesome Quant Research Intake"), line 131.

This bar is operationalized as a per-plan invariant: every implementation plan's Tech Stack line asserts "No new dependencies" (regime, sizing, monitoring, borrow/financing, market-impact, and notably the options-Greeks overlay, whose design at `2026-05-28-options-greeks-hedging-overlay-design.md` deliberately chose analytic Black-Scholes off cached bars so there is "no options-data vendor, no new dependencies"). Consequently, **no key is adopted as a runtime dependency unless it fills a concrete, currently-starved consumer that an already-installed source cannot serve, with a measurable acceptance criterion.**

**Archived keys are preserved, not deleted.** Every key below — including the ones judged irrelevant or redundant — is retained in `.env.archived` (moved out of the active `.env`) with the provenance recorded here, so that a future need (a non-US universe, a paid PIT-membership tier, a costed options overlay) can revisit the decision without re-discovering what the key was. The "Archived (unused, retained for provenance)" section at the end is the canonical inventory of those preserved keys.

## Outcome summary

After applying adversarial review, **no provider clears the bar for adopt-now.** The two strongest candidates (Tiingo, FMP) were downgraded to sandbox because the concrete gaps they target can be lit up first with sources already in the dependency tree (Alpaca's installed adjustment/corporate-actions capability; yfinance's already-fetched `adj_close`; a ~2-line EDGAR `dei` tag), which the dependency bar requires us to exhaust before adding a new runtime vendor.

| Decision | Keys |
| --- | --- |
| **Adopt now** | _(none)_ |
| **Sandbox (research-only trial)** | `TIINGO_API_KEY`, `FMP_API_KEY`, `POLYGON_API_KEY`, `FINNHUB_API_KEY` |
| **Archive (retained for provenance)** | `ALPHA_VANTAGE_API_KEY`, `NEWSAPI_KEY`, `OPEN_FIGI_API_KEY`, `X_RAPIDAPI_KEY`, `FX_MACRO_DATA_API_KEY`, `TRADE_WATCH_API_KEY`, `AUTHORIZATION` |

## Provenance table

| Key | Provider | Origin / where it came from | Decision |
| --- | --- | --- | --- |
| `TIINGO_API_KEY` | Tiingo | Speculatively-collected market-data key in `.env`, unwired (not referenced in quant-trading code) | Sandbox |
| `FMP_API_KEY` | Financial Modeling Prep (FMP) | Live key in `.env` (`jo2k66jV0U2LO8rTDM1lTUVtomRFRxri`), live-probed 2026-05-31 against `/stable/` API; unwired | Sandbox |
| `POLYGON_API_KEY` | Polygon.io (rebranded "Massive") | Speculatively-collected market-data key in `.env`, unwired | Sandbox |
| `FINNHUB_API_KEY` | Finnhub (finnhub.io) | Speculatively-collected market-data key in `.env`, unwired; vendor named by repo | Sandbox |
| `ALPHA_VANTAGE_API_KEY` | Alpha Vantage | Speculatively-collected market-data key in `.env`, unwired | Archive |
| `NEWSAPI_KEY` | NewsAPI.org | Speculatively-collected news key in `.env`, unwired | Archive |
| `OPEN_FIGI_API_KEY` | OpenFIGI (Bloomberg L.P.) | Speculatively-collected identifier-mapping key in `.env`, unwired | Archive |
| `X_RAPIDAPI_KEY` | RapidAPI gateway (underlying API unspecified) | Bare line in `.env` (`bc550bb31bmsh...jsn...`), no `X-RapidAPI-Host` anywhere in repo | Archive |
| `FX_MACRO_DATA_API_KEY` | FXMacroData (fxmacrodata.com) | Key (`8imyc2...`) in `.env` matching vendor token format, unwired | Archive |
| `TRADE_WATCH_API_KEY` | TradeWatch (tradewatch.io) | `.env` line 28 grab-bag key; not declared in `quant/util/config.py` Settings (`extra="ignore"` drops it) | Archive |
| `AUTHORIZATION` | None — internal bearer token | News-dashboard self-auth header (`frontend/src/lib/api.ts:38`, `backend/app/auth.py:28-36`); not a vendor | Archive |

## Rubric methodology

Each key was first **identified** (matched to a real provider and verified against current 2026 docs; `identified=false` where the key is a generic gateway/header with no pinnable provider). It was then scored on six axes (1-5), and a recommendation derived under the dependency bar:

- **relevance** — does the provider serve the US-equities/ETF systematic system at all?
- **data_quality** — accuracy, point-in-time integrity, no-lookahead fitness.
- **free_tier_sufficiency** — is the free tier usable as a *real* input (universe coverage, rate limits, history depth), not just marketing?
- **integration_effort** — how clean is the seam (BarProvider Protocol, `set_adjustments`, an EDGAR concept tag)? Higher = easier.
- **overlap_penalty** — does it duplicate an existing free/already-paid source (yfinance, Alpaca, FRED, EDGAR)? Higher = less redundant.
- **blast_radius** — how contained is a failure if adopted? Higher = safer.

**Decision rule.** `adopt-now` requires (a) a concrete downstream consumer for a HIGH/MEDIUM gap, (b) a measurable acceptance criterion, (c) a usable free tier, and **(d) that no already-installed source can serve the same task** — the decisive constraint. `sandbox` = a legitimate research-only trial worth running (concrete-enough task, but lower priority or beaten by an in-house path) — never wired as a runtime dependency. `archive` = redundant, irrelevant, or unusable; retained with provenance.

**Adversarial pass.** Each adopt-now candidate was stress-tested on five tests: (1) redundancy, (2) integration concreteness, (3) measurable acceptance, (4) free-tier adequacy, (5) real edge vs. speculative. A failed Test 1 (an already-available source serves the task) is dispositive against adopt-now. Both adopt-now candidates failed Test 1 and were downgraded to sandbox.

### Gap context (what we are trying to fill)

The HIGH-priority data gaps are: **(1)** no corporate-actions (splits/dividends) source feeding the tested-but-starved `quant/intraday/data/adjustments.py` + `store.set_adjustments()` back-adjustment machinery (grep confirms only tests call it); **(2)** no point-in-time / historical universe membership (survivorship bias — `universe.py.sp500_constituents()` returns only today's Wikipedia list); **(3)** no shares-outstanding, so `quant/strategies/multi_factor.py:202-220` fakes market cap as `price * 1.0`, breaking the SIZE factor and biasing book-to-market. MEDIUM: **(4)** no real options/IV source for the Greeks overlay; **(5)** no redundant/alternative price source (single-vendor Alpaca concentration; `bars.py` falls back to yfinance via a bare except-branch, not the `BarProvider` Protocol). LOW: **(6)** no news/sentiment source; **(7)** macro/EDGAR caches never refresh (a TTL bug, not a missing source).

## Per-provider evaluations

### Tiingo — `TIINGO_API_KEY` — **Sandbox** (downgraded from adopt-now)

**Capabilities.** Free EOD equities/ETF/MF prices with both raw and adjusted OHLCV (30+ yrs history); per-bar corporate-action fields `splitFactor` and `divCash` on the **free** tier; precomputed `adjClose`; second independent daily-bar vendor; IEX intraday last-price, crypto, forex (not needed). Fundamentals API and News API are **paid only**.

**Free tier.** Starter/free tier: ~500 unique symbols/month, ~50 symbols/hour. Includes EOD prices with `{open,high,low,close,volume, adjOpen…adjClose, divCash, splitFactor}` — i.e. adjusted prices **and** raw split/dividend fields. Excludes (paywalled): Fundamentals API, News API.

**Overlap.** EOD daily OHLCV overlaps yfinance and Alpaca — but for gap #5 that overlap is the *point* (a real injected 2nd BarProvider vs. today's bare except-branch). The split/dividend feed overlaps no *wired* source in the repo. Tiingo Fundamentals would overlap free EDGAR (and is paywalled); Tiingo News overlaps free Alpaca `get_news` (and is paywalled).

**Gap fit.** Gap #1 (HIGH, corporate actions) and gap #5 (MEDIUM, redundant price source).

**Decision + rationale.** Originally adopt-now, **downgraded to sandbox** by adversarial review on Test 1 (redundancy):
- The load-bearing premise "no free corporate-actions source exists; EDGAR/yfinance/Alpaca provide no wired input" is **false against installed code**. The already-installed Alpaca SDK (v0.43.4) exposes `CorporateActionsClient` + `CorporateActionsRequest`, and `StockBarsRequest` carries an `adjustment` param (`RAW/SPLIT/DIVIDEND/ALL`) — the same vendor already in use can return adjusted bars and a corporate-actions feed with zero new dependency.
- `quant/data/bars.py` already fetches yfinance with `auto_adjust=False` and renames `"adj close"`→`"adj_close"` (line 136), then **discards it** only because `_BAR_COLUMNS=[open,high,low,close,volume]` filters it out (lines 18, 130, 138). A free adjusted-close reference is already retrieved and thrown away by a column filter, not absent.

Integration concreteness was also overstated: `set_adjustments`/`adjust_prices` and `BarProvider`/`get_provider_bars` are each called only by a unit test (unused scaffolding, not wired seams); `quality.py` has no cross-vendor reconciliation function (it would be net-new code); and the adjustment machinery lives in the **intraday** `MarketDataStore`, while daily `bars.py` has no adjustment layer and no coupling to the store — feeding the intraday store does not fix the daily path. Measurable acceptance does hold (`tests/intraday/data/test_store_pit.py` is a tight float-tolerance assertion across a 4:1 split), and excluding paywalled Fundamentals/News is sound — so this is genuinely sandbox, not archive. **Disposition:** prototype the Alpaca-native / yfinance-`adj_close` fix first; graduate Tiingo only if those prove insufficient (e.g., for a true second-vendor reconciliation cross-check). Relevant files: `quant/intraday/data/adjustments.py`, `quant/intraday/data/store.py`, `quant/data/providers.py`, `quant/data/quality.py`, `quant/data/bars.py` (lines 18/130/136-138), `tests/intraday/data/test_store_pit.py`.

### Financial Modeling Prep (FMP) — `FMP_API_KEY` — **Sandbox** (downgraded from adopt-now)

**Capabilities.** `/stable/shares-float` (`outstandingShares`, `floatShares`, `freeFloat`) from SEC filings — **free**; `/stable/delisted-companies` (symbol/exchange/ipoDate/delistedDate) — **free**; `/stable/splits` and `/stable/dividends` histories — **free**; EOD prices + company profile — free but redundant; `/stable/historical-sp500-constituent` (PIT membership with add/remove dates) — **paywalled on free**.

**Free tier.** 250 requests/day, US-focused, 500MB trailing-30-day bandwidth. Current `/stable/` API only (legacy `/api/v3/` retired 2025-08-31). Live-probed 2026-05-31: `shares-float` returned `outstandingShares=14,687,356,000` for AAPL; `delisted-companies` returned a real list; `splits`/`dividends` returned full histories (AAPL 4:1 2020, 7:1 2014). PIT membership confirmed restricted on free.

**Overlap.** `shares-float` overlaps EDGAR *in principle* (EDGAR exposes `CommonStockSharesOutstanding` / `dei:EntityCommonStockSharesOutstanding`) but `multi_factor.py` never wired it. Splits/dividends overlap Alpaca `get_corporate_actions` (same paper account). EOD prices and profile duplicate yfinance/Alpaca.

**Gap fit.** HIGH gap #3 (shares-outstanding); HIGH gap #2 partial (delisted-ticker list, but not PIT membership); HIGH gap #1 alternative path (corp actions).

**Decision + rationale.** Originally adopt-now, **downgraded to sandbox** on Test 1. The gaps are real, but mis-attributed as needing a new vendor: `quant/data/edgar.py` already downloads and caches the free SEC `/companyfacts` JSON; `_CONCEPT_TAGS` (lines 53-66) simply omits a shares-outstanding tag and `_extract_concept` (line 154) reads only the `us-gaap` namespace, never `dei`. **Adding shares-outstanding is a ~2-line change to an already-cached free feed** (append a `CommonStockSharesOutstanding`/`EntityCommonStockSharesOutstanding` tag + a "shares" unit) — no new key, no Protocol, no 250/day budget. The rationale itself concedes "EDGAR exposes the concept but multi_factor never wired it," which is fatal: this is an *unwired-existing-source* gap, not a no-source gap. The proposed acceptance criterion was also self-refuting (it reconciles FMP against "latest EDGAR `CommonStockSharesOutstanding`," a value the codebase doesn't yet expose — so wire EDGAR first, at which point FMP is the redundant cross-check) and targeted the wrong universe (cited the 102-name `DEFAULT_UNIVERSE`, but the size factor lives in daily `multi_factor.py` whose `MEGACAP_UNIVERSE` is only ~19 names). Corp-actions: `set_adjustments` is test-only (a real starved path), but Alpaca `get_corporate_actions` is the no-new-vendor substitute. The one genuinely non-redundant item — PIT S&P-500 membership — is paywalled. **Disposition:** close gaps via the zero-dependency EDGAR `dei`-tag and Alpaca paths first; keep FMP as a sandbox cross-vendor reconciliation probe (shares-float vs. EDGAR `dei`).

### Polygon.io (rebranded "Massive") — `POLYGON_API_KEY` — **Sandbox**

**Capabilities.** Daily/intraday OHLCV aggregates; tick trades + NBBO quotes (paid); corporate actions — splits (v3) and cash dividends included on free Basic, updated daily, no real-time delay; ticker reference + ticker-events; company financials; market news; options chains + real-time Greeks/IV (Advanced ~$199/mo); WebSocket streaming (delayed on Starter).

**Free tier.** Basic: 5 API calls/min, end-of-day aggregates only, ~2-year history, 15-min delay on real-time feeds. Crucially, splits + dividends reference endpoints are included free, updated daily, no delay. The 5 calls/min + 2yr lookback make it unusable as a backtest bar source for a wide universe.

**Cost.** Free Basic; Starter ~$29/mo (unlimited calls, 15-min delayed, 5yr+ history); Developer ~$79-99/mo (real-time); Advanced ~$199/mo (tick history, real-time options Greeks/IV).

**Overlap.** Heavy. Daily bars duplicate yfinance/Alpaca; corporate actions duplicate Alpaca `get_corporate_actions` (already available on the paper account); financials duplicate EDGAR; intraday duplicates Alpaca SIP; news duplicates Alpaca `get_news`. Does NOT fill shares-outstanding (an EDGAR tag fix) or PIT membership (not in free tier).

**Gap fit.** HIGH gap #1 (corporate actions) — but already closeable via Alpaca with no new dependency; MEDIUM gap #5 (redundant price source / reconciliation).

**Decision + rationale.** **Sandbox.** Not adopt-now: the single HIGH gap it touches (corp actions) is already closeable with zero new dependency via Alpaca, free splits/dividends have documented holes (e.g., missing SPY dividends) making it a weaker corp-actions source than Alpaca, and it does not solve shares-outstanding or PIT membership. Not archive: it is a legitimate, well-known provider with a genuine non-redundant angle — an injected second `BarProvider` for cross-vendor reconciliation in `quality.py` (gap #5) and a free corporate-actions cross-check against Alpaca. That is a research-only trial worth running, not a runtime dependency.

### Finnhub — `FINNHUB_API_KEY` — **Sandbox**

**Capabilities.** Real-time US quotes (`/quote`), company profile (`/stock/profile2`), company/general news, **earnings calendar (`/calendar/earnings`) — free**, basic financials, recommendation trends, insider transactions, peers, earnings surprises. Historical OHLC candles (`/stock/candle`) are **premium-only** (403 on free keys even for US large-caps); detailed statements, dividends/splits, international data are premium.

**Free tier.** 60 calls/min (30/sec burst). Decisive restriction: `/stock/candle` is paywalled, so Finnhub **cannot** serve as a free second `BarProvider` for gap #5 — that gap is better closed by injecting the already-present yfinance as a real `BarProvider`.

**Cost.** Free $0; premium from ~$50-60/mo, with narrowly-scoped add-on tiers ~$11.99-$49.99/mo; historical candles + detailed financials require paid.

**Overlap.** Heavy with free incumbents: bars overlap Alpaca/yfinance (and candles are paywalled anyway); fundamentals overlap EDGAR (PIT, no-lookahead — Finnhub's are non-PIT and the deep data is paid); macro overlaps FRED; news overlaps Alpaca `get_news`. The only non-overlapping free capability is the forward earnings calendar.

**Gap fit.** A forward earnings-calendar gate for the pairs OU half-life filter — the only free, non-overlapping, concretely-spec'd use.

**Decision + rationale.** **Sandbox.** The one genuine free fit is `quant/data/earnings.py` per the repo's own deferred spec (`docs/superpowers/specs/2026-05-26-deferred-followups.md` §2): wrap free `/calendar/earnings`, cache to `data/earnings/<symbol>.parquet`, and gate pair-selection in `quant/strategies/pairs_trading.py` to drop pairs whose nearest earnings falls inside the OU half-life window, with acceptance "gated pairs backtest does not regress Sharpe." But the project explicitly **deferred** this as a refinement, and the existing OU half-life filter already excludes pairs that won't mean-revert within a normal earnings cycle — so it is a LOW-priority nice-to-have, not the HIGH/MEDIUM gap adopt-now requires. Note it does **not** satisfy the `BarProvider` Protocol (candles are paywalled). Research-only trial behind the charter bar.

### Alpha Vantage — `ALPHA_VANTAGE_API_KEY` — **Archive**

**Capabilities.** Daily/weekly/monthly time series (unadjusted daily free; daily-adjusted premium); full fundamental suite (overview, income/balance/cash-flow, dividends, splits, shares-outstanding — snapshot, **not** PIT); listing/delisting status; economic indicators (GDP/CPI/yields/fed-funds/unemployment/NFP); 50+ technical indicators, FX, crypto. Intraday, daily-adjusted, options, news/sentiment, real-time quotes are premium.

**Free tier.** 25 requests/**day**, 5 req/min. Cannot cover even `DEFAULT_UNIVERSE` in a single pass.

**Cost.** Free $0; Standard $49.99/mo (75 RPM), Premium $99.99, Professional $149.99 (realtime), Enterprise $249.99.

**Overlap.** Heavy with every free incumbent: fundamentals duplicate EDGAR (and AV is **non-PIT**, so it would inject look-ahead, violating the no-lookahead charter principle); economic indicators duplicate FRED; daily bars duplicate yfinance/Alpaca; corp actions/options/news reachable via the existing Alpaca paper account.

**Gap fit.** None.

**Decision + rationale.** **Archive.** Every gap it could touch is filled better by a free zero-dependency incumbent, and its non-PIT fundamentals would actively violate the charter. The one non-redundant angle (LISTING_STATUS for delisting) is throttled to uselessness by 25 req/day and gives delisting status, not the index add/remove dates the PIT-universe gap needs. The free tier is too thin to be a runtime dependency and the news/sentiment angle is premium and duplicates Alpaca — so not even sandbox-worthy.

### NewsAPI.org — `NEWSAPI_KEY` — **Archive**

**Capabilities.** Keyword/full-text article search across 150k+ sources (`/v2/everything`), top-headlines (`/v2/top-headlines`); article metadata (source/author/title/description/url/publishedAt) with content truncated to 200 chars; filter by source/domain/language/date.

**Free tier.** $0 Developer plan, but **development/testing-only** (prohibited in staging/production per ToS), 100 req/day, **24-hour article delay**, 1-month lookback. Functionally unusable for backtests or live signals.

**Cost.** First production-capable tier is Business at **$449/month** (real-time, 5yr history); Advanced $1,749/mo.

**Overlap.** Directly overlaps Alpaca `get_news` (free under the existing paper account, same vendor as bars/intraday/execution; the gap map names it as the intended path), and is worse (general non-tickerized news, no sentiment, 200-char content).

**Gap fit.** Only the LOW-priority, self-described "most speculative" news/sentiment gap.

**Decision + rationale.** **Archive** on three independent counts: (1) the free tier is dev-only/delayed/short-lookback — unusable; (2) it is redundant with the free, already-credentialed Alpaca `get_news`; (3) it fails the charter bar — grep finds zero `get_news`/`NewsEvent` consumers, so the proposed `NewsEvent` join into `events.py` and a regime news-volatility feature are purely hypothetical with no acceptance criterion. If news ever becomes a real research question, prototype against free Alpaca `get_news`, not a $449/mo general-news feed.

### OpenFIGI (Bloomberg L.P.) — `OPEN_FIGI_API_KEY` — **Archive**

**Capabilities.** `/v3/mapping` (convert 25+ id types — TICKER, ISIN, CUSIP, SEDOL, BB_GLOBAL, OCC_SYMBOL — to FIGIs + metadata); `/v3/search`; `/v3/filter`. Returns **reference metadata only** (FIGI, ticker, name, exchange, market sector, security type).

**Free tier.** Fully free, no daily caps. Unauthenticated: mapping 25 req/min; authenticated key raises limits. $0 for all use.

**Overlap.** Duplicates the only id-mapping need in the repo — ticker→CIK for US equities, already free via EDGAR `company_tickers.json` (`quant/data/edgar.py:_ticker_to_cik_map`).

**Gap fit.** None. It is reference data, not market data — no price bars, no corporate actions, no shares-outstanding, no options/IV, no PIT membership, no news.

**Decision + rationale.** **Archive.** It fills none of the seven gaps (every HIGH/MEDIUM gap needs a time-series or event feed it does not supply), cannot satisfy the `BarProvider` Protocol (returns no OHLCV), and its actual strengths (global cross-market symbology, fixed income, OCC option symbols) are irrelevant to a US-ticker-keyed equities/ETF system. No consumer module, no Protocol, no acceptance criterion — fails the charter bar outright.

### RapidAPI gateway — `X_RAPIDAPI_KEY` — **Archive** (provider unidentified)

**Capabilities.** Gateway/marketplace access to ~98,000 heterogeneous third-party APIs; a single key authenticates to any subscribed API, with the actual provider chosen per-request via the `X-RapidAPI-Host` header. No intrinsic market-data capability of its own.

**Free tier.** Not applicable to the gateway itself; each fronted API sets its own (typically a restrictive BASIC tier ~100-500 req/month, then PRO ~$25 / ULTRA ~$75 / MEGA ~$150, plus 25% RapidAPI commission and per-call overage).

**Overlap.** Likely duplicates yfinance/FRED/EDGAR/Alpaca via repackaged Yahoo/Alpha-Vantage-style finance APIs behind the gateway.

**Gap fit.** None definable.

**Decision + rationale.** **Archive**, `identified=false`. It is a generic gateway credential, not a provider. Grep across the entire quant-trading codebase (`*.py`, `*.toml`, `*.md`) finds **zero** references to `rapidapi`, `X-RapidAPI-Host`, or `X_RAPIDAPI_KEY` — the key appears only as a bare line in `.env`. With no host pinned, the underlying API is unidentifiable, so no schema to integrate against, no acceptance criterion definable, and the charter bar cannot be cleared. Marginal value is negative (paying a middleman for data already free directly).

### FXMacroData — `FX_MACRO_DATA_API_KEY` — **Archive**

**Capabilities.** 76+ macro indicators (GDP, CPI/PPI, unemployment, policy/risk-free rates, M1/M2/M3, 2Y-10Y gov bond yields) across ~40 currencies; **point-in-time release timestamps with `?at=` vintage snapshots** (the one genuinely attractive feature); FX spot (ECB reference); CFTC COT positioning; commodities (gold/silver/platinum); release calendar; consensus/IMF predictions; live SSE stream. Ships an official Python client.

**Free tier.** No-key: release calendar, catalogue, FX spot, market sessions. USD macro free for the most recent **365 days**. Non-USD indicators and commodities require a paid key.

**Cost.** Professional from $25/month (14-day trial requires a credit card).

**Overlap.** Directly duplicates FRED for the only slice this US-equities system would use. `quant/data/macro.py` already pulls VIXCLS/DGS10/DGS2/UNRATE/CPIAUCSL/DFF/GDPC1 free and unlimited; FXMacroData sources its USD indicators from FRED itself.

**Gap fit.** None. It grazes only the LOW-priority macro-cache-staleness gap, which is a TTL/refresh bug in existing FRED caching, not a missing source.

**Decision + rationale.** **Archive.** Zero US-equity/ETF coverage, so it touches none of the HIGH/MEDIUM gaps. Its free portion duplicates FRED; everything beyond FRED is non-USD FX/macro with no consumer in a US-only system, behind a $25/mo paywall. No concrete task, free part redundant, paid part irrelevant.

### TradeWatch (tradewatch.io) — `TRADE_WATCH_API_KEY` — **Archive**

**Capabilities.** Real-time + historical OHLCV candles and tick history via REST; WebSocket streaming (paid only); 2,000+ symbols across **Crypto, Forex, Commodities, Indices** (primary marketed coverage); normalized JSON schema. Stocks/ETFs appear only in free-tier marketing copy, absent from the homepage coverage statement.

**Free tier.** $0, hard 1 request/min, **daily-update-only** data, REST-only (no WebSocket).

**Cost.** Free $0; Starter ~$15/mo (delayed, no WebSocket); Pro ~$57/mo (real-time + 10 WS); Enterprise ~$165/mo.

**Overlap.** Duplicates free yfinance for daily bars (the only equities-relevant use) and is worse on freshness/rate limits. Core crypto/forex/commodities/indices coverage is irrelevant.

**Gap fit.** None. It offers no corporate actions, shares-outstanding, PIT membership, fundamentals, options/IV, or news.

**Decision + rationale.** **Archive.** Crypto/forex-first; offers none of the actual equities gaps, each of which has a no-new-vendor in-house fix (Alpaca `get_corporate_actions`; EDGAR `CommonStockSharesOutstanding`). The only gap it could nominally touch (redundant bar source) is unusable: 1 req/min + daily-only cannot reconcile or fail over an S&P-500 universe, and daily-bar use is pure yfinance overlap. The key is also orphaned — `.env` line 28, **not declared** in `quant/util/config.py` Settings (`extra="ignore"` silently drops it), referenced nowhere.

### AUTHORIZATION — `AUTHORIZATION` — **Archive** (not a provider)

**Capabilities / free tier / cost.** N/A. `AUTHORIZATION` is not an external data provider — it is a generic HTTP `Authorization: Bearer <token>` header carrying the **news-dashboard's own internal auth token**. Code evidence is conclusive: `frontend/src/lib/api.ts:38` sets `Authorization: Bearer ${authToken}` on requests to the dashboard's own backend (`BASE_URL = http://127.0.0.1:8000`), and `backend/app/auth.py:28-36` (`require_auth`) validates that header against `DASHBOARD_TOKEN`, the dashboard's self-protection secret exchanged from `DASHBOARD_PASSWORD` via `login()`.

**Decision + rationale.** **Archive**, `identified=false`. There is no third-party endpoint, no vendor, no market data — nothing to research. It cannot fill any gap and has no integration point against `quant/data/providers.py` `BarProvider` or any other module. Retained in `.env` only because it belongs to the adjacent news-dashboard's plumbing.

## Adopt-now integration points and acceptance criteria

**None.** After adversarial review, no provider is adopted as a runtime dependency. The two candidates that reached adopt-now in first-pass evaluation (Tiingo, FMP) were downgraded because the dependency bar requires exhausting already-installed sources first. The concrete, **zero-new-dependency** fixes that supersede them — and which should be implemented before any sandbox graduation — are:

1. **Shares-outstanding (HIGH gap #3)** — add a `CommonStockSharesOutstanding` / `dei:EntityCommonStockSharesOutstanding` tag to `_CONCEPT_TAGS` in `quant/data/edgar.py` (lines 53-66) and extend `_extract_concept` (line 154) to read the `dei` namespace. Wire the result into `quant/strategies/multi_factor.py:_fundamentals_panel` (lines 202-228) so `market_cap = price * shares_outstanding`, enabling the advertised `size = -log(market_cap)` factor. _Acceptance:_ cross-sectional rank correlation between `size` and raw price drops from ~1.0 (current price-proxy) to <0.5 over `MEGACAP_UNIVERSE`.
2. **Corporate actions (HIGH gap #1)** — feed `Adjustment` objects from the already-installed Alpaca `CorporateActionsClient` (and/or the `adjustment` param on `StockBarsRequest`) into `quant/intraday/data/store.set_adjustments()`, and stop discarding the `adj_close` already fetched in `quant/data/bars.py` (lines 18/130/136-138) for the daily path. _Acceptance:_ `get_minute_bars(as_of=...)` for a split name (NVDA 10:1, 2024) returns continuous back-adjusted prices with no >20% single-day discontinuity at the split date, matching `tests/intraday/data/test_store_pit.py`'s float-tolerance assertion across a split boundary.

Tiingo, FMP, and Polygon may then re-enter as **sandbox** cross-vendor reconciliation probes (a true injected second `BarProvider` plus a `quality.py` reconciliation check) once the in-house path is proven — graduating to a runtime dependency only if the in-house sources prove insufficient.

## Archived (unused, retained for provenance)

These keys are moved to `.env.archived` (gitignored, beside the active `.env`) and are **not deleted**. They are recorded here so a future need can revisit the decision without rediscovering what each key was. None is wired into quant-trading today.

| Key | Provider | Why archived (one line) | Revisit trigger |
| --- | --- | --- | --- |
| `ALPHA_VANTAGE_API_KEY` | Alpha Vantage | Every capability duplicated by a better free incumbent (EDGAR/FRED/yfinance/Alpaca); non-PIT fundamentals would inject look-ahead; 25 req/day free tier too thin. | Never likely — strictly dominated by EDGAR+FRED. |
| `NEWSAPI_KEY` | NewsAPI.org | Free tier dev-only/24h-delayed/1-month; production $449/mo; redundant with free Alpaca `get_news`; no concrete consumer. | Only if a costed news-research plan emerges — and even then use Alpaca `get_news` first. |
| `OPEN_FIGI_API_KEY` | OpenFIGI (Bloomberg) | Reference/identifier data, not market data; ticker→CIK already free via EDGAR; fills no gap. | A non-US / multi-listing / ISIN/CUSIP universe requirement. |
| `X_RAPIDAPI_KEY` | RapidAPI gateway | Generic gateway, underlying API unidentifiable (no `X-RapidAPI-Host`, zero code refs); almost certainly repackages free data behind a paid middleman. | Only if a specific fronted API with a pinned host is ever chosen. |
| `FX_MACRO_DATA_API_KEY` | FXMacroData | FX + non-US macro; only relevant slice (USD macro) is FRED-sourced and redundant with the existing FRED feed; no equity coverage. | A non-US macro factor or FX trading mandate. |
| `TRADE_WATCH_API_KEY` | TradeWatch | Crypto/forex-first; no equities gaps covered; free tier (1 req/min, daily-only) unusable; not even declared in `config.py` Settings. | Never likely — dominated by yfinance/Alpaca for equities. |
| `AUTHORIZATION` | None (internal token) | Not a vendor — the news-dashboard's own `Authorization: Bearer` self-auth header (`DASHBOARD_TOKEN`); carries no market data. | N/A — belongs to the news-dashboard, not quant-trading. |