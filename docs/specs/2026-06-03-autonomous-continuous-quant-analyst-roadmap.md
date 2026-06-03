# Autonomous Continuous Quant Analyst — Roadmap

_Author: autonomous build session, 2026-06-03. Status: PLAN (no live-path changes from this doc itself)._

## 0. The vision (user's words, distilled)

A **continuous, autonomous quant analyst/trader** that runs **all day, every day** — not a
set of cron jobs that fire once at a fixed time — employing the highest level of
**Statistics, Mathematics, Economics, Finance, Politics, Computer Science, and Data Science**.
It is **trained on industry-standard methods + historical data + as many data sources as
possible** and makes **fast, data-driven, algorithmic/model decisions autonomously**. It is
**NOT reliant on the Claude API for every decision** — the deterministic quant engine makes the
routine decisions; Claude is reserved for **impactful judgments and summaries**.

### Honest framing (so the goal is real, not marketing)
- **"Ultra-fast" on this stack = seconds-to-minutes, not microseconds.** Hardware is a Mac mini
  (M4), execution is Alpaca paper REST, data is daily + minute bars + free macro/fundamentals.
  Literal HFT (sub-ms, colocation, L2 book) is out of scope and not the goal. The realistic and
  fully-worthwhile ceiling is a **sophisticated, always-on systematic quant desk** that reacts
  within the session.
- **The two-layer principle is already the architecture and is correct.** Tier-0 deterministic
  quant makes ~all decisions; Claude is advisory/event-driven. We keep it that way.
- **ML does not get a free pass.** Every model must clear the same DSR/PSR/CPCV/bootstrap/regime
  gauntlet the strategies do. The governance system is the moat against overfit garbage. We never
  lower a threshold to force something live (see the DSR one-way-trap rule).

---

## 1. Current state (honest inventory, 2026-06-03)

**What exists and works (this is already an upper-tier personal quant stack):**
- **Deterministic quant machinery:** Ledoit-Wolf / HRP sizing, Baum-Welch HMM + Kalman regime
  detection, DSR/PSR/CPCV/bootstrap validation, walk-forward, BSM options Greeks, portfolio risk
  (historical VaR/CVaR/vol/beta).
- **Governance:** evidence gates decide which strategies are live; evidence-schema shield; circuit
  breakers (equity-health, fail-closed pretrade Guard 4, portfolio-risk WARN Guard 5).
- **Signals engine (shipped today, `f992818`):** trailing-only battery (momentum/trend/breadth,
  vol term-structure/VIX/VRP, correlation/dispersion, RSI/z-scores, drawdown, curve) + a composite
  risk-posture score, fed read-only into the Claude analyst.
- **Claude overlay:** digest / brief / intraday watch / shadow propose — all read-only, event-light,
  cost-controlled (Haiku routine, Opus high-stakes), immutable audit log.
- **Deployment:** M4 is the sole 24/7 executor — launchd tick (60s) + guard daemon (KeepAlive).
  One live strategy (`defensive-etf-allocation`); 5 quarantined by gates.
- **Latent assets not yet in the live loop:** an **event-driven intraday scaffold**
  (`quant/intraday/`: `IntradayStrategy.on_event`, `StrategyContext`, `Order`/`QuoteBar`), and
  **EDGAR + fundamentals loaders** (`quant/data/edgar.py`, `fundamentals.py`).

**The core gaps vs the vision:**
1. **Architecture is discrete, not continuous.** The tick fires scheduled jobs at fixed times.
   There is no always-on engine maintaining a live market state and reacting within the session.
   _This is the #1 thing to close._
2. **Data is narrow in the live loop.** Daily bars + ~3 FRED series. No intraday bars driving
   decisions, no live fundamentals/factors, no news/sentiment (NLP), no political/event-risk data,
   no options/vol-surface signals.
3. **No ML forecasting layer.** Strategies are rule-based; the HMM is the only fit model. No
   return/vol forecasting, no factor model, no validated ensemble.
4. **Discipline coverage is partial** (see matrix).

### Discipline coverage matrix
| Discipline | Today | Target additions |
|---|---|---|
| Statistics | DSR/PSR/CPCV/bootstrap, HMM, Kalman, z-scores, correlation | GARCH/HAR vol, Bayesian change-point, cointegration suite, online/rolling estimators |
| Mathematics | Ledoit-Wolf, HRP, BSM Greeks | Convex portfolio optimization, regularization, stochastic-vol calibration |
| Economics | FRED VIX/10y/2y | Nowcasting (recession/cycle), full curve, credit spreads, financial-conditions index, PMI/claims/breakevens |
| Finance | risk-parity, multi-factor (quarantined), options modules | Live factor model, multi-strategy optimizer, systematic tail hedge, TCA |
| Politics | none | Economic/event calendar (FOMC/CPI/NFP/elections), policy-uncertainty (EPU) + geopolitical-risk (GPR) indices, event-window risk rules |
| Computer Science | launchd tick/guard, governance, atomic IO, intraday event scaffold | Always-on supervised engine, event bus, streaming feature store, execution algos |
| Data Science | regime features, signals battery, walk-forward | Feature store, ML forecasting (purged CV), NLP sentiment, model registry + retraining |

---

## 2. Target architecture — the continuous engine

A long-running, supervised **`quant engine` daemon** (a sibling launchd agent to tick/guard),
always on, with a market-hours-aware cadence:

```
                         ┌─────────────────────────────────────────────┐
   data sources  ─────▶  │  INGEST (bars/minute, macro, fundamentals,   │
   (bars, FRED,          │  news, options, calendar)  → feature store    │
    EDGAR, news,         └─────────────────────────────────────────────┘
    calendar, IV)                          │
                         ┌─────────────────▼───────────────────────────┐
                         │  MODELS (regime, vol-forecast, factor/return, │   Tier 0
                         │  ensemble) — each gate-validated              │   DETERMINISTIC
                         └─────────────────┬───────────────────────────┘   (makes ~all
                         ┌─────────────────▼───────────────────────────┐    decisions, no
                         │  MARKET STATE  (signals + regime + risk +     │    Claude)
                         │  vol + posture)  → state.json + state.jsonl   │
                         └────────┬──────────────────────┬──────────────┘
                  deterministic   │                      │  material change?
                  rules           ▼                      ▼
                         ┌─────────────────┐    ┌─────────────────────────┐
                         │  STRATEGIES →    │    │  EVENT BUS              │
                         │  PORTFOLIO →     │    │  (regime flip, vol spike,│ Tier 1
                         │  GOVERNANCE →    │    │   risk breach, anomaly)  │ GOVERNANCE
                         │  EXECUTION       │    └──────────┬──────────────┘ (gates/limits)
                         └─────────────────┘               │ impactful only
                                                           ▼
                                              ┌─────────────────────────┐
                                              │  CLAUDE (event-driven):  │  Tier 2
                                              │  judgment + summary +    │  ADVISORY
                                              │  weekly synthesis        │  (cheap, rare)
                                              └─────────────────────────┘
```

**Cadence:** during RTH, loop every ~30–60s (minute-bar resolution); pre/post-market lighter;
overnight/weekend triggers research + retraining. The loop is **read-only/advisory first** — it
emits state + events and actuates **nothing** until each actuation path is separately human-gated.

**Decision hierarchy (the "not reliant on Claude" guarantee, made explicit):**
- **Tier 0 — continuous deterministic quant:** signals, regime, vol, factor model, strategy
  targets. No Claude. This makes essentially every routine decision.
- **Tier 1 — deterministic governance:** which strategies are live, circuit breakers, risk limits,
  reconciliation. No Claude.
- **Tier 2 — Claude, event-driven & advisory:** only on impactful events (regime change, risk
  breach, anomaly) + daily/weekly summaries + weekly research synthesis. Haiku for routine, Opus
  for high-stakes. Rate-limited, deduped, prompt-cached → low cost.

---

## 3. Phased roadmap

Each phase marks **autonomy**: 🟢 safe-autonomous (read-only/shadow, I can build it),
🔴 human-gated (touches the live order path, governance flip, or real-money posture).

### Phase 6 — Continuous engine core (the "all-day" spine)  🟢 build / 🔴 actuate
- `quant engine run`: always-on supervised loop (new launchd agent under the guard's KeepAlive),
  maintaining a live `MarketState` (reuse `gather_analyst_context` + the signals engine) every
  cycle → `data/engine/state.json` (hot) + `state.jsonl` (audit).
- **Event bus:** deterministic detectors (regime flip, vol/VIX spike, breadth collapse, composite
  posture crossing a band, risk-limit approach, large intraday move) → structured events → Slack +
  a single rate-limited Claude "impactful event" call + input to the next rebalance/guard.
- Read-only/advisory: actuates nothing; isolated from tick/guard/rebalance.
- **Acceptance:** runs a full session writing fresh state each cycle; events fire on synthetic
  injections; provably zero effect on the existing live order path; shadow-observed several sessions.

### Phase 7 — Data breadth ("all data sources possible")  🟢
- **A. Intraday/minute bars** into the live loop + intraday feature store (build on `quant/intraday/`).
- **B. Fundamentals** (EDGAR/`fundamentals.py`) → value/quality/profitability factors.
- **C. Macro/econ nowcasting:** broaden FRED (full curve, BAA–AAA & HY-OAS credit spreads, financial
  conditions, PMI, jobless claims, breakevens) + a recession/cycle nowcast.
- **D. News & sentiment (NLP):** ingest free headlines/filings; score with a **local** model
  (FinBERT-class) for the continuous stream; Claude summarizes only the highest-impact items.
- **E. Politics / policy / event risk:** economic + event calendar (FOMC/CPI/NFP/elections),
  EPU policy-uncertainty + GPR geopolitical-risk indices, event-window risk-off rules.
- **F. Options / vol surface:** IV, term structure, skew (`quant/options/`) → vol-regime + tail signals.
- Each source: cached, fail-open loader + features + tests + backfilled history; feeds the signals
  engine + MarketState as **advisory** first.

### Phase 8 — Models & ML ("trained on industry-standard methods + historical data")  🟢 research / 🔴 promote
- **Vol forecasting:** GARCH/EGARCH/HAR-RV → sizing + vol-targeting.
- **Return/edge:** cross-sectional factor model (FF + momentum + quality), regularized
  (ridge/elastic-net/GBM) with **purged** CV.
- **Regime:** macro-conditioned HMM + Bayesian online change-point detector.
- **Ensemble:** validated stacking combiner (purged CV) → strategy tilts; no naive averaging.
- **Walk-forward retraining** (nightly/weekly) = "continuously trained on historical data"; every
  retrain re-passes the gates (shadow → gated → live). Logged to the experiment registry.
- 🔴 Taking any model live requires passing the honest gates — never lower a threshold; honor the
  DSR one-way-trap rule.

### Phase 9 — Strategy & portfolio construction  🟢 research / 🔴 live
- Rehabilitate the 5 quarantined strategies via honest research (better signals/costs/neutralization).
- Multi-strategy optimizer: HRP/convex combination, regime-conditional weights, vol-targeting,
  drawdown control.
- Systematic options tail-hedge overlay in stressed regimes.

### Phase 10 — Execution realism  🔴
- Limit/TWAP/VWAP, intraday fill-manager, slippage modeling + TCA (build on the intraday scaffold).

### Phase 11 — Claude decision-layer maturation (event-driven, cheap)  🔴 to enable
- **C:** one-way de-risk actuator (after the shadow bake-in that began today) — can only REDUCE risk.
- **D:** human-approved tilts. **E:** weekly research synthesis (Opus reads registry + signals history).
- Cost discipline: Haiku continuous stream, Opus weekly/high-impact; prompt-cache; event-driven.

### Cross-cutting (always-on tracks)
- **Observability:** live dashboard/TUI (`tui.py`) of MarketState + events + P&L + model health.
- **Cost governance:** a Claude-spend budget + meter.
- **Safety governance:** every capability enters SHADOW → WARN → (human-gated) LIVE; guardrails
  (equity-health, Guard 4, Guard 5) inviolate; reconciliation; kill-switch; the engine daemon
  supervised with crash-restart + state recovery.

---

## 4. Recommended immediate next step (pending your greenlight)

**Phase 6, the read-only continuous engine.** It is the literal "all day every day" ask, it is the
spine every later phase plugs into, and it is **safe** — read-only/advisory, isolated from the live
order path, shadow-observable for several sessions before it influences anything. I would build it
incrementally (state loop → event bus → Claude escalation) with the same design+adversarial-review
discipline used for the signals engine and Guard 5.

## 5. Non-goals / honest constraints
- Not microsecond HFT (REST + daily/minute bars).
- **Paper only** unless you deliberately decide otherwise — real money is a separate, explicit gate.
- "All data sources" is bounded by free/affordable APIs; premium tick/L2/alt-data is optional and paid.
- ML must pass the honest gates; no overfit strategy goes live; thresholds are never lowered.
