# Raise-the-Ceiling Roadmap (2026-06-02)

A dependency-aware, **value × safety**-sequenced plan for evolving the system from
"advanced personal quant, now live on Alpaca paper" toward the institutional /
autonomous-analyst vision — **without ever de-authorizing the one live strategy.**

Authored from a 5-agent design workflow (portfolio-risk gates, execution realism,
alpha/universe, Claude decision-maker B–E) cross-checked against the codebase.
This is a planning document; nothing here is enabled until the gates below say so.

---

## Where we are (as of 2026-06-02, end of session)

- **LIVE on Alpaca paper.** Cutover done; first autonomous rebalance filled at
  15:55 ET (`DBC/EEM/GLD`, ~$1.0M book). Only `defensive-etf-allocation` is
  governance-LIVE; the other 5 are quarantined by the evidence battery.
- **Shipped this session** (branch `feat/m4-deploy-e1`):
  - `a84bd45` equity-health guardrail + live broker equity (closed the `$0`-scored-ok blind spot)
  - `ac8e489` fail-closed pre-trade risk gate (Guard 4) + non-silent submit failures
  - `3244a79` paper-live cutover (jobs.toml + guard plist off `--dry-run`)
  - `d082a17` Claude decision-maker **Phase A** (read-only structured brief)
  - `4c63569` portfolio risk module (VaR/CVaR/vol/beta) — read-only analysis
  - `e375d48` pairs β-neutral sizing + `adf_p_max` gate (quarantined)
  - `6073509` Claude decision-maker **Phase B** (advise-and-log, governance-clamped, applies nothing)
  - `2668b8b` daily auto-brief job (17:50 ET)

## Governing safety principles (non-negotiable)

1. **Governance is supreme.** Nothing trades live except by passing the deterministic
   evidence gates. The Claude layer may *advise/veto within* governance — never bypass it.
2. **The launchd agents run from the working tree.** Any change takes effect on the next
   scheduled job. Therefore every commit keeps the suite green, and order-path/governance
   changes are throttled behind flags + bake-in windows.
3. **Never de-authorize the live strategy by accident.** See the DSR trap below.
4. **Asymmetric Claude authority.** The decision-maker can only ever make the book *safer*
   autonomously (one-way de-risk); anything risk-increasing needs a human.
5. **Fail-open on monitoring gaps, fail-closed on the order path.**

## ⚠️ The DSR one-way trap (read before touching validation)

`quant/backtest/validation.py:206` feeds `deflated_sharpe` the CPCV path count
(`C(6,2)=15`), **not** the true walk-forward grid-search cardinality (~152 for
defensive-etf). Correcting this is a real fix — but the design agents **empirically
estimate it drops `defensive-etf` DSR from ~0.602 to ~0.246, below the 0.30 gate.**
Because `quant/governance/refresh.py` reads persisted gate booleans on the weekly
refresh and the guard auto-halt is one-way, landing the corrected code on the live
tree could **quarantine the only live strategy and halt trading** at the next
Saturday `governance refresh`. This is why the DSR fix was **deliberately NOT made
autonomously** and is sequenced first behind a shield + an off-tree re-validation +
a human go/no-go (Phase 0).

---

## Sequenced phases

### Phase 0 — Disarm the DSR trap (prerequisite for anything governance-adjacent)
**Goal:** make it impossible for a math/code change alone to de-authorize the live
strategy, and learn its true standing under the corrected count *off the live tree*.
- **Evidence-schema-version shield**: add `evidence_schema_version`/`trial_count`/
  `dsr_threshold_used` to `validation_report.json` + `ValidationEvidence`; treat
  old-schema evidence for the *currently-live* slug as authoritative until it has a
  fresh passing report under the new schema. Regression test: a recompute never flips
  the live slug LIVE→QUARANTINED. No live behavior change. **Land first.**
- **Off-tree manual re-validation**: on a branch, run `quant validate
  defensive-etf-allocation` with a temp data/out dir the scheduler can't pick up, to
  learn the true corrected DSR vs the 0.30 gate.
- **Correct the DSR trial count** to grid-search cardinality (capture per-trial Sharpes
  + grid size in `WalkforwardResult`; keep CPCV for robustness reporting). **Must not
  reach the launchd tree until the re-validation result is human-reviewed.**

### Phase 1 — Pure, order-path-free foundations (low-risk, parallelizable)
- Extend `PortfolioRisk`: parametric VaR, CVaR, sector/asset-class exposure (`SECTOR_MAP`),
  and a `computable`/`degraded_metrics` fail-state so a future gate can tell "within
  limits" from "couldn't compute". Keep `render()` backward-compatible.
- `PortfolioRiskLimits` dataclass + `RiskGateMode {WARN,BLOCK}` + `fail_closed_on_uncomputable`,
  env-overridable via Settings, **default WARN** + generous limits calibrated to the
  defensive sleeve (per-sector cap ≥ 1.0 — risk-off is 100% defensive by design).
- Pure `build_portfolio_risk_gate()` over post-trade holdings (reconstructed from
  positions + netted deltas); WARN records would-be violations without blocking.
- `OrderTemplate` gains `order_type`/`limit_price`/`time_in_force` — **defaults reproduce
  today's market+DAY byte-for-byte**, COID unchanged, behind a per-strategy flag (OFF).
- Survivorship-bias-free, liquidity-screened ~300–500 name universe (research-only param;
  live ETF universe untouched; fetched in the data job, never the 60s tick).
- Transaction-cost/capacity awareness in `size_to_shares` (ADV cap + no-trade band),
  permissive defaults regression-pinned to leave defensive-etf sizing byte-identical.
- **Phase B** proposal emitter + deterministic clamp library *(✅ shipped this session as
  `quant analyst propose`; the roadmap's `proposals.py`/`clamps.py` split is a refactor).*

### Phase 2 — WARN-mode observability on the live path (visible, non-enforcing)
- Scenario/stress-shock evaluation (`quant/risk/scenarios.py`) as a separate WARN metric.
- Wire the portfolio risk gate into `run_rebalance` as **Guard 5**, WARN-only, behind
  `try/except-continue` (two independent fail-open guards), writing a per-run artifact and
  a `CheckResult` — **never mutates `netted`**.
- Surface it in the read-only brief + a `quant risk gate` CLI. **Bake in across a full
  monthly rebalance** (success = defensive-etf records ok every run) before any BLOCK flip.

### Phase 3 — Research-only alpha + off-live Claude loops (gate-bound, never live)
- Beta/sector neutralization of the multi-factor composite (params default OFF).
- Expanded, sector-balanced pairs discovery from the liquid universe (pairs quarantined).
- Risk-model-aware (IC/covariance) factor combination (Ledoit-Wolf reuse; default 'equal').
- **Phase E** research-synthesis loop: Claude proposes *experiments* for quarantined
  strategies only, caps grid cardinality and **displays the implied DSR-bar cost**;
  promotion stays 100% deterministic. Hard wall: no governance write-path import.
- **Phase D** human-approved allocation-tilt machinery (within the 0.40/0.05 caps) —
  ship the code; it's a guaranteed no-op while only one strategy is live.

### Phase 4 — Order-path enforcement flips (HUMAN-GATED, one cohort per cycle)
- Risk gate **WARN→BLOCK** (per-metric order: vol/VaR/CVaR → beta → sector → stress last),
  BLOCK branch mirrors the proven Guard 4 (skips the batch only; never halts/de-authorizes);
  separate commit from any governance change; rollback = `mode=WARN`.
- Execution: TWAP/VWAP/POV child-slicing (fail-closed participation caps, deterministic child
  COIDs); same-session reconcile for sliced/partial orders; `ExecutionPolicy` defaulting
  MARKET/single-shot; the **intraday fill manager** as a *separate supervised job* validated
  on one tiny order first (highest-risk item).
- **Phase C** one-way de-risk actuator (`gross *= clamp(posture,0,1)`, fail-open to 1.0);
  flip to 'C' only after weeks of clean Phase-B logs.

---

## Requires explicit human sign-off (do NOT do autonomously)

1. **Corrected-DSR code reaching the live tree / scheduled refresh** — only after the
   schema shield is proven AND the off-tree re-validation is human-reviewed. If defensive-etf
   genuinely fails the corrected count, that's a real governance finding (deliberate pause vs
   documented degraded-evidence override) — **never** resolved by an autonomous merge or by
   lowering the 0.30 threshold.
2. **Flipping the risk gate WARN→BLOCK** — needs a full clean monthly WARN bake-in; separate
   commit from any governance change.
3. **Flipping any execution flag to LIMIT/TWAP/VWAP for the live strategy, or adding the
   fill-manager job to `jobs.toml`** — a committed jobs.toml entry starts placing intraday
   orders on the next tick; the fill manager must be validated on one tiny live order first.
4. **Raising `claude_actuation_phase` above 'A'** — Phase C needs weeks of clean Phase-B logs;
   Phase D is meaningless until a second strategy passes the gates.
5. **Changing any live `spec.universe` / `enable_live`** — all alpha/universe work is
   research-only and reaches live only via the deterministic battery, triggered by a human;
   defensive-etf `target_positions` must be regression-pinned byte-identical before merge.

---

*Full per-item designs (effort/risk/files/approach for all four dimensions) are in the
workflow output for run `wf_02be5e38-42c`. This roadmap supersedes the audit's high-level
phases with verified code anchors and the DSR-trap sequencing.*
