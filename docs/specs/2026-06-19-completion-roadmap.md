# quant-trading — Completion Roadmap

**Date:** 2026-06-19
**Status:** Approved (master roadmap). Phase 1 spec'd separately and built first.
**Goal:** Take `quant-trading` from "claims to work / feature-complete on paper" to "provably works and runs itself."

---

## Context

`quant-trading` is a mature autonomous quant research-and-trading system (Python 3.12 / `uv` /
Click CLI), public at `github.com/ajaiupadhyaya/quant-trading`, designed to run 24/7 on an **M4 Mac
Mini** executing on **Alpaca paper**. All 10 spec strategy classes exist, the intraday showcase
(spine/execution/market-making/RL/DL) is built, and the CHARTER's methodology gaps (capacity,
GARCH/GJR, gradient-boosting) are largely closed.

The work is therefore **not** "build a half-finished system." It is closing the gap between
*claimed* completeness and *verified, self-running* completeness.

### Environment constraints (load-bearing)

- **This machine is the MacBook Pro (Apple M2 Pro) dev clone — NOT the M4.** The live host is a
  separate M4 Mac Mini reachable only by relay (operator runs M4-side steps and pastes results).
- **Single-main workflow.** Work directly on one clean `main`; no long-lived branches/worktrees.
- As of this doc, the dev box was fast-forwarded to `origin/main` @ `fe4e01e` (was 15 behind).

---

## Phases

Each phase is its own spec → plan → build cycle. We build **one phase at a time** and checkpoint
the results before spec'ing the next. The honesty audit (Phase 1) deliberately runs first because
its punch-list re-plans the phases after it.

### Phase 0 — Consolidate to one source of truth  ✅ DONE (2026-06-19)
Fast-forwarded the dev box to `origin/main`; discarded stale regenerable artifacts. Folded into the
front of Phase 1 execution. Remaining sub-item: a `.gitignore` sweep so generated run-artifacts stop
showing up as dirty (handled as a Phase 1 quick-win if trivial).

### Phase 1 — Trust: prove it's real, not vibecoded  ← BUILD FIRST
Systematic honesty audit across five dimensions (build health, stub/fake hunt, backtest honesty,
live-path reality, determinism/repro). **Output:** an evidence-backed, triaged punch-list at
`docs/audits/2026-06-19-honesty-audit.md` classifying every load-bearing claim as
VERIFIED-REAL / BROKEN / FAKE-STUB / UNVERIFIED, each with a reproduction command and severity.
Phase 1 fixes only trivial/blocking items inline; everything substantive is logged and fixed in its
proper later phase. **The punch-list re-plans Phases 2–4.** Detailed design:
`docs/specs/2026-06-19-phase1-trust-audit-design.md`.

### Phase 2 — Reliability: actually runs 24/7 on the M4
launchd jobs proven (engine/guard/tick plists exist but unverified), auto-restart on
failure/reboot, structured logging + alerting on circuit-breaker / data-feed failure, a readable
live dashboard, and a real 48-hour no-crash soak. Executed via **relay** to the M4. Scope refined by
Phase 1 findings about the live path.

### Phase 3 — Finish the half-built pieces
Complete NLP module E (sentiment → alpha), wire the intraday showcase (0/A/B/C/D) into something
usable, and close residual CHARTER items (RNG-seeding audit, ARIMA conditional-mean decision), plus
any stubs surfaced by Phase 1. Scope is driven by the Phase 1 punch-list.

### Phase 4 — Performance: expand the live roster with honest edge
Research to get more strategies past the honest validation gates (DSR/PBO). Trend is already
rehabilitated-to-live on origin; live roster today is defensive-etf + trend. Grow the roster only
where edge is statistically real. Sequenced last so performance numbers come from a verified,
running system.

## Ordering rationale

Verify before you build (1 before 3/4); get it running-and-watchable (2) before trusting
performance numbers from it (4). Phase 1's findings are allowed to rewrite the scope of 2–4.

## Out of scope (YAGNI)

- Real-money/live (non-paper) trading — structurally deferred.
- New broker integrations beyond Alpaca.
- Rewrites of modules that Phase 1 verifies as real and correct.
