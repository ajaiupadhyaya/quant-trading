# Phase 1 — Trust Audit (design)

**Date:** 2026-06-19
**Parent:** `docs/specs/2026-06-19-completion-roadmap.md`
**Status:** Approved design — ready for implementation plan.

---

## Goal

Produce one evidence-backed, triaged punch-list that classifies every load-bearing claim in
`quant-trading` as **VERIFIED-REAL / BROKEN / FAKE-STUB / UNVERIFIED**, each with a reproduction
command and a severity. This is a **diagnosis** phase: it fixes only trivial/blocking items inline
and logs everything substantive for later phases. The punch-list re-plans Phases 2–4.

### Output artifact

`docs/audits/2026-06-19-honesty-audit.md`, structured as:

- **Executive summary** — counts per verdict, top 5 risks, "is the live system actually trading?"
  yes/no with evidence.
- **Findings table** — one row per claim: `id | area | claim | verdict | severity | evidence
  (file:line or command + output) | proposed phase to fix`.
- **Reproduction appendix** — exact commands so any verdict can be independently re-run.

Severity scale: **S1** (system is lying / live path broken / data leak in a live strategy) →
**S2** (module fake or unreproducible but not live) → **S3** (cosmetic / docs drift / minor TODO).

## Non-goals

- No substantive fixes. A fake module or unreproducible backtest is **logged**, not repaired here.
- No new features, no refactors. Pure verification.
- No changes to the live M4 (read-only inspection only).

---

## Step 0 — Consolidation finish (quick)

`origin/main` fast-forward is already done. Remaining: if `.gitignore` is trivially missing the
generated run-artifact dirs (`data/snapshots/`, `data/governance/*.json`, `data/risk/*.json`,
`docs/analyst/`, `docs/live-recon/`, `.coverage`, `context.md`), add them so the tree is clean. This
is the one inline fix allowed up front. If non-trivial, log it instead.

## The five audit dimensions

Each dimension is run by a dedicated subagent (read-heavy, independent → parallel fan-out via the
`dispatching-parallel-agents` pattern). Each returns **structured findings** (the findings-table row
shape above). The orchestrator then **independently verifies** each non-trivial finding before it
goes in the doc — no verdict is trusted on an agent's say-so alone.

### D1 — Build health
Run on THIS box and record real numbers (not claimed):
- `uv sync` clean?
- full `pytest` suite: pass/fail/skip counts, and which tests are skipped and why (e.g. torch-gated).
- `ruff check` clean? `mypy` clean?
- coverage actual %.
Verdict per item with the literal command output captured.

### D2 — Stub / fake hunt
Grep the entire `quant/` tree (exclude tests) for: `NotImplementedError`, `raise NotImplemented`,
`pass  # …`, `# TODO`/`# FIXME`/`# HACK`/`# XXX`, `placeholder`, `dummy`, `stub`, `return 0.0`/
`return None` on signal/order paths, `random`/`np.random`/synthetic-series used where real data is
implied, and any order/signal function that can silently no-op. Each hit triaged: real concern vs
benign. Special attention to `quant/nlp/` (module E — suspected half-built) and the intraday
showcase wiring.

### D3 — Backtest honesty
For each strategy in `quant/strategies/`:
- Re-run its validation and check the committed `data/backtests/<strat>/validation_report.json` and
  `chosen_params.json` **reproduce** (within tolerance).
- Confirm leak-guards (PIT), the cost model (slippage/commission/borrow/impact), and DSR/PBO gating
  are actually applied — not bypassed.
- Flag any strategy whose live-claimed Sharpe/alpha cannot be regenerated from the repo.
Focus on the **live roster** (defensive-etf, trend) first — those are S1 if they don't reproduce.

### D4 — Live-path reality
The central question: *does the live engine actually place Alpaca paper orders, or is it a
dry-run/no-op?*
- Read the live code path (`quant/live/`, `quant/engine/`, the launchd plists in `deploy/`) and
  trace from scheduled job → signal → order submission. Identify any `dry_run`/shadow flag that
  would suppress real orders.
- Pull the **real Alpaca paper account** state read-only via the `alpaca-paper` MCP tools: recent
  orders, fills, positions, account activity. Reconcile against `docs/live-recon/`.
- Verdict: is there hard evidence (actual filled orders, or a documented reason for zero) that the
  system trades? The recent recon showed 0 trades in-window — determine whether that's "correctly
  no signal" vs "the order path is dead."

### D5 — Determinism / reproducibility
Spot-check the claims: RNG seeding across backtests, "deterministic governance manifests,"
intraday-DL determinism flag restoration. Re-run one governance manifest twice and diff. Confirm
seeds are set where stochastic methods run.

## Method & verification of the audit itself

- Parallel subagents for D1–D5; orchestrator re-verifies each non-trivial finding by re-running the
  cited command or reading the cited file:line.
- **Every verdict carries evidence.** VERIFIED-REAL → a re-runnable command. BROKEN/FAKE → exact
  file:line + failing output. No evidence ⇒ verdict is UNVERIFIED, not REAL.
- The audit doc is the deliverable; it is committed. No code changes except the Step-0 `.gitignore`
  quick-win (and any one-line blocking test fix, logged in the doc).

## Definition of done

- `docs/audits/2026-06-19-honesty-audit.md` exists, committed, with every dimension covered and
  every finding carrying evidence + a target phase.
- The executive summary answers, with evidence: (a) does it build/test green here, (b) is any live
  strategy's backtest unreproducible, (c) **is the live system actually trading?**
- Tree is clean (gitignore sweep done or logged).
- We checkpoint: review the punch-list together, then it drives the Phase 2 spec.
