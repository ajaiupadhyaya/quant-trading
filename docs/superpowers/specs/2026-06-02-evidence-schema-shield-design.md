# Evidence-Schema-Version Shield — design spec (Phase 0, step 1)

Date: 2026-06-02. Branch: `feat/dsr-schema-shield` (off `feat/m4-deploy-e1`).
Roadmap anchor: `docs/specs/2026-06-02-raise-the-ceiling-roadmap.md` Phase 0.

## Problem (the DSR one-way trap)

`quant/backtest/validation.py` feeds Deflated Sharpe the CPCV path count
(`C(6,2)=15`) as its trial count, not the true walk-forward grid-search
cardinality (~152 for defensive-etf). Correcting this is a legitimate fix but
empirically drops `defensive-etf-allocation` DSR from ~0.602 to ~0.246, **below
the 0.30 gate** — flipping `gate_deflated_sharpe` True→False.

`defensive-etf-allocation` is the **only** governance-LIVE strategy. The Saturday
`weekly-validation-governance` job runs `__validate_matrix__` → `governance
refresh`: it regenerates each `validation_report.json` sidecar with whatever DSR
math is on the tree, then re-reads the sidecars and reclassifies, writing
`strategy_states.json`. The next `daily-rebalance` is fail-closed (only `state ==
LIVE` trades). So landing the corrected DSR code would **silently quarantine the
sole live strategy and halt all trading at the next refresh, with no human in the
loop.** That autonomous de-authorization is exactly what this shield prevents.

## Goal

Make it **impossible for a code/math change alone** to flip the currently-LIVE
incumbent LIVE→QUARANTINED at the automated refresh, while:

- **zero live behavior change today** (the DSR fix is NOT landed here; the shield
  is dormant until a real schema bump),
- **never a silent permanent governance bypass** (incumbent-only, never promotes,
  loud, time-bounded, fail-closed),
- smallest safe diff (this is a "land first" safety commit on a live system).

## Design (scoped core)

A pure **tripwire** evaluated AFTER the existing classifier produces its
provisional decision. No new governance module.

1. **Schema stamp co-located with the math.** New constant
   `EVIDENCE_SCHEMA_VERSION: int = 1` in `quant/backtest/validation.py` (next to
   `THRESHOLDS`). `quant validate` stamps it into every fresh sidecar; the future,
   separately-human-gated DSR fix bumps it to `2` **in the same commit that
   changes the math.** Readers default an absent key to `1` (today's stamp-free
   defensive-etf sidecar).

2. **New, additive, defaulted fields** (all keep the suite + mypy green):
   - `ValidationEvidence.evidence_schema_version: int = 1`
   - `StrategyState.shielded: bool = False`
   - `StrategyState.shield_consecutive: int = 0` (display only)
   - `StrategyState.evidence_schema_version: int = 1` (the **blessed** version)
   - `StrategyState.shield_first_at: date | None = None` (calendar-wall anchor)

3. **Prior state threaded in.** `build_governance_artifacts` loads the prior
   `strategy_states.json` once before its loop (absent → `{}`; **present-but-
   malformed → GovernanceError propagates / fail loud**, never silently degraded
   to "no incumbent"). Each prior `StrategyState` is passed to a new pure helper
   `apply_schema_shield(provisional, *, evidence, asof, prior_state)`.
   `classify_strategy` gains a keyword-only `prior_state=None` only to stamp the
   new default fields; the shield logic lives in the helper.

4. **Shield predicate — fires (retains LIVE) iff ALL of:**
   1. provisional state is `QUARANTINED` (only ever salvage; never promote),
   2. quarantine cause is **gate-failures only** (every reason code starts
      `failed_gate_`; any non-gate reason — `manual_block`, `stale_validation`,
      `future_validation_date`, `evidence_slug_mismatch`, missing artifacts,
      `missing_validation` — blocks the shield),
   3. `prior_state` exists and `prior_state.state is LIVE` (incumbent-only),
   4. **genuine schema bump:** `evidence.evidence_schema_version >
      prior_state.evidence_schema_version` (strict; same-schema decay and
      downgrades quarantine normally — this is the self-disarm against masking
      ordinary alpha decay),
   5. **calendar wall not exhausted:** `asof - shield_first_at <
      MAX_SHIELD_CALENDAR_DAYS` (first fire: `shield_first_at = asof`).

   On fire: `state = LIVE`; `reason_codes = ["schema_shield_retained_live",
   *gate_failures]`; `shielded = True`; `shield_first_at` carried from prior (or
   `asof`); `shield_consecutive += 1`; **`evidence_schema_version` keeps the prior
   BLESSED version** (so subsequent refreshes still see new > blessed and keep
   protecting until re-blessed or the wall).

   Wall exhausted (1–4 hold, 5 fails): `state = QUARANTINED`; reason codes
   `["shield_backstop_exhausted", *gate_failures]`; `shielded = False`. Trading
   stops — the safe default.

5. **Lift conditions:** honest re-bless (a fresh new-schema sidecar that PASSES →
   provisional LIVE → shield is a no-op, fields reset, `evidence_schema_version`
   stamped to the new version); same-schema failure; the calendar wall; any
   non-gate reason; lost incumbency.

### Why a 30-day calendar wall (not single-shot / refresh-cap)

A low refresh cap would **prematurely** quarantine the live strategy when
same-day `SAME_DAY`/`CATCH_UP` replays advance the count faster than calendar
time — the opposite of the goal. The bound must be wall-clock. `MAX_SHIELD_
CALENDAR_DAYS = 30` mirrors the existing `GovernancePolicy.max_validation_age_days
= 30` ("evidence this old isn't trusted"); the only reason staleness can't serve
as the bound here is that `run_date` resets every weekly run, whereas
`shield_first_at` does not. 30 days = ample human-decision window, then fail-safe.

This shield does **not** contradict "no autonomous de-authorization": it forbids
the *silent same-refresh* flip the roadmap names, and the wall only fails safe
after 30 days of a loud, unresolved shielded state.

## Zero-behavior-change-today proof

Today: `strategy_states.json` v1 has defensive-etf `state=live reason_codes=[]`;
its sidecar has DSR 0.602, all gates True, no `evidence_schema_version` key.
On the next refresh on this tree: `EVIDENCE_SCHEMA_VERSION` is still `1`, so fresh
sidecars stamp 1 and the old sidecar defaults to 1 → identical gate math →
provisional LIVE → predicate condition 1 (`QUARANTINED`) is FALSE → shield never
engages → identical `StrategyState` plus additive default fields. The five
quarantined slugs have `prior_state.state == QUARANTINED` → condition 3 FALSE →
byte-identical. `allocate_capital` is **unchanged** (single-live → `{slug: 1.0}`).
Locked by `test_build_governance_artifacts_today_no_change_idempotent`.

## Explicitly deferred (documented follow-ups, NOT in this commit)

Captured from the adversarial design review (`wf_8784ea6d-0a3`). These harden the
shield further but are **not required** for a correct, non-bypass, loud, bounded
Phase-0 shield, and each adds live-path surface area better landed separately —
critically, **all of these must be revisited before the human-gated DSR fix
actually lands**, since that is when the shield first becomes active:

- **Durable `manual_block`** (governance-owned `manual_blocks.json` +
  `governance block/unblock` CLI). Today the sidecar writer never emits
  `manual_block`, so the override is near-vestigial; the shield correctly excludes
  `manual_block` from gate-only reasons, but making a human block *durable across
  `__validate_matrix__`* is its own task. **Prerequisite before the DSR fix lands.**
- **Separate monotonic `shield_ledger.json`** (belt-and-suspenders against a
  git-rolled-back / restored `strategy_states.json` rewinding the wall anchor).
  The calendar wall on `StrategyState` + atomic writes are the primary bound;
  rollback of committed governance state is a deliberate exceptional human action.
- **`identity_fingerprint`** (spec name + chosen-params hash) to defeat slug
  reuse. Mitigated for now by incumbent-only + a runbook rule: *retiring a slug
  must clear its `strategy_states.json` entry.*
- **Allocation `shielded_haircut` (0.5×)** blast-radius reduction while shielded.
  Dormant today (nothing is shielded); good to add before the DSR fix lands.
- **Active paging** (`AlertClient.send_emergency`) from the weekly chain on any
  `shielded=True`, plus surfacing `.shielded` in `analyst/context.py`, `audit.py`,
  and the doctor governance helper. The reason code + `governance status` marker
  provide visibility today; active paging is an enhancement.

## Residual risks accepted for Phase 0

- The shield only **defers** the quarantine of a genuinely-failing incumbent (up
  to 30 days) while making it loud — by design; the point is to force a human
  decision, not keep a dead strategy alive. The wall guarantees finite shielded
  life.
- A deliberate git rollback of committed governance state could reset the wall
  anchor (re-granting a window). Out of scope for the automated-path threat
  model; the ledger follow-up closes it.
- Slug-reuse laundering is mitigated by incumbent-only + the runbook retire rule
  until `identity_fingerprint` lands.

## Test plan

Unit (`tests/governance/test_policy.py`): shield retains on bump+gate-fail;
self-disarm on same-schema fail; calendar-wall denial; inert when gates pass under
new schema; non-gate reason blocks shield; never promotes (prior
QUARANTINED/RESEARCH/None); schema-downgrade refusal; `classify_strategy` default
`prior_state=None` unchanged.
Store (`tests/governance/test_store.py`): round-trip new fields (non-default);
legacy file loads defaults; present-but-malformed/bool-as-int raises; atomic write
leaves prior file intact on writer failure; manifest schema-version round-trip.
E2E (`tests/governance/test_refresh.py`): today-no-change idempotent (incl. each
quarantined slug's exact reason_codes); shield-fires-e2e through
`build_governance_artifacts`; fail-loud on malformed prior; absent-prior
quarantines a failing bump. CLI: sidecar carries `evidence_schema_version`;
`governance status` renders the SHIELDED marker.
