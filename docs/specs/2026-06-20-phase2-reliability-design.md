# Phase 2 — Reliability: actually runs 24/7 on the M4 (design)

**Date:** 2026-06-20
**Parent:** `docs/specs/2026-06-19-completion-roadmap.md`
**Predecessor:** `docs/audits/2026-06-19-honesty-audit.md` (Phase 1 punch-list)
**Status:** Draft design — for checkpoint before the implementation plan.

---

## Goal

Move `quant-trading` from "the live order path is proven real" (Phase 1 result) to "the system
**provably runs itself** 24/7 on the M4 and **tells us when it doesn't.**" Phase 1 verified the
trading is real; it also showed the *reliability surface is largely unproven*: `KeepAlive` is set but
auto-restart was never demonstrated, reboot recovery is partly manual, alerting silently no-ops when
misconfigured, a hung job can hold the batch lock forever, logging is unstructured, and there is no
ops dashboard or soak evidence.

**Definition of done:** the M4 survives a deliberate crash and a reboot with all agents
auto-recovering; every failure mode pages the operator (proven by firing each one); a 48-hour soak
runs with zero unplanned restarts and a committed soak report; and ops state (halt / heartbeat /
recon / scheduler) is visible at a glance.

## Environment constraint (load-bearing)

This box is the **MacBook Pro dev clone — NOT the M4.** Therefore Phase 2 splits cleanly:

- **Build-here (dev clone):** all the *code and tooling* that makes reliability detectable,
  enforced, observable, and testable. Fully unit-testable here.
- **Prove-on-M4 (relay):** the *physical* facts — actual launchd respawn, reboot survival, live
  alert delivery, the 48h soak. The operator runs these M4-side and pastes results; this spec ships
  the exact runbook + harness so those steps are mechanical.

The **one-executor invariant** (M4 vs. any cloud cron must never both trade) is preserved throughout;
`tests/deploy/test_migration_fidelity.py` already guards it and nothing here relaxes it.

---

## Workstreams

Grouped by build-here vs. prove-on-M4. Each is a plan-task candidate; the plan will sequence + TDD them.

### A. Reliability self-check — `quant ops selfcheck`  *(build-here)*
Today `quant doctor` checks trading *readiness* (keys, data, connectivity) but nothing about the
reliability surface. Add a dedicated `quant ops selfcheck` (and/or extend `doctor`) that validates,
read-only:
- launchd agents loaded + last exit status + restart count (parse `launchctl print`).
- alerting **configured** for the active profile (see B) — fail if a live host has no
  Pushover/healthchecks creds.
- newsyslog rotation conf installed and covering **all** agent logs (engine + intraday too).
- `pmset` reboot settings present (`autorestart 1`, `disablesleep 1`).
- disk free above a floor; log dir not unbounded.
Exit non-zero on any failure. This is what makes "auto-restart works" and "alerting is wired"
*checkable* instead of assumed. Pure check functions (testable here); the `launchctl`/`pmset` probes
are thin impure shells that no-op-skip when not on a configured host.

### B. Alerting hardening + self-test  *(build-here)*
Close the holes the audit found (alerts.py no-ops silently; launchd-spawn failure uncovered;
emergency-send failures swallowed):
- **Startup config assertion:** a "live profile requires alerting" guard — if the host is configured
  to trade live but no alert channel resolves, fail loudly at install/selfcheck time (not silently
  at 3am). Implemented as a pure `resolve_alert_config()` returning configured/missing channels.
- **`quant ops alert-test`:** fires each configured channel (Pushover test page, healthchecks ping,
  Slack line) so the operator proves delivery *before* relying on it — replaces the README's
  unchecked "test alert channels" checkbox with a command.
- **Dead-man's-switch as mandatory backstop:** document + assert that healthchecks.io is set, since
  it is the *only* thing that catches a launchd-spawn failure (Python never starts → no in-app ping).
- Escalation note: emergency-send remains best-effort but logs at ERROR and the dead-man's-switch
  covers total-alerting-failure.

### C. Job timeout enforcement  *(build-here — genuine bug)*
`max_runtime_s` is parsed from `jobs.toml` but **never enforced** — a hung job holds the batch
`flock` indefinitely. Wire it into the dispatcher: `subprocess.run(..., timeout=max_runtime_s)`, and
on `TimeoutExpired` kill the process group, record `any_failure`, fire `ping_fail`, write **no**
marker (so catch-up retries within the window). Unit-testable with a sleeping fake command. This
directly protects the 60s tick cadence and the timing-critical rebalance lock.

### D. Structured logging + full rotation  *(build-here)*
- Add a structured (JSON) loguru sink for file output, **ANSI-free** (color only on TTY/stderr), so
  log files are grep/jq-able and don't leak escape codes.
- Extend `deploy/newsyslog/quant-deploy.conf` to rotate `engine.*` and `intraday-live.*` (currently
  unbounded).
- Guard-status **history**: append-only JSONL (`data/ops/monitor_status.jsonl`, host-local,
  gitignored) alongside the overwrite-only `monitor_status.json`, so the dashboard + soak can show a
  time-series of guard health instead of a single point.

### E. Live ops dashboard  *(build-here — DECISION below)*
The current `quant monitor` TUI is a portfolio viewer that omits halt / heartbeat / recon /
healthcheck — it is not an ops console, and a headless M4 needs remote visibility. Make ops state
visible: halt status, guard heartbeat + last verdict, scheduler markers (what ran / what's due),
recon status, healthcheck liveness, equity/drawdown. **Open decision (E1 vs E2):**
- **E1 — extend the Textual TUI** to read `monitor_status.json(l)` + `halt.json` + markers. No new
  dependency; but requires an SSH session to view (no browser).
- **E2 — minimal read-only HTTP status page** (stdlib `http.server`, **no new dependency**) bound to
  LAN/localhost, serving the same JSON + a plain HTML view. Browser-viewable on a headless box;
  read-only (no controls) so it can't trade. *Recommended* for a headless M4. A human-gated "resume"
  button is explicitly **out of scope** here (resume stays CLI-only, human-only — deferred to E3 in
  the older roadmap).

### F. Orphaned intraday-live agent  *(build-here — DECISION below)*
`deploy/launchd/com.quant.intraday-live.plist` isn't wired into install/uninstall/newsyslog and its
`flatten` is a stub. It is not part of the live roster. **Open decision:**
- **F1 (recommended):** **do not install it in Phase 2** — move/mark it clearly experimental and
  defer adoption to Phase 3 (when the intraday showcase is actually wired, per the roadmap). Keeps
  Phase 2 focused on the real live path; avoids a KeepAlive daemon whose flatten is a stub.
- **F2:** adopt it now (wire into install.sh + newsyslog + selfcheck). Only if intraday is meant to
  run live on the M4 imminently — which contradicts the roadmap's Phase 3 placement.

### G. Host-verify + reboot-survival scripting  *(build-here script; prove-on-M4)*
- `deploy/verify_host.sh`: asserts the hardcoded assumptions actually hold on the box (user,
  repo path, `uv` at `/opt/homebrew/bin/uv`, pmset settings, FileVault state) — closes the
  "no install-time verification" gap the audit flagged.
- Script the FileVault auto-unlock-on-planned-reboot flow (`fdesetup authrestart`) into a documented
  helper so a *planned* reboot self-recovers; an *unexpected* power loss still pages the operator
  (accepted, per the locked 2026-06-02 decision — paper box).

### H. 48h soak harness + M4 relay runbook  *(build-here harness; prove-on-M4)*
- `deploy/soak.sh` (or `quant ops soak`): samples at an interval over N hours — agent up/restart
  counts, heartbeat continuity (last healthcheck ping age), log growth, RSS/memory, halt state — and
  emits a `docs/soak/<date>.md` report. Build + unit-test the sampler here; run for real on the M4.
- **Relay runbook** `deploy/PHASE2-VERIFY.md`: the exact M4-side commands for the operator to (1)
  `git pull` + `selfcheck`, (2) **prove crash-restart** (`kill -9` the engine PID, confirm launchd
  respawns within ThrottleInterval), (3) **prove reboot survival** (reboot, confirm all agents
  RunAtLoad + a tick fires), (4) `alert-test` each channel + simulate a MISSED_CRITICAL (pages,
  does NOT trade), (5) start the 48h soak and collect the report. Each step has an explicit
  pass/fail and the evidence to paste back.

### I. alpaca-paper MCP creds  *(operator/relay)*
The `alpaca-paper` MCP returns 401 (its own creds invalid; D4-9). Trading is unaffected (prod uses
the SDK `AlpacaClient`), but independent account audits can't run. Operator refreshes the MCP creds;
documented in the relay runbook so future audits (and a re-run of D4 Part B) work.

---

## Ordering rationale

C (timeout) and B (alerting) are the highest-value *correctness* fixes — they make failures safe and
visible, so do them before relying on a soak. A (selfcheck) + D (logging) make the soak *measurable*.
E (dashboard) + F/G/H are then build-and-document. The M4 proofs (crash, reboot, alert-fire, 48h
soak) run last via relay once the tooling exists. I is operator config, do anytime.

## Open decisions for the checkpoint

1. **Dashboard (E):** minimal read-only HTTP page (E2, recommended, no new dep, browser-viewable) vs.
   extend the TUI (E1, SSH-only)?
2. **Intraday-live plist (F):** defer/mark-experimental (F1, recommended) vs. adopt now (F2)?
3. **Scope size:** all of A–I in one Phase 2, or split into 2a (correctness: B/C/D/A) and 2b
   (observability + M4 proofs: E/G/H/I)? A–I is a lot; 2a first gives the biggest safety win fast.

## Non-goals (YAGNI)

- WARN→BLOCK enforcement flips, TWAP/VWAP execution, resume-button automation — those stay later
  human-gated work (older roadmap Phase 4 / E3).
- Real-money trading, new brokers — deferred per the master roadmap.
- Rewriting the dispatcher/guard internals Phase 1 verified correct — only the named gaps are touched.

## Definition of done

- `quant ops selfcheck` + `alert-test` exist, pass here, and are in the runbook.
- Job timeout enforced (test proves a hung job is killed and re-tried, lock released).
- Structured ANSI-free file logs + all agent logs rotated; guard-status history written.
- Ops dashboard shows halt/heartbeat/recon/scheduler/healthcheck (per the E decision).
- `deploy/verify_host.sh`, `deploy/soak.sh`, `deploy/PHASE2-VERIFY.md` shipped.
- **On the M4 (relay), with pasted evidence:** crash-restart proven, reboot survival proven, every
  alert channel fired, MISSED_CRITICAL pages-without-trading, and a committed 48h soak report with
  zero unplanned restarts.
- Checkpoint: review → this drives the Phase 3 spec.
</content>
