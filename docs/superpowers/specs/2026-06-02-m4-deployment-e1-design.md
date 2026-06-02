# M4 Always-On Deployment — Roadmap + Sub-Project E1 Design

**Date:** 2026-06-02
**Status:** Design (awaiting user review → implementation plan)
**Topic:** Turn the M4 Mac mini into the sole always-on host for the quant-trading
system, replacing GitHub Actions cron with a local, DST-correct, catch-up-aware
tick scheduler, plus host hardening and off-box alerting.

This document specifies **sub-project E1** (the foundation) in full, and sketches
the broader four-layer roadmap (E1 → E2 → E3 + the separate intraday track) only
to the depth needed to keep E1's boundaries clean. E2 and E3 get their own
spec → plan → build cycles later.

---

## 1. Context & goals

Today the daily book trades the Alpaca **paper** account via six GitHub Actions
cron workflows (`daily-rebalance`, `premarket-health`,
`posttrade-reconciliation`, `nightly-backtest`, `weekly-grid-search`,
`weekly-validation-governance`), with `ci` and `smoke-test` as event-triggered
CI. Governance is fail-closed; only `defensive-etf-allocation` is live; an
always-on monitoring daemon (`quant guard run`) exists but is not currently
supervised on any always-on host.

The user wants the M4 Mac mini (Apple Silicon, macOS 26.x, SSH-reachable) to
become an **always-running, autonomous, intelligent quant trading analyst** — a
home server that hosts the whole system 24/7. This is decomposed into four
layers; **E1 is the foundation and must land first.**

**E1 goal:** the M4 runs the existing daily system reliably and unattended (to
the extent the chosen security posture allows), with the trade-critical path at
least as safe as the system's existing fail-closed / evidence-gated / PIT
discipline, and with off-box alerting that pages a human when anything is wrong.

### Non-goals for E1
- The LLM analyst (E2), the dashboard (E3), and the intraday engine (C/D) are
  **out of scope** here. E1 only must leave clean seams for them.
- No new trading logic, no strategy changes, no governance-gate changes.

---

## 2. Locked decisions (this session)

| Decision | Choice | Consequence baked into the design |
|---|---|---|
| Vision | Full autonomous brain, decomposed E1→E2→E3 + intraday | E1 is foundation; strict ordering |
| Daily-system host | **Fully retire GH cron**; M4 is sole executor | GH workflows kept as **manual `workflow_dispatch` fallback only**; daily-reboot + power-cycle survival test are **hard cutover gates** |
| Disk encryption | **FileVault ON (Path B)** | No unattended recovery from *unexpected* reboot; planned reboots use `fdesetup authrestart`; unexpected reboot pages the operator for manual unlock |
| Alert provider | **Pushover** (primary) | Emergency priority (retry/expire/receipt) for halts; healthchecks.io as the off-box dead-man's-switch |
| Orchestration | **Approach C** — launchd-supervised tick dispatcher | Pure scheduler core + thin impure shell; manifest is the source of truth |

---

## 3. Roadmap (decomposition)

- **E1 — Bulletproof always-on host (this spec).** Host hardening + local tick
  scheduler (replaces GH cron) + supervised `quant guard run` + dual
  heartbeat/halt alerting + GH-cron migration.
- **E2 — Intelligent analyst (Claude API).** A new `quant/analyst/` package +
  `quant analyst digest` job (a **non-trading**, read-only manifest slot, runs
  post-reconciliation): daily market+portfolio digest, anomaly explanation,
  trade-rationale narration over the day's netted orders, research-idea
  generation. Output is a committed `docs/analyst/` artifact, surfaced by E3 and
  a non-emergency phone push. **Never** submits orders or resumes a halt.
- **E3 — LAN dashboard + alerts.** A small read-only web app (third LaunchAgent)
  rendering heartbeat/halt state, the per-day run-ledger, equity/positions,
  recon reports, and the E2 digest; one **human-gated** "resume" action that
  shells `quant governance resume --reason …` behind explicit confirmation.
- **Intraday track (C → D).** Separate from the daily book, not yet built. D's
  live async engine eventually runs as its **own** KeepAlive LaunchAgent on the
  same E1-hardened host (a continuous loop, **not** a manifest tick-job — like
  `quant guard run`). Reuses E1 host hardening + alerting; orthogonal to the
  scheduler.

**Dependencies:** E1 first. E2 needs E1's artifact cadence + alert client + a
free manifest slot. E3 needs E1 (host, LAN, artifacts, alert-ack) and reads E2's
digest if present (degrade gracefully if absent). The intraday track can proceed
in parallel **after** E1 host hardening exists. Cross-cutting invariant inherited
by all layers: **fail-closed, evidence-gated, PIT, single shared `data_dir`** so
the guard daemon's `halt.json` is authoritative for every trading code path;
**nothing auto-resumes a halt.**

---

## 4. E1 architecture

**Principle: a pure decision core with an injected clock, behind a thin impure
shell.** Everything that decides *what to run and when* is side-effect-free and
unit-tested with an injected time/calendar/markers; the only place with real
I/O (clock, filesystem beyond markers, subprocess, network) is the dispatcher.
This is what makes idempotency, catch-up, and DST behavior provable in tests.

### Components

| Unit | Purpose | Key interface | Depends on | Files |
|---|---|---|---|---|
| `calendar_clock` (pure) | UTC→ET via `zoneinfo America/New_York` (kills GH fixed-UTC DST drift); trading-day / early-close classification by **reusing** `quant/util/trading_calendar.py` | `to_et(now_utc)`, `is_trading_day(d)` (re-exports `trading_calendar.is_trading_day`), **`session_close_et(d)` — NEW, derived from `trading_calendar.is_early_close`** (16:00, or 13:00 on early-close), `within_window(et_dt, target, tol)` — all take explicit args, no `now()` inside | `trading_calendar.py`, stdlib | `quant/deploy/calendar_clock.py`, `tests/deploy/test_calendar_clock.py` |
| `manifest` + `jobs.toml` | Version-controlled source of truth replacing the 6 crons: per-job ET trigger, day rule, catch-up policy, max-lateness horizon, `max_runtime` (stale-lock timeout), command chain, commit paths, `timing_critical` flag | `load_manifest(path) → Manifest` (validates uniqueness, times, policies, commands-are-lists); `Job` dataclass | `calendar_clock`, `tomllib` | `quant/deploy/manifest.py`, `quant/deploy/jobs.toml`, `tests/deploy/test_manifest.py` |
| `scheduler` (pure) | The heart of approach C: given (now_et, manifest, markers) → which jobs to dispatch and as what kind | `due_jobs(now_et, manifest, markers) → list[Dispatch]` where `Dispatch.kind ∈ {FRESH, CATCH_UP, MISSED_CRITICAL, MISSED}` | `calendar_clock`, `manifest` | `quant/deploy/scheduler.py`, `tests/deploy/test_scheduler.py` |
| `markers` | Once-per-**session** idempotency markers + per-day run-ledger; atomic `tmp+os.replace`; written only after success (except the trade pre-submit marker, §6) | `marker_path`, `read_markers(data_dir) → {job: session_date}`, `write_marker(...)` | stdlib | `quant/deploy/markers.py`, writes `data/ops/scheduler/<job>.<session-date>.json`, `tests/deploy/test_markers.py` |
| `dispatcher` (impure shell) | The 60s tick entrypoint: read real ET clock → load manifest+markers → `due_jobs` → shell each job via `uv run quant …`; single-flight lock; markers; heartbeat/halt pings | CLI `quant ops tick` (one tick, exit) and `quant ops run-job <name> [--catch-up] [--force]`; `main(now=None)` seam for tests | all above + `alerts` | `quant/deploy/dispatcher.py`, new `quant ops` group in `quant/cli.py`, `tests/deploy/test_dispatcher.py` |
| `alerts` | Reusable: liveness pings (healthchecks) + direct emergency push (Pushover/ntfy) | `AlertClient(cfg).ping_success/ping_start/ping_fail/send_emergency`; HTTP transport injectable | `Settings` | `quant/deploy/alerts.py`, extend `quant/util/config.py`, `tests/deploy/test_alerts.py` |
| launchd + installer | OS integration: two LaunchAgents + power/log config | plists, idempotent `install.sh`/`uninstall.sh`, `pmset.sh`, `newsyslog` conf | — | `deploy/` (plists, scripts, `newsyslog/quant-deploy.conf`, `README.md`) |

---

## 5. Scheduler model (detailed)

### 5.1 Tick cadence
`com.ajaiupadhyaya.quant-tick.plist`: `StartInterval=60`, `RunAtLoad`,
`ProcessType=Standard`. **`StartInterval` (run-and-exit), not `KeepAlive`** — the
10s respawn-throttle gotcha applies only to KeepAlive forever-loops; a periodic
run-and-exit tick is exactly what `StartInterval` is for and exits well under a
tick. `ProgramArguments` uses absolute `/opt/homebrew/bin/uv`; `WorkingDirectory`
= repo (so `.env` + relative `data_dir` resolve); `EnvironmentVariables` sets a
`PATH` including Homebrew (launchd's default PATH excludes it).

### 5.2 ET + calendar
Each tick: `now_et = to_et(datetime.now(UTC))` (UTC has no DST → conversion is
always unambiguous). Day predicates per job:
- **`WEEKDAYS_TRADING`** — Mon–Fri **and** `is_trading_day` (holidays skipped,
  unlike the GH crons which fired on holidays and relied on downstream guards).
- **`SPECIFIC_WEEKDAY(Sat)`** — the two weekly jobs.
- **`TRADING_DAY_EVENING`** — attributed to a **completed** session (see §5.5).

### 5.3 Idempotency markers (session-scoped)
`data/ops/scheduler/<job>.<session-date>.json`, atomic write, recording
`{job, session_date, fired_at_utc, kind, exit_code, duration_s}`. Markers are
keyed by the job's **session-date**, not merely "today," so a missed prior
session is representable (fixing the single-marker-per-calendar-date flaw the
review caught). **Markers are git-ignored** (host-local run state); job *output*
artifacts continue to be committed as today. `read_markers` returns the latest
session-date per job.

### 5.4 Due / catch-up / missed rules (pure)
For each job on a matching day, with that day's session date `D`. Two job
families resolve on **disjoint** ladders so a job never falls into two classes:

**Catch-up-safe jobs** (`catch_up=SAME_DAY`) ride FRESH → CATCH_UP → MISSED:
- **FRESH** — `now_et` within `[trigger_et(D), trigger_et(D) + FRESH_TOL]` and no
  marker for `D`. `FRESH_TOL = 3 minutes` (the general window width).
- **CATCH_UP** — `now_et` past the FRESH window but still within the job's
  **bounded catch-up horizon** `[trigger_et(D), max_lateness(D)]`, and no marker
  for `D`. Covers a tick missed because the box was asleep/rebooting.
- **MISSED** — the catch-up horizon for `D` has fully elapsed with no marker:
  classified MISSED (not silently dropped, not fired) and **alerts**.

**Timing-critical job** (daily-rebalance, `catch_up=NONE`) **skips the
CATCH_UP/MISSED ladder entirely** and rides only FRESH → MISSED_CRITICAL:
- **FRESH** — within `[close−5min, close−2min]` (§5.6), no marker for `D`.
- **MISSED_CRITICAL** — at/after the hard cutoff `close−2min` with no marker:
  **not auto-fired**; records a MISSED_CRITICAL marker and fires a direct
  emergency push so a human decides (`quant ops run-job daily-rebalance --force`).

Per-job `max_lateness`: reconciliation → 23:59 ET same day; nightly-backtest →
09:00 ET next morning (same logical session, see §5.5); weekly jobs → Sun 23:59
ET. The "asleep Thu 14:00 → wake Fri 03:00" scenario therefore yields:
Thursday's reconciliation/backtest classified **MISSED + alerted** (their Thursday
horizon elapsed), Friday's jobs do **not** pre-fire at 3am (their FRESH windows
haven't opened), and daily-rebalance Thursday is MISSED_CRITICAL.

### 5.5 `TRADING_DAY_EVENING` semantics
The nightly-backtest is **attributed to the completed session it follows**.
Trigger 22:00 ET; catch-up horizon spans the midnight boundary:
**22:00 ET day D → 09:00 ET day D+1, attributed to session D.** The session-date
attribution rule is explicit by tick time: **`session_date = now_et.date()` if
`now_et` is before midnight (the 22:00–23:59 window), else
`previous_trading_day(now_et.date())`** (the 00:00–09:00 window). A tick at
01:30 ET Saturday with no Friday marker therefore resolves to session-date
Friday and catches up Friday's backtest (or alerts MISSED), never silently
skips it.

### 5.6 Timing-critical trigger derived from `session_close_et`
**The daily-rebalance trigger is NOT hardcoded 15:55.** It is
`session_close_et(D) − 5 min`:
- Regular day → close 16:00 ET → trigger 15:55 ET.
- **Early-close day** (13:00 ET, e.g. day-after-Thanksgiving, Christmas Eve) →
  trigger **12:55 ET**. A tick at 15:55 ET on such a day is MISSED_CRITICAL, not
  a post-close fire.

**Hard wall-clock cutoff:** no order submission after `session_close_et(D) − 1 min`
regardless of tolerance. FRESH window = `[close−5min, close−2min]`; at/after
`close−2min` with no marker → MISSED_CRITICAL. This guarantees a late network
recovery cannot trade post-close (§7).

### 5.7 Shell-out + single-flight
Each `Job` carries `commands` as a list of arg-lists; the dispatcher runs them as
`/opt/homebrew/bin/uv run quant <args>` (cwd=repo). The daily-rebalance chain:
`data refresh` (**non-fatal**) → `data quality` → `risk pretrade` →
**`doctor` (BLOCKING — stop chain on non-zero)** → `rebalance`. (Adding `doctor`
as a hard preflight gate for the trade is a deliberate tightening vs the GH
daily-rebalance, which ran `doctor` only in premarket.) Reconciliation shells
`uv run python scripts/reconcile_live.py`. Committing jobs reproduce the GH
push-race **after** their commands (see §9, §11).

**Single-flight:** non-blocking `flock` (`LOCK_NB`). The **daily-rebalance has
its own lock**, separate from the batch-job lock, so a long-running weekly
grid-search can never silently block the trade window. The lock file records
PID+timestamp; a holder whose PID is dead, or whose age exceeds that job's
**`max_runtime`** (a per-job field in `jobs.toml`, e.g. grid-search 8h,
rebalance 10min), is broken with an alert. If the rebalance window is reached
while its (own) lock is held, an alert fires (the rebalance is being blocked) and
the situation is classified rather than silently skipped.

---

## 6. Trade-submission idempotency (the critical correctness fix)

The review found that `make_client_order_id` appends `…uuid4().hex[:8]`
(`quant/execution/orders.py:35`) which gives
the broker **zero idempotency**, so a crash after Alpaca accepts orders but before
the success marker is written would let the next 60s tick **re-submit** within the
still-open window. Fixes (all in scope for E1):

1. **Deterministic `client_order_id`** per `(strategy, symbol, session-date)`,
   so Alpaca rejects a duplicate same-day order outright.
2. **Pre-submit marker**: `quant rebalance` writes its own once-per-session
   "submitted" marker the **instant** orders are dispatched — **before** any
   git/commit step. A post-submit crash therefore leaves a marker that blocks
   re-fire. The git-push/commit steps **must not** gate this marker; a push
   failure alerts but never causes a re-trade.
3. **Reconcile-then-refuse**: on entry, `rebalance` queries Alpaca for today's
   orders by date and refuses to submit if same-day orders already exist
   (enforced in code, not prose).
4. The `--force` manual path (documented recovery for MISSED_CRITICAL) still runs
   the pre-submit broker check, so a forced run colliding with a scheduled tick
   cannot double-trade.

---

## 7. Network-down policy

- `quant rebalance` treats network-unreachable as a **hard non-fire**
  (fail-closed, no orders); the close-window hard cutoff (§5.6) means a late
  network recovery past `close−2min` is MISSED_CRITICAL, never a post-close trade.
- `data refresh` is non-fatal, but a **persistent** refresh failure must not let
  the trade proceed on stale bars silently: `doctor` (now a blocking preflight)
  must fail on bar staleness, and the chain stops.
- The alert client distinguishes "network unreachable" from other failures. On
  `send_emergency` failure it **persists the pending emergency to disk and
  retries every subsequent tick** until delivery is confirmed — so a halt that
  trips during an outage is delivered the moment connectivity returns.

---

## 8. Halt / status durability (fail-closed)

The review found `halt.py` uses `path.write_text` (non-atomic) and `load_halt`
does not fail-closed on corruption, while every trading path calls `load_halt`
(`rebalance.py:182`). Fixes:

- `set_halt`/`clear_halt`/`write_status` become **atomic** (`tmp + os.replace`),
  matching marker discipline.
- `load_halt` **fails closed**: a malformed/unreadable `halt.json` is treated as
  `active=True` (assume halted), never raising-through or defaulting to
  not-halted.
- The dispatcher's per-tick halt read tolerates a transient mid-write parse error
  (retry within the tick) without alarming or crash-looping.
- Test: corrupt `halt.json` → rebalance refuses to trade **and** the dispatcher
  does not crash-loop.

---

## 9. Alerting (Pushover + healthchecks, two independent paths)

**Path 1 — liveness / dead-man's-switch (off-box, passive).** healthchecks.io
free tier (2 of 20 checks used). The **cloud** runs the watcher so a dead M4
still alerts.
- **Tick heartbeat** (Period 1 min, Grace 3 min): the dispatcher pings on a clean
  tick. **A tick where the timing-critical/trade job failed does NOT ping
  success** — so a failed rebalance opens the heartbeat gap and the cloud check
  alarms. (No dedicated cloud *cron* check for the rebalance: a literal cron
  string like `55 15 * * 1-5` cannot track holidays or the early-close 12:55
  trigger, so it would false-alarm. The rebalance is covered instead by two
  alive-box mechanisms — the suppressed success heartbeat above, and the direct
  MISSED_CRITICAL emergency push in §5.4/Path 2 — plus this tick heartbeat for
  the box-down case.)
- **Guard liveness**: the **guard daemon pings its OWN check directly from inside
  `run_loop`**, decoupled from the dispatcher — so dispatcher-death and
  guard-death are independently distinguishable. (The dispatcher also watches
  `monitor_status.json` freshness as a secondary signal.)
- Cold-start: a missing `monitor_status.json` on a box up < guard-interval is
  benign (no alarm); missing/stale after that → `/fail`.
- A wake-up catch-up storm is **coalesced into a single summary ping** (respect
  the 5/min limit).
- healthchecks forwards a miss to the phone via its **native Pushover
  integration** (no webhook glue).

**Path 2 — halt / critical (on-box, active, immediate).** The instant a fresh
halt is detected (`halt.active` flips true, or `monitor_status.halt_triggered_this_tick`),
the dispatcher curls Pushover **directly** at `priority=2 retry=60 expire=3600`
(required for prio 2; returns an ack receipt) — bypassing healthchecks' poll
cadence. The same direct push fires on MISSED_CRITICAL. **Resume is always human.**
A guard KeepAlive restart must **not** re-emit an emergency for an
already-known/acked halt (dedup on halt identity, not on the transient
`halt_triggered_this_tick` flag).

**Secret hygiene:** `HEALTHCHECKS_*_URL` and `NTFY_TOPIC_URL` are **bearer-secrets**
(possession = ability to spoof liveness / publish). Never log the full URL (log
the check **name** only); pass URLs to `curl` via env/stdin, never as logged
argv; `quant doctor` must redact all Settings secret fields. A test asserts no
Settings secret value appears in captured dispatch stdout/stderr.

---

## 10. Host hardening (FileVault Path B)

### 10.1 Power (`pmset`, persistent, survives reboot)
```
sudo pmset -a sleep 0 disablesleep 1 displaysleep 0 disksleep 0 \
  autorestart 1 womp 1 powernap 0 standby 0 tcpkeepalive 1
```
`disablesleep 1` hard-blocks idle sleep; `autorestart 1` reboots on AC return;
`womp 1` wake-on-LAN. `caffeinate` is **not** the primary mechanism (per-process,
dies on disconnect, no reboot survival) — used only ad-hoc during bring-up.
Verify with `pmset -g` and `pmset -g assertions`.

### 10.2 FileVault Path B operational model
FileVault **ON**. Consequences and handling:
- **Planned reboots** (macOS updates; opt-in daily safety reboot) use
  `sudo fdesetup authrestart` (one unlock-free boot) so the LaunchAgents come
  back. Schedule updates **manually** in a market-closed window and verify both
  agents + `.env`/keychain reachable afterward.
- **Unexpected power loss** → box halts at the unlock screen; the off-box
  dead-man's-switch fires (the locked box can't ping) → operator is paged →
  manual SSH/physical unlock. **This is the documented, accepted recovery path**
  (see §12). Treat any FileVault state change as a paged event.
- **Daily 04:00 safety reboot** (`pmset repeat restart` via `authrestart`) is
  **OPT-IN and OFF by default** until power-cycle recovery is proven N times —
  because each reboot re-hits the FileVault/auto-login path.
- A **post-boot self-check** LaunchAgent (or first-tick-after-boot) verifies both
  agents are loaded (`launchctl print`) and `.env` is reachable, then fires a
  **positive "box recovered, agents up"** push after every reboot. Absence of
  that push after a known reboot is itself an alarm.

### 10.3 launchd
- `com.ajaiupadhyaya.quant-tick.plist` — `StartInterval=60` (§5.1).
- `com.ajaiupadhyaya.quant-guard.plist` — the always-on `uv run quant guard run`
  (no `--once`/`--max-ticks` → forever loop), `KeepAlive=true`,
  `ProcessType=Standard` (no throttling of the safety loop), `ThrottleInterval=30`
  (back off a crash-loop), `ExitTimeOut=30`. Exactly **one** instance against the
  shared `data_dir`. Bring-up runs it `--dry-run` (cannot halt live) until
  verified, then drop `--dry-run`.
- Agents run in `gui/$UID` (keychain reachable in the GUI session; do **not**
  spawn from raw SSH where the keychain may be locked). Verify:
  `launchctl print gui/$(id -u)/<label>`. Install is idempotent
  (bootout-then-bootstrap). `sudo systemsetup -setremotelogin on` for SSH.

### 10.4 Secrets & logs
- Keep `.env` at the repo root (loaded by `pydantic-settings`, already wired),
  `chmod 600`. **No secrets in plist `EnvironmentVariables`** (plists are
  world-readable). Extend `.env`/`.env.example`/`Settings` with
  `HEALTHCHECKS_TICK_URL`, `HEALTHCHECKS_GUARD_URL`,
  `PUSHOVER_APP_TOKEN`, `PUSHOVER_USER_KEY` (Optional fields so CI's dummy env
  still instantiates). Ensure `.env` is not stageable by any committing job.
- **Logs** → `~/Library/Logs/quant-deploy/` (`mkdir -p` before first load).
  Rotate via built-in **newsyslog** (`/etc/newsyslog.d/quant-deploy.conf`),
  always specifying `ajaiupadhyaya:staff` owner (else newsyslog recreates them
  `root:root` and the user-context agents lose write). Mode **600** (logs contain
  financial state, e.g. equity per guard tick). The guard agent holds its log
  handle open → use an in-app `RotatingFileHandler` for guard **or** a
  newsyslog-only policy, never both. `chmod 700` the log dir. Validate:
  `sudo newsyslog -nv`.

### 10.5 Clock & disk guards (new failure modes the review surfaced)
- **NTP**: assert `timed`/network time sync is enabled and healthy; a sanity
  check rejects an implausible `now_utc` (clock-jump guard) before the scheduler
  trusts it.
- **Disk-space guardrail**: before marker/status/halt writes and before
  committing jobs, check free space against a floor; below it, **alert** and
  refuse to start new heavy jobs (a full disk makes `os.replace` fail → jobs
  would re-fire forever and halts couldn't persist). `data/backtests/` and
  `data/snapshots/` growth is bounded by a retention policy in the spec's
  follow-ups.

---

## 11. GitHub Actions → M4 migration

1. **Keep event CI on GH.** `ci.yml`, `smoke-test.yml` (push/PR, not host-bound)
   stay active and unchanged. The M4 does not reproduce them.
2. **Disable the 6 cron schedules without deleting files.** Comment out the
   `schedule:` block, **keep `workflow_dispatch`** (manual fallback executor),
   add a header note: *"SCHEDULE RETIRED 2026-06 — executor migrated to the M4
   tick scheduler; source of truth = `quant/deploy/jobs.toml`. Kept for manual
   `workflow_dispatch` fallback + history."* (Config-only; archive-don't-delete.)
3. **Map each cron 1:1 into `jobs.toml`** (ET, DST-correct, holiday-skipping):

   | Job | Trigger (ET) | Day rule | Catch-up | Commits |
   |---|---|---|---|---|
   | premarket-health | 09:00 | WEEKDAYS_TRADING | SAME_DAY | none (read-only) |
   | **daily-rebalance** | **close−5min** (15:55 / 12:55 early-close) | WEEKDAYS_TRADING | **NONE / timing_critical** | data/live, data/raw, data/ops/health, data/risk |
   | posttrade-reconciliation | 17:30 | WEEKDAYS_TRADING | SAME_DAY | docs/live-recon |
   | nightly-backtest | 22:00 | TRADING_DAY_EVENING | SAME_DAY (horizon → 09:00 D+1) | data/backtests |
   | weekly-grid-search | Sat 02:00 | SPECIFIC_WEEKDAY(Sat) | SAME_DAY | data/backtests |
   | weekly-validation-governance | Sat 04:00 | SPECIFIC_WEEKDAY(Sat) | SAME_DAY | data/backtests, data/governance |

4. **Honest timing change (NOT a faithful 1:1).** The GH crons were fixed-UTC, so
   in **EST winter** they ran an hour earlier in ET. The manifest pins ET, which
   **intentionally moves winter execution +1h**: daily-rebalance 14:55→15:55 ET
   (tighter to the close — hence the hard cutoff + close-derived trigger),
   reconciliation 16:30→17:30 ET. The migration regression test asserts the
   **intended differences** (and the early-close special-case), not equivalence.
5. **State durability.** The M4 owns local `main`; committing jobs do
   `git add <commit_paths>` → skip if clean → commit → 3× `git pull --rebase
   --autostash origin main && git push` (backup to origin). Each committing job
   **asserts it is on the `main` branch with no rebase/merge in progress** before
   committing, and runs `git rebase --abort` on failure so a poisoned working
   tree never leaks into the next job. (The session started on a detached `HEAD` —
   the dispatcher must guarantee a named branch.)
6. **Cutover & rollback.** Run the M4 scheduler in parallel for several days with
   `quant rebalance --dry-run` + guard `--dry-run`; confirm
   markers/catch-up/heartbeat/alerts and match outputs against GH artifacts; then
   disable GH schedules and flip live. Rollback = re-enable a GH schedule.

---

## 12. Availability posture & accepted risks

Going **M4-only + FileVault Path B** is a deliberate trade of cloud HA for
self-hosting + at-rest security. Explicitly accepted:

- The single order-submitting job (daily-rebalance) has **no automatic off-box
  fallback**. An M4 outage during the ~3-min close window = a missed rebalance
  handled manually. Given the only live strategy is a daily defensive-ETF
  rebalance, a missed day is low-harm.
- An **unexpected** reboot leaves the box FileVault-locked until manual unlock;
  the dead-man's-switch pages the operator (locked box can't ping). Documented
  recovery: SSH/physical unlock, then `quant ops run-job daily-rebalance --force`
  if still pre-close.
- **healthchecks.io is a third-party SPOF** for liveness; a scheduled weekly
  **synthetic test push that the operator must ack** detects silent push death
  (the #1 real-world alerting failure) and the watcher being down.
- The guard daemon **fails safe-to-OK on missing data** (an Alpaca outage means
  guardrails can't evaluate → no halt). Flagged; not changed in E1.

**Hard cutover gates (blocking):** power-cycle survival proven (planned reboot
via `authrestart` brings both agents back); every alert channel tested with a
real "send test notification"; the parallel dry-run period matches GH outputs.

---

## 13. Testing strategy

**Core principle:** scheduler decision logic is **pure** and unit-tested with
injected clock+calendar+markers — no `datetime.now()`, filesystem (beyond
injected marker paths), subprocess, or network in
`calendar_clock`/`manifest`/`scheduler`/`markers`. New tests under `tests/deploy/`
run in the existing `uv run pytest` and the still-active GH `ci.yml` (no
network/alpaca/slow markers on the pure tests).

1. **Idempotency** — exactly one FRESH fire per session-date; a present marker
   suppresses re-fire; ticking every 60s across a simulated trading day yields
   one fire per job; a stubbed failure leaves no marker → next tick re-attempts a
   catch-up-safe job.
2. **Catch-up (session-scoped)** — "asleep Thu 14:00 → wake Fri 03:00": Thursday
   reconciliation/backtest are **MISSED + alerted** (not fired as Friday jobs);
   Friday's jobs do **not** pre-fire at 3am; daily-rebalance Thursday is
   MISSED_CRITICAL (not CATCH_UP), not auto-dispatched, and a direct emergency
   push fires. Box-asleep-through-close → no post-close order chain runs.
3. **DST** — winter (EST, UTC−5) and summer (EDT, UTC−4) instants land each job
   at the same ET wall-clock. **Synthetic test jobs at 01:30 and 02:30 ET**
   exercise the fall-back ambiguous hour (exactly one fire across both 01:30
   passes) and spring-forward gap (02:30 either fires in the compressed window or
   is MISSED, never silently dropped). Documented: no production job may be
   scheduled 01:00–03:00 ET on weekends without this coverage.
4. **Calendar** — holiday (Juneteenth 2026, Good Friday) → no WEEKDAYS_TRADING
   fire; early-close (day-after-Thanksgiving) → rebalance FRESH ≈ 12:55 ET and a
   15:55 tick that day is MISSED_CRITICAL; nightly-backtest fires the evening of a
   completed session, not after a holiday.
5. **Trade idempotency** — deterministic `client_order_id`; pre-submit marker
   blocks re-fire after a simulated post-submit/pre-commit crash;
   reconcile-then-refuse rejects a second same-day submit; `--force` still runs
   the broker check.
6. **Dispatcher (impure, stubbed clock/subprocess/git/HTTP/AlertClient)** —
   correct chains (absolute uv, args, cwd); daily-rebalance stops on non-zero
   `doctor`, `data refresh` non-fatal; push-race retries 3× then alerts;
   per-job non-blocking lock prevents concurrent double-dispatch; a long job
   holding the batch lock across the close window does **not** block the rebalance
   (separate lock) and alerts; fresh halt → direct emergency push; success
   heartbeat suppressed on a failed timing-critical job; **no Settings secret
   value appears in captured stdout/stderr**.
7. **Halt durability** — corrupt `halt.json` → rebalance refuses + dispatcher
   doesn't crash-loop; atomic writes verified.
8. **Alerts** — injected HTTP transport asserts ping URLs + timeout/retry +
   ≤1/min debounce; `send_emergency` builds Pushover `priority=2 retry=60
   expire=3600`; undelivered emergency persisted + retried; tokens never logged.
9. **Manifest fidelity** — `jobs.toml` loads, names unique, commands are lists,
   policies known; the 6 mapped jobs match the retired crons in commands +
   commit_paths and assert the **intended** ET timing differences.

**Live/integration (manual, pre-cutover, not in CI):** bring-up runbook — parallel
dry-run for several days; real test push on every channel; simulated power-cycle
proving recovery (planned-reboot path under Path B); re-test push after any iOS
update.

---

## 14. Resolved open questions

| Question | Resolution |
|---|---|
| FileVault | **Path B (ON)**; planned reboots via `fdesetup authrestart`; unexpected = paged + manual unlock |
| Off-box safety net | **Fully retire GH cron**; GH kept as manual `workflow_dispatch` fallback; power-cycle survival + channel tests are blocking cutover gates |
| Alert provider | **Pushover** (Emergency priority); healthchecks.io off-box watcher; ntfy not used initially |
| Markers in git? | **Git-ignored** (host-local run state); job output artifacts still committed |
| Tick tolerance | General FRESH tol = **3 min**; **daily-rebalance** FRESH = `[close−5min, close−2min]`, hard no-submit after `close−2min` |
| `doctor` preflight | **Yes** — blocking gate on the daily-rebalance chain (stricter than GH) |
| Git push-race | **Kept** (origin backup), plus branch-assertion + `rebase --abort`-on-failure |
| Daily 04:00 reboot | **Opt-in, OFF by default** until power-cycle recovery proven; via `authrestart` |
| nightly-backtest | Attributed to the completed session; catch-up horizon 22:00 ET D → 09:00 ET D+1 |

---

## 15. Out of scope / deferred

- **E2 (analyst), E3 (dashboard), intraday C/D** — own specs.
- Per-symbol borrow rates, multi-host HA, Keychain migration for secrets,
  `data/` retention/pruning policy beyond the disk-floor guard, two-way phone
  control — noted for follow-up, not built in E1.
