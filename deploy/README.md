# M4 Deployment — operator runbook

Goal: keep the **M4 Mac mini as the sole always-on executor** of the Alpaca
paper system. The local launchd agents run the scheduler tick, guard daemon, and
continuous read-only engine. GitHub Actions workflows are retained for manual
fallback/history, but active cron schedules were removed from `main`.

## Current Reality (Read First)

As of 2026-06-05, `main` contains the M4 deployment stack and the active GitHub
cron schedules are removed. The expected production shape is:

- `com.ajaiupadhyaya.quant-tick`: launchd `StartInterval=60`; dispatches due
  jobs from `quant/deploy/jobs.toml`.
- `com.ajaiupadhyaya.quant-guard`: launchd `KeepAlive`; can halt the paper book
  on severe guardrail breaches, never resumes automatically.
- `com.ajaiupadhyaya.quant-engine`: launchd `KeepAlive`; read-only continuous
  market-state engine, never trades or halts.

The current paper-live scheduled job is `daily-rebalance` in
`quant/deploy/jobs.toml`. It places **Alpaca paper** orders at the close window
when the readiness chain passes. Re-add `--dry-run` to that job if you want a
no-orders shakedown.

## 🔒 THE ONE-EXECUTOR INVARIANT (the prime safety rule)
**Never let a GitHub scheduled workflow and the M4 both place orders.** Doing so
double-trades the paper account. `main` currently has no active workflow
`schedule:` triggers, and `tests/deploy/test_migration_fidelity.py` enforces
that. If you ever re-enable cloud schedules, first stop or dry-run the M4
`daily-rebalance` job.

---

## Step 0 — Bootstrap The Repo On The M4
```bash
git clone https://github.com/ajaiupadhyaya/quant-trading.git ~/Documents/quant-trading
cd ~/Documents/quant-trading
cp .env.example .env && chmod 600 .env     # then fill in secrets (below)
uv sync --all-extras
uv run quant doctor                        # expect 7/7 (needs a bar cache; see note)
```
`.env` (chmod 600, never committed) must contain:
- **`ALPACA_API_KEY` / `ALPACA_SECRET_KEY`** — the PAPER keys (required to trade).
- **`FRED_API_KEY`** — macro data.
- **`PUSHOVER_APP_TOKEN` / `PUSHOVER_USER_KEY`** — emergency phone push.
- **`HEALTHCHECKS_TICK_URL` / `HEALTHCHECKS_GUARD_URL`** — off-box dead-man's-switch.
- `SLACK_WEBHOOK_URL`, `ANTHROPIC_API_KEY` — optional (daily digest/brief).

`quant doctor`'s `bar_freshness` check needs a cache: run `uv run quant data refresh
--start 2018-01-01` once first. The plists assume the repo is at
`/Users/ajaiupadhyaya/Documents/quant-trading`, the user is `ajaiupadhyaya`, and
`uv` is at `/opt/homebrew/bin/uv` — **verify these match this M4** (`whoami`,
`which uv`, `pwd`); if not, edit both `deploy/*.plist` (WorkingDirectory, log
paths, uv path) before installing.

## Step 1 — Host hardening (one-time)
1. `./deploy/pmset.sh`              # never sleep, auto-restart on power, WoL, remote login (needs sudo)
2. FileVault ON (System Settings → Privacy & Security). For planned reboots:
   `sudo fdesetup authrestart` (one unlock-free boot).
3. Create 2 healthchecks.io checks (tick: period 1m, grace 3m; guard: period 5m,
   grace 11m); wire each to the Pushover integration; put their ping URLs in `.env`.
4. `mkdir -p ~/Library/Logs/quant-deploy`

## Step 2 — Manual Preflight Before Loading Agents

Run these from the repo root:

```bash
uv run quant doctor
uv run quant data quality --start 2018-01-01 --symbols SPY,TLT,IEF,GLD,DBC,VNQ,EFA,EEM
uv run quant risk pretrade
uv run quant guard run --once --dry-run
uv run quant engine run --once --dry-run
uv run quant ops tick
```

Expected:
- `doctor` is `7/7`.
- data quality passes with zero missing bars.
- pretrade risk passes.
- guard reports account/reconciliation/bar freshness without placing orders.
- engine writes one read-only market-state cycle.
- `ops tick` exits 0; if run during a due window, due jobs should complete.

## Step 3 — Install Or Reload LaunchAgents

```bash
./deploy/install.sh
```

This loads/reloads all three agents: tick, guard, and engine. The guard can HALT
but never resumes. The engine is read-only. The tick job may place paper orders
only when `daily-rebalance` is due and its readiness chain passes.

## Step 4 — Operational Gates
- [ ] Power-cycle survival: `sudo fdesetup authrestart`; after boot both agents
      auto-reload (`launchctl print …/quant-tick`, `…/quant-guard`,
      `…/quant-engine`).
- [ ] Every alert channel tested: healthchecks "send test ping" → phone; a manual
      `uv run quant governance halt` → emergency Pushover received & **acked**;
      then `uv run quant governance resume --reason "test"`.
- [ ] Simulated MISSED_CRITICAL (tick after the close window) pages and does NOT trade.
- [ ] First live M4 paper rebalance verified: a fresh marker under
`data/ops/scheduler/`, a new Alpaca order with a deterministic client_order_id
`defensive-etf-allocation-YYYYMMDD-<symbol>`, and a green tick heartbeat.

## Step 5 — Rollback
```bash
./deploy/uninstall.sh                                  # stop the M4 agents
```

If you intentionally restore GitHub cloud scheduling later, do so only after the
M4 `daily-rebalance` job is stopped or changed to `--dry-run`.

## Operations
- Status:
  - `launchctl print gui/$(id -u)/com.ajaiupadhyaya.quant-tick`
  - `launchctl print gui/$(id -u)/com.ajaiupadhyaya.quant-guard`
  - `launchctl print gui/$(id -u)/com.ajaiupadhyaya.quant-engine`
- Logs:   `~/Library/Logs/quant-deploy/{tick,guard}.{stdout,stderr}.log`
          and `~/Library/Logs/quant-deploy/engine.{stdout,stderr}.log`
- Manual job (recovery for a missed window): `uv run quant ops run-job daily-rebalance` (see `--help`)
- Resume after a halt (human only): `uv run quant governance resume --reason "..."`
- Deploy code changes: `git pull` on the M4 (it does NOT auto-pull); agents run
  from the working tree, so the next tick picks up the change.
- Uninstall: `./deploy/uninstall.sh`

## Accepted risks (M4-only + FileVault Path B)
- No off-box executor fallback while live; an M4 outage in the ~3-min close window
  = a missed rebalance, handled manually (low-harm: a daily defensive-ETF book).
- Unexpected power loss → box at the FileVault unlock screen; the dead-man's-switch
  pages you; recover via SSH/physical unlock.
- healthchecks.io is a 3rd-party SPOF; a weekly synthetic test push you must ack
  detects silent push death.
