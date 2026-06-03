# M4 Deployment (E1) — operator runbook + GH→M4 cutover

Goal: make the **M4 Mac mini the sole always-on executor** of the live Alpaca
paper system, and **retire the GitHub Actions cron**. Design spec:
`docs/superpowers/specs/2026-06-02-m4-deployment-e1-design.md`.

## ⚠️ Current reality (read first)
As of the 2026-06-03 cutover, the live system is **NOT** running on the M4 yet —
it runs on **GitHub Actions cloud cron on the `main` branch** (`daily-rebalance.yml`
still has `cron: "55 19 * * 1-5"`; GitHub schedules from `main` regardless of this
branch). This branch (`feat/m4-deploy-e1`) removes those schedules, but it is **not
merged to `main`**, so the cron is still live. The M4 has never been provisioned.

## 🔒 THE ONE-EXECUTOR INVARIANT (the prime safety rule)
**Never let the GitHub cron and the M4 both place LIVE orders.** Doing so
double-trades the account. Therefore:
- The M4 runs **dry-run** during shakedown (GH cron keeps trading — no conflict,
  because dry-run places no orders).
- You flip the M4 to **live** ONLY in the same sitting that you **disable the GH
  cron** (Step 5). One executor at all times.
- Rollback at any point = re-enable the GH cron + uninstall the M4 agents (Step 7).

---

## Step 0 — Bootstrap the repo on the M4 (development clone)
```bash
git clone https://github.com/ajaiupadhyaya/quant-trading.git ~/Documents/quant-trading
cd ~/Documents/quant-trading
git checkout feat/m4-deploy-e1
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

## Step 2 — Guard daemon (monitoring; safe — never trades)
Bring it up dry-run first, confirm it cannot halt the book, then go live:
1. Manually: `uv run quant guard run --once --dry-run` → inspect output.
2. Edit `deploy/com.ajaiupadhyaya.quant-guard.plist` to add `--dry-run` to the args,
   `./deploy/install.sh`, watch `~/Library/Logs/quant-deploy/guard.*.log` for a tick.
3. Remove `--dry-run`, re-run `./deploy/install.sh`. The guard can now HALT (but
   never resumes — resume is always manual).

## Step 3 — M4 shakedown in DRY-RUN (while the GH cron still trades)
This is safe to run for several trading days alongside the GH cron — dry-run
places no orders.
1. Edit `quant/deploy/jobs.toml`: set the `daily-rebalance` job's command to
   `["rebalance", "--dry-run"]` (it is currently `["rebalance"]` = live).
2. `./deploy/install.sh` to (re)load the tick + guard launch agents.
3. Sanity: `uv run quant ops tick` once by hand; confirm a marker appears under
   `data/ops/scheduler/`.
4. Over a few sessions confirm: premarket-health / daily-rebalance(dry) /
   reconciliation markers each weekday; catch-up after a forced sleep; tick +
   guard healthchecks stay green on phone; a deliberately-induced failure pages
   Pushover.

## Step 4 — Cutover gates (ALL must pass before going live)
- [ ] Power-cycle survival: `sudo fdesetup authrestart`; after boot both agents
      auto-reload (`launchctl print …/quant-tick` and `…/quant-guard`).
- [ ] Every alert channel tested: healthchecks "send test ping" → phone; a manual
      `uv run quant governance halt` → emergency Pushover received & **acked**;
      then `uv run quant governance resume --reason "test"`.
- [ ] Simulated MISSED_CRITICAL (tick after the close window) pages and does NOT trade.
- [ ] Dry-run rebalance preview matches the GH artifacts for the same days.

## Step 5 — GO LIVE (the cutover — do all of this in one sitting)
**a. Disable the GitHub cron FIRST** (instant, reversible; the decisive safety step):
```bash
for wf in daily-rebalance premarket-health posttrade-reconciliation \
          nightly-backtest weekly-grid-search weekly-validation-governance; do
  gh workflow disable "$wf"
done
gh workflow list            # confirm all show "disabled"
```
(Or GitHub web UI → Actions → each workflow → ••• → Disable workflow.)
Confirm no scheduled run is mid-flight: `gh run list --limit 5`.

**b. Flip the M4 to live:** revert Step 3 — set `daily-rebalance` back to
`["rebalance"]` (no `--dry-run`) in `quant/deploy/jobs.toml`; commit it.

**c. Reload the agents:** `./deploy/install.sh`. The next `daily-rebalance` tick
(15:55 ET) now places real orders — and the M4 is the **sole** executor.

**d. Verify the first live M4 rebalance:** a fresh marker under
`data/ops/scheduler/`, a new Alpaca order with a deterministic client_order_id
`defensive-etf-allocation-YYYYMMDD-<symbol>`, and a green tick heartbeat.

## Step 6 — (optional, later) merge to `main` for hygiene
Once the M4 is proven, merge `feat/m4-deploy-e1` → `main` so `main`'s workflow
files also have the schedules removed (belt-and-suspenders; the `gh workflow
disable` in Step 5 is what actually stops them today).

## Step 7 — ROLLBACK (if the M4 misbehaves)
```bash
./deploy/uninstall.sh                                  # stop the M4 agents
for wf in daily-rebalance premarket-health posttrade-reconciliation \
          nightly-backtest weekly-grid-search weekly-validation-governance; do
  gh workflow enable "$wf"                             # restore the cloud executor
done
```
You are back to the working GitHub-cron system within minutes.

## Operations
- Status: `launchctl print gui/$(id -u)/com.ajaiupadhyaya.quant-tick`
- Logs:   `~/Library/Logs/quant-deploy/{tick,guard}.{stdout,stderr}.log`
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
