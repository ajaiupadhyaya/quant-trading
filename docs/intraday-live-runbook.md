# Intraday-Live Operator Runbook

## What it is

The intraday-live daemon runs the **60-second tick loop** that trades a ring-fenced
ETF sleeve (QQQ / IWM / DIA) on the Alpaca paper account.  It is **additive to and
isolated from** the daily system: the sleeve holds a disjoint set of symbols, keeps
its own ledger, and is subject to independent guardrails.  The daily system (quant
engine) continues to trade its own universe unaffected.

---

## Guardrail profile

| Parameter | Value |
|---|---|
| Sleeve equity cap | `min(10 % of account equity, $10 000)` |
| Per-trade notional | ~$2 000 |
| Max round-trips / day | ~20 |
| Daily-loss auto-flatten + halt | 1.5 % of sleeve equity |
| End-of-day flatten buffer | Flat 15 min before market close |

---

## Install / uninstall

### One-time setup

Copy (or symlink) the plist into the LaunchAgents folder and load it:

```bash
# symlink (preferred — picks up edits automatically)
ln -sf "$(pwd)/deploy/launchd/com.quant.intraday-live.plist" \
    ~/Library/LaunchAgents/com.quant.intraday-live.plist

# load and enable
launchctl load -w ~/Library/LaunchAgents/com.quant.intraday-live.plist
```

### Check status

```bash
launchctl list | grep intraday-live
```

A PID in the first column means the daemon is running.  Exit code `0` in the
second column means the last run exited cleanly.

### Uninstall

```bash
launchctl unload -w ~/Library/LaunchAgents/com.quant.intraday-live.plist
rm ~/Library/LaunchAgents/com.quant.intraday-live.plist
```

---

## Operate

### Status

```bash
uv run quant intraday live status
```

Shows the current halt state and the most-recent journaled tick (day P&L, round-trips,
order count).

### Halt (emergency stop)

```bash
uv run quant intraday live halt --reason "manual halt: investigating position"
```

Sets `data/intraday/live/sleeve_halt.json`.  The running daemon will see the halt flag
on its next tick and stop submitting orders.  launchd keeps the process alive; the
loop enters a no-op spin until resumed or killed.

### Resume

```bash
uv run quant intraday live resume --reason "investigation complete"
```

### Reconcile

```bash
uv run quant intraday live recon
```

Compares the in-process ledger against live broker positions (filtered to the sleeve
universe) and prints any mismatches.

### Dry-run (test without submitting orders)

```bash
uv run quant intraday live run --max-ticks 3 --dry-run
```

Runs exactly 3 ticks, logging what orders would have been submitted without touching
the broker.

---

## Where state lives

| Artifact | Path |
|---|---|
| Tick journal | `data/intraday/live/ticks.parquet` |
| Sleeve halt flag | `data/intraday/live/sleeve_halt.json` |
| launchd stdout | `~/Library/Logs/quant-deploy/intraday-live.stdout.log` |
| launchd stderr | `~/Library/Logs/quant-deploy/intraday-live.stderr.log` |

---

## Crash recovery

On daemon startup the ledger is **rebuilt from broker positions** (filtered to the
sleeve universe: QQQ / IWM / DIA).  Any fills placed before the crash that are
reflected in the broker account are automatically picked up; no manual reconciliation
is required for a clean restart.

---

## EXPLICIT WARNING — symbol isolation

> **NEVER add a daily-system symbol to the sleeve universe.**

The daily system trades: **SPY / TLT / IEF / GLD / DBC / VNQ / EFA / EEM**.

The sleeve trades: **QQQ / IWM / DIA**.

These universes are disjoint by design.  Both systems share a single Alpaca paper
account.  If a symbol appears in both universes the sleeve ledger and the daily
ledger will develop conflicting position views of the same ticker, order netting will
break, and the combined risk will be unmeasured.

---

## Training / acting split

The live daemon **only acts** — it reads the currently promoted strategy parameters
and executes signals.  It **never trains**.

Offline tuning jobs (run separately) retrain the mean-reversion model and promote new
parameters to the config store.  Drift is observed via the tick journal (`ticks.parquet`)
and the `recon` command, not by the live loop itself.
