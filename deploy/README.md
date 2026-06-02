# M4 Deployment (E1)

The M4 Mac mini is the sole always-on host. See the design spec:
`docs/superpowers/specs/2026-06-02-m4-deployment-e1-design.md`.

## One-time setup
1. `./deploy/pmset.sh`            # power: never sleep, auto-restart, WoL, remote login
2. Ensure FileVault is ON (System Settings → Privacy & Security). For planned
   reboots use: `sudo fdesetup authrestart`  (one unlock-free boot).
3. Fill `.env` (chmod 600) with PUSHOVER_APP_TOKEN, PUSHOVER_USER_KEY,
   HEALTHCHECKS_TICK_URL, HEALTHCHECKS_GUARD_URL.
4. Create 2 healthchecks.io checks (tick: period 1m grace 3m; guard: period 5m
   grace 11m) and wire each to the Pushover integration.
5. `mkdir -p ~/Library/Logs/quant-deploy`
6. Bring up the guard in dry-run first (edit the guard plist to add `--dry-run`),
   `./deploy/install.sh`, confirm it cannot halt live, then remove `--dry-run`
   and re-run install.

## Parallel dry-run (BEFORE retiring GH — but GH is already manual-only, so this
## is the M4 shakedown)
- Temporarily set the daily-rebalance manifest command to `["rebalance","--dry-run"]`.
- Run for several trading days. Confirm each day under `data/ops/scheduler/`:
  markers appear for premarket/rebalance/reconciliation; catch-up works after a
  forced sleep; the tick + guard healthchecks stay green; a deliberately-induced
  failure pages Pushover.

## Cutover gates (ALL must pass before live)
- [ ] Power-cycle survival: `sudo fdesetup authrestart`, confirm both agents
      reload and a "box recovered" state is observable after boot.
- [ ] Every alert channel tested: healthchecks "send test ping" → phone; a
      manual `quant` halt → emergency Pushover received & acked.
- [ ] A simulated MISSED_CRITICAL (tick after the close window) pages and does
      NOT trade.
- [ ] Dry-run outputs match the historical GH artifacts for the same days.
- [ ] Remove `--dry-run` from the rebalance manifest command; flip live.

## Operations
- Status: `launchctl print gui/$(id -u)/com.ajaiupadhyaya.quant-tick`
- Logs:   `~/Library/Logs/quant-deploy/{tick,guard}.{stdout,stderr}.log`
- Missed rebalance recovery (if still pre-close): `uv run quant ops run-job daily-rebalance --force`
- Resume after a halt (human only): `uv run quant governance resume --reason "..."`
- Uninstall: `./deploy/uninstall.sh`

## Accepted risks (M4-only + FileVault Path B)
- No off-box executor fallback; an M4 outage in the ~3-min close window = a
  missed rebalance handled manually (low-harm: daily defensive-ETF book).
- Unexpected power loss → box at FileVault unlock screen; dead-man's-switch
  pages you; recover via SSH/physical unlock.
- healthchecks.io is a 3rd-party SPOF; a weekly synthetic test push you must ack
  detects silent push death.
