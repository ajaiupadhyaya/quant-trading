# Quant Trading Operator Runbook

This system is built for Alpaca paper trading. Real-money deployment is out of
scope until a separate risk, compliance, and capital-control review exists.

## Startup

1. Install dependencies:

   ```bash
   uv sync --all-extras
   ```

2. Confirm paper-trading credentials are loaded in `.env`:

   ```bash
   uv run quant doctor
   ```

3. Refresh local market data:

   ```bash
   uv run quant data refresh --start 2010-01-01
   ```

4. Rebuild governance from validation evidence:

   ```bash
   uv run quant governance refresh
   uv run quant governance status
   ```

## Daily Check

Run this before the scheduled paper rebalance or whenever the machine has been
offline:

```bash
uv run quant doctor
uv run quant governance status
uv run quant rebalance --dry-run
```

Expected healthy state:

- At least one strategy is `live`.
- Allocation is non-zero only for `live` strategies.
- Quarantined strategies show an explicit reason.
- Dry-run prints a concrete order plan or a specific fail-closed reason.

## Paper Rebalance

Dry-run first:

```bash
uv run quant rebalance --dry-run
```

Submit paper orders only after dry-run output is sensible:

```bash
uv run quant rebalance
```

Never use `--include-quarantined` outside dry-run. The CLI rejects that path by
design.

## Weekly Validation

Use the GitHub workflow when possible:

```bash
gh workflow run weekly-validation-governance.yml
```

Local equivalent for the current production baseline:

```bash
uv run quant validate defensive-etf-allocation --bootstrap-resamples 5000
uv run quant governance refresh
uv run quant governance status
```

Full-grid `multi-factor` and `pairs` validation belongs in the scheduled
workflow because runtime is intentionally bounded per strategy.

## Drift And Reconciliation

Generate advisory paper P&L drift flags:

```bash
uv run quant governance drift
```

Generate signal-to-fill and execution-cost reconciliation:

```bash
uv run python scripts/reconcile_live.py
```

`no_mid_price` means Alpaca minute bars were unavailable for a fill-time
midpoint; it is visible telemetry, not silent success.

## Emergency Stop

1. Pause the GitHub Actions workflow `daily-rebalance`.
2. Block paper capital by setting a manual block in validation evidence or by
   removing live-passing evidence, then run:

   ```bash
   uv run quant governance refresh
   uv run quant governance status
   ```

3. Verify dry-run is fail-closed:

   ```bash
   uv run quant rebalance --dry-run
   ```

4. If Alpaca positions must be flattened, do that in Alpaca paper UI or a
   separately reviewed liquidation script.

## Recovery

After a crash, reboot, or network outage:

```bash
git pull --ff-only
uv sync --all-extras
uv run quant doctor
uv run quant governance status
uv run quant governance drift
uv run quant rebalance --dry-run
```

If governance artifacts are missing or malformed, the rebalance path fails
closed. Re-run validation for the baseline and refresh governance before
resuming paper orders.
