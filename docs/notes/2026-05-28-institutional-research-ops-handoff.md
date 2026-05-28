# 2026-05-28 Institutional Research/Ops Handoff

Branch: `main`

This note tracks the institutional-grade research and paper-operations work
completed on 2026-05-28. The feature branch work was merged and pushed to
`main` in commit `e8ba591`, then follow-up pretrade-risk integration work
continued on `main` through `4b46928`. A final market-open readiness pass was
completed and pushed in `f3526c3`.

## Completed This Session

- Created a feature branch: `codex/institutional-research-ops`.
- Added an append-only research experiment registry:
  - `quant/research/registry.py`
  - `data/research/experiments.jsonl` as the intended artifact path
  - CLI commands: `quant research list`, `show`, `compare`, `leaderboard`
  - `quant validate` now appends validation experiment records with strategy,
    git SHA, command, params, metrics, gates, artifacts, wall time, and data
    snapshot id.
- Added immutable data snapshot support:
  - `quant/data/snapshot.py`
  - CLI command: `quant data snapshot`
  - snapshot manifests written under `data/snapshots/<snapshot-id>/manifest.json`
  - manifests hash raw parquet inputs and record symbol coverage.
- Added data quality gates:
  - `quant/data/quality.py`
  - CLI command: `quant data quality`
  - reports missing bars, duplicate timestamps, impossible OHLC, stale symbols
  - writes `data/ops/health/data_quality.json`.
- Added paid-data-ready provider boundary:
  - `quant/data/providers.py`
  - `BarProvider` protocol plus `get_provider_bars`
  - test proves provider substitution returns strategy-compatible MultiIndex bars.
- Added emergency halt/resume:
  - `quant/governance/halt.py`
  - CLI commands: `quant governance halt --reason ...` and `quant governance resume --reason ...`
  - `run_rebalance` now blocks non-dry-run execution when halt is active.
- Added portfolio pretrade risk primitive:
  - `quant/risk/pretrade.py`
  - CLI command: `quant risk pretrade`
  - writes `data/risk/pretrade_report.json`
  - builds a side-effect-free dry-run rebalance order plan.
  - reports proposed-order gross exposure, per-symbol concentration, reference
    prices, strategy outcomes, violations, and skipped reasons.
- Added rebalance planning mode:
  - `run_rebalance(..., record_bookkeeping=False)` computes the same proposed
    orders without appending equity rows, trades, or strategy-position snapshots.
  - strategy outcomes now include latest reference prices from the strategy bar
    cache for use by pretrade risk.
- Added ops workflows:
  - `.github/workflows/premarket-health.yml`
  - `.github/workflows/posttrade-reconciliation.yml`
  - daily rebalance now runs data quality and pretrade risk, uploads ops artifacts, and commits `data/ops/health/` and `data/risk/`.
  - premarket and daily rebalance data-quality checks now target the live
    defensive ETF universe (`SPY, TLT, IEF, GLD, DBC, VNQ, EFA, EEM`) instead
    of every cached research symbol.
- Added roadmap docs:
  - `docs/institutional-research-ops.md`
  - README now references new research/risk/data/governance commands.
- Completed market-open readiness fixes:
  - `quant doctor` now derives readiness from governance-live strategies, not
    broad code-level `enabled_live` flags. This prevents quarantined research
    strategy snapshots from blocking the actual evidence-gated live path.
  - `quant data quality` uses the embedded NYSE trading calendar instead of a
    plain weekday calendar, so market holidays are not counted as missing bars.
  - `quant data quality` defaults to the last completed trading session rather
    than the current in-progress session.
  - Added the 2025-01-09 Jimmy Carter national day of mourning closure to the
    embedded NYSE calendar.

## Tests Already Run

Passing focused tests:

```bash
.venv/bin/pytest tests/research/test_registry.py tests/data/test_snapshot_quality.py tests/risk/test_pretrade.py tests/governance/test_halt.py
.venv/bin/pytest tests/test_cli.py::test_research_cli_lists_and_compares_experiments tests/test_cli.py::test_data_snapshot_and_quality_commands_write_artifacts tests/test_cli.py::test_risk_pretrade_command_writes_report tests/test_cli.py::test_governance_halt_and_resume_commands tests/live/test_rebalance.py::test_emergency_halt_blocks_non_dry_run
.venv/bin/pytest tests/test_cli.py::test_validate_command_runs_to_completion_on_known_strategy
.venv/bin/pytest tests/data/test_providers.py
```

Additional passing checks after the pretrade-risk integration:

```bash
.venv/bin/pytest tests/live/test_rebalance.py::test_planning_mode_does_not_write_bookkeeping tests/test_cli.py::test_risk_pretrade_command_writes_report tests/risk/test_pretrade.py
.venv/bin/ruff check .
.venv/bin/mypy quant
.venv/bin/pytest tests/live/test_rebalance.py tests/test_cli.py tests/risk/test_pretrade.py
.venv/bin/pytest
```

The broader nearby test slice passed with `46 passed, 1 warning`. Full
repository `pytest` passed with `414 passed, 1 warning`.

Final market-open readiness checks:

```bash
.venv/bin/ruff check .
.venv/bin/mypy quant
.venv/bin/pytest
.venv/bin/quant doctor
.venv/bin/quant data quality --start 2018-01-01 --symbols SPY,TLT,IEF,GLD,DBC,VNQ,EFA,EEM
.venv/bin/quant risk pretrade
```

Results:

- `ruff`: passed.
- `mypy`: passed.
- full `pytest`: `418 passed, 1 warning`.
- `quant doctor`: `7/7 checks passed`; Alpaca paper account reachable; one
  governance-live strategy: `defensive-etf-allocation`.
- live ETF data quality: passed with zero missing bars, zero duplicate
  timestamps, and zero bad OHLC rows for the live ETF universe.
- pretrade risk: passed; proposed dry-run orders were buys for `DBC`, `EEM`,
  and `SPY`, with no risk violations.

## Known Remaining Work

1. Add richer risk decomposition:
   - equity beta, duration, commodity, gold, REIT, developed ex-US, emerging exposure.
   - report whether the book is mostly SPY/TLT beta.

2. Add full institutional evidence packet fields:
   - parameter stability analysis.
   - false-discovery/variant count tracking.
   - holdout access logging.
   - turnover/capacity/drawdown recovery/tail metrics.

3. Add strategy remediation scaffolds:
   - trend crisis sleeve and volatility target experiments.
   - momentum crash protection/defensive overlay.
   - risk-parity adaptive covariance and stress deleveraging.
   - multi-factor PIT diagnostics.
   - pairs shortability/cost realism.

4. Make ops workflows production-polished:
   - ensure GitHub shell paths are covered by tests.
   - decide whether `premarket-health` should have `contents: write` and commit health artifacts, or remain upload-only.
   - confirm schedule times in ET/UTC comments.

5. Add TUI panels for:
   - research leaderboard.
   - risk/pretrade status.
   - data quality.
   - halt status.

6. Run periodic acceptance commands after future strategy/data changes:

   ```bash
   uv run pytest
   uv run ruff check .
   uv run mypy quant
   uv run quant data quality
   uv run quant validate defensive-etf-allocation --bootstrap-resamples 5000
   uv run quant governance refresh
   uv run quant governance status
   uv run quant risk pretrade
   uv run quant rebalance --dry-run
   ```

## Current Git State At Handoff

Before this documentation update, `main` was clean and synchronized with
`origin/main` at `f3526c3`.

Latest relevant pushed commits:

- `f3526c3 fix(ops): align readiness checks with governance live trading`
- `4b46928 feat(risk): build pretrade reports from rebalance plans`
- `e8ba591 feat(research): add institutional audit and ops layer`

## Suggested Resume Order Tomorrow

1. Check GitHub Actions for the premarket health and daily rebalance runs.
2. Inspect `data/risk/pretrade_report.json` and `data/ops/health/data_quality.json`.
3. Run `quant governance status`, `quant risk pretrade`, and
   `quant rebalance --dry-run` before any manual intervention.
4. Continue with portfolio risk decomposition and TUI risk/research panels.
