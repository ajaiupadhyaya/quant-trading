# 2026-05-28 Institutional Research/Ops Handoff

Branch: `main`

This note tracks the institutional-grade research and paper-operations work
completed on 2026-05-28. The feature branch work was merged and pushed to
`main` in commit `e8ba591`, then follow-up pretrade-risk integration work
continued on `main`.

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
- Added roadmap docs:
  - `docs/institutional-research-ops.md`
  - README now references new research/risk/data/governance commands.

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

## Known Remaining Work

1. Rerun quality checks:

   ```bash
   .venv/bin/ruff check .
   .venv/bin/mypy quant
   .venv/bin/pytest
   ```

2. Fix any resulting lint/type/test failures.

3. Add richer risk decomposition:
   - equity beta, duration, commodity, gold, REIT, developed ex-US, emerging exposure.
   - report whether the book is mostly SPY/TLT beta.

4. Add full institutional evidence packet fields:
   - parameter stability analysis.
   - false-discovery/variant count tracking.
   - holdout access logging.
   - turnover/capacity/drawdown recovery/tail metrics.

5. Add strategy remediation scaffolds:
   - trend crisis sleeve and volatility target experiments.
   - momentum crash protection/defensive overlay.
   - risk-parity adaptive covariance and stress deleveraging.
   - multi-factor PIT diagnostics.
   - pairs shortability/cost realism.

6. Make ops workflows production-polished:
   - ensure GitHub shell paths are covered by tests.
   - decide whether `premarket-health` should have `contents: write` and commit health artifacts, or remain upload-only.
   - confirm schedule times in ET/UTC comments.

7. Add TUI panels for:
   - research leaderboard.
   - risk/pretrade status.
   - data quality.
   - halt status.

8. Run acceptance commands from the implementation plan:

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

9. Commit and push the current follow-up pretrade-risk changes when green.

## Current Git State At Handoff

Uncommitted follow-up changes currently exist on `main`. Key touched areas:

- `docs/notes/2026-05-28-institutional-research-ops-handoff.md`
- `quant/cli.py`
- `quant/live/rebalance.py`
- `tests/live/test_rebalance.py`
- `tests/test_cli.py`

## Suggested Resume Order Tomorrow

1. Run `git status -sb` and confirm whether the current follow-up changes are
   still uncommitted on `main`.
2. Run the remaining acceptance commands that need live/data credentials.
3. Commit and push the pretrade-risk follow-up changes.
4. Continue with portfolio risk decomposition and TUI risk/research panels.
