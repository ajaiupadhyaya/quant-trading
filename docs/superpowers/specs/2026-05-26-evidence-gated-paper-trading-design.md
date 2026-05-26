# Evidence-Gated Paper Trading Design

**Date:** 2026-05-26
**Status:** Brainstorm approved, ready for implementation planning
**Repo:** `/Users/ajaiupadhyaya/Documents/quant-trading`

## Problem Statement

The project already has a capable systematic-trading stack: strategy registry, walk-forward backtesting, validation diagnostics, live Alpaca paper execution, reconciliation, safety checks, CLI, and TUI. The immediate weakness is not lack of alpha ambition. It is governance: the live rebalance path currently trusts `StrategySpec.enabled_live`, even when the latest notes show some strategies failed validation gates, had incomplete validation, or were enabled by manual judgment.

The next phase should make paper trading fail closed. Research can remain broad and experimental, but strategies should receive paper capital only when current evidence says they are eligible.

## Goals

1. Add a strategy-governance layer that computes operational state from evidence on disk.
2. Ensure `quant rebalance` trades only governance-approved strategies by default.
3. Keep failed or unknown strategies available for research, backtests, validation, and dry-run observation.
4. Make strategy state visible in CLI output so code-level intent and evidence-level eligibility cannot silently diverge.
5. Keep artifacts simple, inspectable, commit-friendly, and consistent with the repo's existing parquet/JSON audit-trail style.

## Non-Goals

- No real-money deployment.
- No new alpha models, ML, or RL in this phase.
- No database, daemon, web app, or external scheduler changes.
- No permanent deletion of weak strategies.
- No claim that validation proves future profitability.

## Current Context

Actual CLI state on 2026-05-26 shows all five strategies live-enabled:

| Strategy | Code-level live flag | Evidence concern |
| --- | --- | --- |
| `trend` | yes | Best-supported strategy; prior notes show all gates passing. |
| `momentum` | yes | Strong DSR/PSR/bootstrap/holdout, but regime-fragile. |
| `multi-factor` | yes | Notes show validation was incomplete at go-live. |
| `risk-parity` | yes | Notes show weak validation but manually enabled for observation. |
| `pairs` | yes | Notes show weak or incomplete validation, tightened for observation. |

This is acceptable for exploratory paper trading only if the system says so explicitly. It is not acceptable as the default safety model for an autonomous 24/7 trading project.

## Operational States

Each strategy gets a computed governance state:

- `live`: eligible for normal paper rebalance capital.
- `quarantined`: known strategy, but blocked from normal rebalance because evidence is missing, stale, failed, or manually blocked.
- `research`: strategy is usable for backtests, validation, and signal exploration, but not live-capable.

The strategy class remains the source for static capability. Governance is the source for current eligibility.

## Governance Rules

A strategy is `live` only when all of these are true:

1. The strategy exists in `REGISTRY`.
2. `StrategySpec.enabled_live` is `True`.
3. A validation manifest entry exists for the strategy.
4. The validation entry is fresh enough for paper trading.
5. Required validation gates pass.
6. The entry points to existing chosen-parameter and walk-forward artifacts.
7. No manual block is active.

If any condition fails, the strategy is `quarantined` unless it is not live-capable, in which case it is `research`.

For the first implementation, "fresh enough" should default to 30 calendar days. This is conservative, easy to reason about, and can later become strategy-specific.

Required gates for version one:

- Deflated Sharpe Ratio gate passes.
- Probabilistic Sharpe Ratio gate passes.
- Bootstrap lower-5% total-return gate passes.
- Holdout gate passes.
- Regime gate passes by default.

Manual overrides should be explicit and auditable, not hidden in comments. Version one should support manual blocks. Manual live overrides can be deferred unless the implementation needs them for compatibility; if included, they must require a reason string and appear prominently in `quant governance status`.

## Artifacts

Add a small governance directory under `data/`:

```text
data/governance/
├── strategy_states.json
└── validation_manifest.json
```

`validation_manifest.json` summarizes latest evidence per strategy:

- strategy slug
- validation run date
- validation data range
- gate booleans
- DSR and PSR values
- bootstrap lower-5% total return when available
- tested/positive regime counts
- holdout total return when available
- chosen params path
- walk-forward path
- source command or provenance string

`strategy_states.json` records the computed state:

- strategy slug
- state: `live`, `quarantined`, or `research`
- reason codes
- human-readable reason
- evaluated timestamp
- validation freshness
- manual block status

Both files should be deterministic JSON so diffs stay readable in git.

## CLI Behavior

Add a `quant governance` command group:

- `quant governance status`: render each strategy with code-level live flag, governance state, validation age, gate summary, and reasons.
- `quant governance refresh`: read latest validation/backtest artifacts, recompute manifest/state, and write JSON artifacts.

Update existing commands:

- `quant strategies`: include governance state when artifacts exist; show `unknown` when they do not.
- `quant rebalance`: use governance-approved `live` strategies by default.
- `quant rebalance --dry-run --include-quarantined`: include quarantined strategies only for observation.
- `quant rebalance --include-quarantined` without `--dry-run`: fail with a clear error.

Fail-closed behavior is mandatory: if governance artifacts are missing or stale, normal rebalance must not allocate to any strategy. Dry-runs may still show research/quarantined intent when explicitly requested.

## Data Flow

```text
validation outputs + chosen params + strategy registry
        |
        v
quant governance refresh
        |
        v
data/governance/validation_manifest.json
data/governance/strategy_states.json
        |
        v
quant strategies / quant governance status / quant rebalance
```

The rebalance path should not parse HTML tear-sheets. It should consume explicit JSON governance artifacts.

## Error Handling

- Missing governance artifacts: normal rebalance fails closed with remediation text: run `quant governance refresh`.
- Stale validation: strategy is quarantined with a reason including last validation date and freshness threshold.
- Missing chosen params or walk-forward artifact: strategy is quarantined.
- Unknown strategy in manifest: ignored with a warning in status output.
- Strategy in registry but absent from manifest: quarantined if live-capable, research otherwise.
- Malformed governance JSON: normal rebalance fails closed.

## Testing

Unit tests:

- State classifier returns `live` only when all required evidence exists and passes.
- Missing manifest entry quarantines live-capable strategies.
- Stale validation quarantines otherwise-passing strategies.
- Manual block quarantines otherwise-passing strategies.
- Non-live-capable strategy becomes `research`.

CLI tests:

- `quant governance status` renders all registered strategies.
- `quant governance refresh` writes deterministic JSON.
- `quant strategies` shows governance state when available.
- Missing governance artifacts produce `unknown` in informational commands.

Rebalance tests:

- Normal rebalance excludes quarantined strategies.
- Missing or malformed governance artifacts fail closed.
- `--include-quarantined` is rejected unless `--dry-run` is set.
- Dry-run with `--include-quarantined` includes blocked strategies for observation only.

## Rollout

1. Implement pure governance models and classifier.
2. Add JSON read/write helpers.
3. Add CLI commands.
4. Wire governance filtering into rebalance.
5. Update tests and README.
6. Generate initial governance artifacts from current repo evidence.

Initial expected state should be conservative:

- `trend`: likely `live` if fresh validation artifacts can be found.
- `momentum`: likely `quarantined` unless regime gate policy is intentionally relaxed in the manifest.
- `multi-factor`: quarantined until validation evidence is complete and fresh.
- `risk-parity`: quarantined because validation notes show failed gates.
- `pairs`: quarantined until validation evidence is complete and fresh.

This leaves the project safer while preserving all research code and dry-run visibility.

## Open Follow-Up

After this phase, the next governance improvements should be:

1. Strategy-level capital allocation by evidence strength instead of equal split.
2. Paper-P&L drift monitoring versus backtest expectations.
3. Automatic scheduled validation refresh.
4. Separate promotion workflow for real-money eligibility.
