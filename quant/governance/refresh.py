"""Build governance artifacts from validation sidecars and the strategy registry."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from quant.governance.models import GovernancePolicy, StrategyState, ValidationEvidence
from quant.governance.policy import classify_strategy
from quant.governance.store import (
    strategy_states_path,
    validation_manifest_path,
    write_strategy_states,
    write_validation_manifest,
)


def validation_report_path(data_dir: Path, slug: str) -> Path:
    return data_dir / "backtests" / slug / "validation_report.json"


def _read_validation_report(data_dir: Path, slug: str) -> dict[str, Any] | None:
    path = validation_report_path(data_dir, slug)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    return payload


def validation_report_to_evidence(data_dir: Path, slug: str) -> ValidationEvidence | None:
    raw = _read_validation_report(data_dir, slug)
    if raw is None:
        return None
    backtest_dir = data_dir / "backtests" / slug
    return ValidationEvidence(
        slug=str(raw["slug"]),
        run_date=date.fromisoformat(str(raw["run_date"])),
        data_start=date.fromisoformat(str(raw["data_start"])),
        data_end=date.fromisoformat(str(raw["data_end"])),
        gate_deflated_sharpe=bool(raw["gate_deflated_sharpe"]),
        gate_probabilistic_sharpe=bool(raw["gate_probabilistic_sharpe"]),
        gate_bootstrap_lower=bool(raw["gate_bootstrap_lower"]),
        gate_regime=bool(raw["gate_regime"]),
        gate_holdout=bool(raw["gate_holdout"]),
        deflated_sharpe=float(raw["deflated_sharpe"]),
        probabilistic_sharpe=float(raw["probabilistic_sharpe"]),
        bootstrap_total_return_p05=(
            None
            if raw.get("bootstrap_total_return_p05") is None
            else float(raw["bootstrap_total_return_p05"])
        ),
        n_positive_regimes=int(raw["n_positive_regimes"]),
        n_tested_regimes=int(raw["n_tested_regimes"]),
        holdout_total_return=(
            None
            if raw.get("holdout_total_return") is None
            else float(raw["holdout_total_return"])
        ),
        chosen_params_path=str(backtest_dir / "chosen_params.json"),
        walkforward_path=str(backtest_dir / "walkforward.parquet"),
        provenance=str(
            raw.get(
                "provenance",
                f"validation_report:{validation_report_path(data_dir, slug)}",
            )
        ),
        manual_block=bool(raw.get("manual_block", False)),
        manual_block_reason=(
            None
            if raw.get("manual_block_reason") is None
            else str(raw["manual_block_reason"])
        ),
    )


def build_governance_artifacts(
    *,
    data_dir: Path,
    registry: dict[str, Any],
    policy: GovernancePolicy,
    asof: date,
) -> dict[str, StrategyState]:
    evidence_by_slug: dict[str, ValidationEvidence] = {}
    states: dict[str, StrategyState] = {}
    for slug, strategy_cls in sorted(registry.items()):
        evidence = validation_report_to_evidence(data_dir, slug)
        if evidence is not None:
            evidence_by_slug[slug] = evidence
        states[slug] = classify_strategy(
            spec=strategy_cls.spec,
            evidence=evidence,
            policy=policy,
            asof=asof,
        )
    write_validation_manifest(validation_manifest_path(data_dir), evidence_by_slug)
    write_strategy_states(strategy_states_path(data_dir), states)
    return states
