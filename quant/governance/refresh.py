"""Build governance artifacts from validation sidecars and the strategy registry."""

from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path
from typing import Any

from quant.governance.allocation import allocate_capital
from quant.governance.models import (
    GovernanceError,
    GovernancePolicy,
    StrategyState,
    ValidationEvidence,
)
from quant.governance.policy import apply_schema_shield, classify_strategy
from quant.governance.store import (
    allocation_path,
    load_strategy_states,
    strategy_states_path,
    validation_manifest_path,
    write_allocation,
    write_strategy_states,
    write_validation_manifest,
)


def validation_report_path(data_dir: Path, slug: str) -> Path:
    return data_dir / "backtests" / slug / "validation_report.json"


def _read_validation_report(data_dir: Path, slug: str) -> dict[str, Any] | None:
    path = validation_report_path(data_dir, slug)
    if not path.exists():
        return None
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value {value}")
            ),
        )
    except Exception as exc:
        raise GovernanceError(f"Malformed validation report: {path}") from exc
    if not isinstance(payload, dict):
        raise GovernanceError(f"Malformed validation report: {path}")
    return payload


def _malformed_report(data_dir: Path, slug: str) -> GovernanceError:
    return GovernanceError(f"Malformed validation report: {validation_report_path(data_dir, slug)}")


def _expect_str(raw: dict[str, Any], key: str, data_dir: Path, slug: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str):
        raise _malformed_report(data_dir, slug)
    return value


def _expect_bool(raw: dict[str, Any], key: str, data_dir: Path, slug: str) -> bool:
    value = raw.get(key)
    if not isinstance(value, bool):
        raise _malformed_report(data_dir, slug)
    return value


def _expect_int(raw: dict[str, Any], key: str, data_dir: Path, slug: str) -> int:
    value = raw.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise _malformed_report(data_dir, slug)
    return value


def _expect_number(raw: dict[str, Any], key: str, data_dir: Path, slug: str) -> float:
    value = raw.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise _malformed_report(data_dir, slug)
    out = float(value)
    if not math.isfinite(out):
        raise _malformed_report(data_dir, slug)
    return out


def _expect_optional_number(
    raw: dict[str, Any], key: str, data_dir: Path, slug: str
) -> float | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise _malformed_report(data_dir, slug)
    out = float(value)
    if not math.isfinite(out):
        raise _malformed_report(data_dir, slug)
    return out


def _report_schema_version(raw: dict[str, Any], data_dir: Path, slug: str) -> int:
    """Evidence schema version: ABSENT (pre-shield sidecar) -> 1; present-but-malformed -> raise."""
    value = raw.get("evidence_schema_version")
    if value is None:
        return 1
    if not isinstance(value, int) or isinstance(value, bool):
        raise _malformed_report(data_dir, slug)
    return value


def _expect_optional_str(raw: dict[str, Any], key: str, data_dir: Path, slug: str) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise _malformed_report(data_dir, slug)
    return value


def validation_report_to_evidence(data_dir: Path, slug: str) -> ValidationEvidence | None:
    raw = _read_validation_report(data_dir, slug)
    if raw is None:
        return None
    backtest_dir = data_dir / "backtests" / slug
    try:
        provenance = raw.get("provenance")
        if provenance is None:
            provenance = f"validation_report:{validation_report_path(data_dir, slug)}"
        elif not isinstance(provenance, str):
            raise _malformed_report(data_dir, slug)

        return ValidationEvidence(
            slug=_expect_str(raw, "slug", data_dir, slug),
            run_date=date.fromisoformat(_expect_str(raw, "run_date", data_dir, slug)),
            data_start=date.fromisoformat(_expect_str(raw, "data_start", data_dir, slug)),
            data_end=date.fromisoformat(_expect_str(raw, "data_end", data_dir, slug)),
            gate_deflated_sharpe=_expect_bool(raw, "gate_deflated_sharpe", data_dir, slug),
            gate_probabilistic_sharpe=_expect_bool(
                raw, "gate_probabilistic_sharpe", data_dir, slug
            ),
            gate_bootstrap_lower=_expect_bool(raw, "gate_bootstrap_lower", data_dir, slug),
            gate_regime=_expect_bool(raw, "gate_regime", data_dir, slug),
            gate_holdout=_expect_bool(raw, "gate_holdout", data_dir, slug),
            deflated_sharpe=_expect_number(raw, "deflated_sharpe", data_dir, slug),
            probabilistic_sharpe=_expect_number(raw, "probabilistic_sharpe", data_dir, slug),
            bootstrap_total_return_p05=_expect_optional_number(
                raw, "bootstrap_total_return_p05", data_dir, slug
            ),
            n_positive_regimes=_expect_int(raw, "n_positive_regimes", data_dir, slug),
            n_tested_regimes=_expect_int(raw, "n_tested_regimes", data_dir, slug),
            holdout_total_return=_expect_optional_number(
                raw, "holdout_total_return", data_dir, slug
            ),
            chosen_params_path=str(backtest_dir / "chosen_params.json"),
            walkforward_path=str(backtest_dir / "walkforward.parquet"),
            provenance=provenance,
            manual_block=(
                False
                if raw.get("manual_block") is None
                else _expect_bool(raw, "manual_block", data_dir, slug)
            ),
            manual_block_reason=_expect_optional_str(raw, "manual_block_reason", data_dir, slug),
            evidence_schema_version=_report_schema_version(raw, data_dir, slug),
        )
    except (KeyError, ValueError) as exc:
        raise _malformed_report(data_dir, slug) from exc


def build_governance_artifacts(
    *,
    data_dir: Path,
    registry: dict[str, Any],
    policy: GovernancePolicy,
    asof: date,
) -> dict[str, StrategyState]:
    evidence_by_slug: dict[str, ValidationEvidence] = {}
    states: dict[str, StrategyState] = {}
    # Prior persisted decision = last refresh's blessed state (read BEFORE we
    # overwrite it below). Absent -> no incumbent (shield inert, fail-closed);
    # present-but-malformed -> GovernanceError propagates (fail loud, never
    # silently degraded to "no incumbent" which would disarm the shield).
    states_path = strategy_states_path(data_dir)
    prior_states = load_strategy_states(states_path) if states_path.exists() else {}
    for slug, strategy_cls in sorted(registry.items()):
        evidence = validation_report_to_evidence(data_dir, slug)
        if evidence is not None:
            evidence_by_slug[slug] = evidence
        provisional = classify_strategy(
            spec=strategy_cls.spec,
            evidence=evidence,
            policy=policy,
            asof=asof,
        )
        states[slug] = apply_schema_shield(
            provisional,
            evidence=evidence,
            asof=asof,
            prior_state=prior_states.get(slug),
        )
    write_validation_manifest(validation_manifest_path(data_dir), evidence_by_slug)
    write_strategy_states(strategy_states_path(data_dir), states)
    write_allocation(
        allocation_path(data_dir),
        allocate_capital(states, evidence_by_slug=evidence_by_slug),
    )
    return states
