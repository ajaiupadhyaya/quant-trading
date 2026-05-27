"""Deterministic JSON stores for strategy governance artifacts."""

from __future__ import annotations

import json
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any

from quant.governance.models import (
    GovernanceError,
    GovernanceState,
    StrategyState,
    ValidationEvidence,
)

VALIDATION_MANIFEST_NAME = "validation_manifest.json"
STRATEGY_STATES_NAME = "strategy_states.json"
ALLOCATION_NAME = "allocation.json"
DRIFT_REPORT_NAME = "drift_report.json"


def governance_dir(data_dir: Path) -> Path:
    return data_dir / "governance"


def validation_manifest_path(data_dir: Path) -> Path:
    return governance_dir(data_dir) / VALIDATION_MANIFEST_NAME


def strategy_states_path(data_dir: Path) -> Path:
    return governance_dir(data_dir) / STRATEGY_STATES_NAME


def allocation_path(data_dir: Path) -> Path:
    return governance_dir(data_dir) / ALLOCATION_NAME


def drift_report_path(data_dir: Path) -> Path:
    return governance_dir(data_dir) / DRIFT_REPORT_NAME


def _date(value: str) -> date:
    return date.fromisoformat(value)


def _datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise GovernanceError(f"Missing governance artifact: {path}")
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value {value}")
            ),
        )
    except Exception as exc:
        raise GovernanceError(f"Malformed governance artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise GovernanceError(f"Malformed governance artifact: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    try:
        text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    except ValueError as exc:
        raise GovernanceError(f"Cannot write governance artifact with non-finite values: {path}") from exc
    path.write_text(text, encoding="utf-8")


def _malformed(path: Path) -> GovernanceError:
    return GovernanceError(f"Malformed governance artifact: {path}")


def _expect_str(raw: dict[str, Any], key: str, path: Path) -> str:
    value = raw.get(key)
    if not isinstance(value, str):
        raise _malformed(path)
    return value


def _expect_bool(raw: dict[str, Any], key: str, path: Path) -> bool:
    value = raw.get(key)
    if not isinstance(value, bool):
        raise _malformed(path)
    return value


def _expect_int(raw: dict[str, Any], key: str, path: Path) -> int:
    value = raw.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise _malformed(path)
    return value


def _expect_optional_int(raw: dict[str, Any], key: str, path: Path) -> int | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise _malformed(path)
    return value


def _expect_number(raw: dict[str, Any], key: str, path: Path) -> float:
    value = raw.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise _malformed(path)
    out = float(value)
    if not math.isfinite(out):
        raise _malformed(path)
    return out


def _expect_optional_number(raw: dict[str, Any], key: str, path: Path) -> float | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise _malformed(path)
    out = float(value)
    if not math.isfinite(out):
        raise _malformed(path)
    return out


def _expect_optional_str(raw: dict[str, Any], key: str, path: Path) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise _malformed(path)
    return value


def _expect_str_list(raw: dict[str, Any], key: str, path: Path) -> list[str]:
    value = raw.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise _malformed(path)
    return value


def write_validation_manifest(
    path: Path, evidence_by_slug: dict[str, ValidationEvidence]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "strategies": {
            slug: {
                "slug": evidence.slug,
                "run_date": evidence.run_date.isoformat(),
                "data_start": evidence.data_start.isoformat(),
                "data_end": evidence.data_end.isoformat(),
                "gate_deflated_sharpe": evidence.gate_deflated_sharpe,
                "gate_probabilistic_sharpe": evidence.gate_probabilistic_sharpe,
                "gate_bootstrap_lower": evidence.gate_bootstrap_lower,
                "gate_regime": evidence.gate_regime,
                "gate_holdout": evidence.gate_holdout,
                "deflated_sharpe": evidence.deflated_sharpe,
                "probabilistic_sharpe": evidence.probabilistic_sharpe,
                "bootstrap_total_return_p05": evidence.bootstrap_total_return_p05,
                "n_positive_regimes": evidence.n_positive_regimes,
                "n_tested_regimes": evidence.n_tested_regimes,
                "holdout_total_return": evidence.holdout_total_return,
                "chosen_params_path": evidence.chosen_params_path,
                "walkforward_path": evidence.walkforward_path,
                "provenance": evidence.provenance,
                "manual_block": evidence.manual_block,
                "manual_block_reason": evidence.manual_block_reason,
            }
            for slug, evidence in sorted(evidence_by_slug.items())
        },
    }
    _write_json(path, payload)


def load_validation_manifest(path: Path) -> dict[str, ValidationEvidence]:
    payload = _load_json(path)
    strategies = payload.get("strategies")
    if not isinstance(strategies, dict):
        raise GovernanceError(f"Malformed governance artifact: {path}")
    out: dict[str, ValidationEvidence] = {}
    for slug, raw in strategies.items():
        if not isinstance(raw, dict):
            raise GovernanceError(f"Malformed governance artifact: {path}")
        try:
            out[str(slug)] = ValidationEvidence(
                slug=_expect_str(raw, "slug", path),
                run_date=_date(_expect_str(raw, "run_date", path)),
                data_start=_date(_expect_str(raw, "data_start", path)),
                data_end=_date(_expect_str(raw, "data_end", path)),
                gate_deflated_sharpe=_expect_bool(raw, "gate_deflated_sharpe", path),
                gate_probabilistic_sharpe=_expect_bool(raw, "gate_probabilistic_sharpe", path),
                gate_bootstrap_lower=_expect_bool(raw, "gate_bootstrap_lower", path),
                gate_regime=_expect_bool(raw, "gate_regime", path),
                gate_holdout=_expect_bool(raw, "gate_holdout", path),
                deflated_sharpe=_expect_number(raw, "deflated_sharpe", path),
                probabilistic_sharpe=_expect_number(raw, "probabilistic_sharpe", path),
                bootstrap_total_return_p05=_expect_optional_number(
                    raw, "bootstrap_total_return_p05", path
                ),
                n_positive_regimes=_expect_int(raw, "n_positive_regimes", path),
                n_tested_regimes=_expect_int(raw, "n_tested_regimes", path),
                holdout_total_return=_expect_optional_number(raw, "holdout_total_return", path),
                chosen_params_path=_expect_str(raw, "chosen_params_path", path),
                walkforward_path=_expect_str(raw, "walkforward_path", path),
                provenance=_expect_str(raw, "provenance", path),
                manual_block=_expect_bool(raw, "manual_block", path),
                manual_block_reason=_expect_optional_str(raw, "manual_block_reason", path),
            )
        except (KeyError, ValueError) as exc:
            raise _malformed(path) from exc
    return out


def write_strategy_states(path: Path, states_by_slug: dict[str, StrategyState]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "strategies": {
            slug: state.to_json_dict() for slug, state in sorted(states_by_slug.items())
        },
    }
    _write_json(path, payload)


def write_allocation(path: Path, allocations_by_slug: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "allocations": {
            slug: float(weight) for slug, weight in sorted(allocations_by_slug.items())
        },
    }
    _write_json(path, payload)


def load_allocation(path: Path) -> dict[str, float]:
    payload = _load_json(path)
    allocations = payload.get("allocations")
    if not isinstance(allocations, dict):
        raise GovernanceError(f"Malformed governance artifact: {path}")
    out: dict[str, float] = {}
    for slug, weight in allocations.items():
        if not isinstance(slug, str):
            raise _malformed(path)
        if not isinstance(weight, int | float) or isinstance(weight, bool):
            raise _malformed(path)
        weight_float = float(weight)
        if not math.isfinite(weight_float) or weight_float < 0:
            raise _malformed(path)
        out[slug] = weight_float
    return out


def load_strategy_states(path: Path) -> dict[str, StrategyState]:
    payload = _load_json(path)
    strategies = payload.get("strategies")
    if not isinstance(strategies, dict):
        raise GovernanceError(f"Malformed governance artifact: {path}")
    out: dict[str, StrategyState] = {}
    for slug, raw in strategies.items():
        if not isinstance(raw, dict):
            raise GovernanceError(f"Malformed governance artifact: {path}")
        try:
            out[str(slug)] = StrategyState(
                slug=_expect_str(raw, "slug", path),
                state=GovernanceState(_expect_str(raw, "state", path)),
                evaluated_at=_datetime(_expect_str(raw, "evaluated_at", path)),
                validation_age_days=_expect_optional_int(raw, "validation_age_days", path),
                reason_codes=_expect_str_list(raw, "reason_codes", path),
                reason=_expect_str(raw, "reason", path),
                code_enabled_live=_expect_bool(raw, "code_enabled_live", path),
                manual_block=_expect_bool(raw, "manual_block", path),
            )
        except (KeyError, ValueError) as exc:
            raise _malformed(path) from exc
    return out
