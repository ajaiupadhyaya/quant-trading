"""Deterministic JSON stores for strategy governance artifacts."""

from __future__ import annotations

import json
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


def governance_dir(data_dir: Path) -> Path:
    return data_dir / "governance"


def validation_manifest_path(data_dir: Path) -> Path:
    return governance_dir(data_dir) / VALIDATION_MANIFEST_NAME


def strategy_states_path(data_dir: Path) -> Path:
    return governance_dir(data_dir) / STRATEGY_STATES_NAME


def _date(value: str) -> date:
    return date.fromisoformat(value)


def _datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise GovernanceError(f"Missing governance artifact: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise GovernanceError(f"Malformed governance artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise GovernanceError(f"Malformed governance artifact: {path}")
    return payload


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
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_validation_manifest(path: Path) -> dict[str, ValidationEvidence]:
    payload = _load_json(path)
    strategies = payload.get("strategies")
    if not isinstance(strategies, dict):
        raise GovernanceError(f"Malformed governance artifact: {path}")
    out: dict[str, ValidationEvidence] = {}
    for slug, raw in strategies.items():
        if not isinstance(raw, dict):
            raise GovernanceError(f"Malformed governance artifact: {path}")
        out[str(slug)] = ValidationEvidence(
            slug=str(raw["slug"]),
            run_date=_date(str(raw["run_date"])),
            data_start=_date(str(raw["data_start"])),
            data_end=_date(str(raw["data_end"])),
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
            chosen_params_path=str(raw["chosen_params_path"]),
            walkforward_path=str(raw["walkforward_path"]),
            provenance=str(raw["provenance"]),
            manual_block=bool(raw.get("manual_block", False)),
            manual_block_reason=(
                None
                if raw.get("manual_block_reason") is None
                else str(raw["manual_block_reason"])
            ),
        )
    return out


def write_strategy_states(path: Path, states_by_slug: dict[str, StrategyState]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "strategies": {
            slug: state.to_json_dict() for slug, state in sorted(states_by_slug.items())
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_strategy_states(path: Path) -> dict[str, StrategyState]:
    payload = _load_json(path)
    strategies = payload.get("strategies")
    if not isinstance(strategies, dict):
        raise GovernanceError(f"Malformed governance artifact: {path}")
    out: dict[str, StrategyState] = {}
    for slug, raw in strategies.items():
        if not isinstance(raw, dict):
            raise GovernanceError(f"Malformed governance artifact: {path}")
        out[str(slug)] = StrategyState(
            slug=str(raw["slug"]),
            state=GovernanceState(str(raw["state"])),
            evaluated_at=_datetime(str(raw["evaluated_at"])),
            validation_age_days=(
                None
                if raw.get("validation_age_days") is None
                else int(raw["validation_age_days"])
            ),
            reason_codes=[str(x) for x in raw.get("reason_codes", [])],
            reason=str(raw.get("reason", "")),
            code_enabled_live=bool(raw.get("code_enabled_live", False)),
            manual_block=bool(raw.get("manual_block", False)),
        )
    return out
