"""Reproducibility audit helpers for validation evidence."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from quant.governance.models import GovernanceError
from quant.governance.store import load_strategy_states, strategy_states_path


@dataclass(frozen=True)
class ValidationAudit:
    strategy_slug: str
    git_sha: str
    validation_command: str | None
    data_range: tuple[str | None, str | None]
    bootstrap_seed: int | None
    bootstrap_resamples: int | None
    chosen_params_hash: str | None
    walkforward_parquet_hash: str | None
    validation_report_hash: str | None
    governance_state: str | None
    reason_codes: tuple[str, ...]
    missing_artifacts: list[str]
    explanation: str


def hash_file(path: Path) -> str:
    """Return a deterministic SHA-256 hex digest for an artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_hash(path: Path, label: str, missing: list[str]) -> str | None:
    if not path.exists():
        missing.append(label)
        return None
    return hash_file(path)


def _git_sha(repo_dir: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip() or "unknown"


def _read_report(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise GovernanceError(f"Malformed validation report: {path}") from exc
    if not isinstance(payload, dict):
        raise GovernanceError(f"Malformed validation report: {path}")
    return payload


def _optional_int(raw: dict[str, Any], key: str) -> int | None:
    value = raw.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _state_reason(data_dir: Path, slug: str) -> tuple[str | None, tuple[str, ...]]:
    try:
        state = load_strategy_states(strategy_states_path(data_dir)).get(slug)
    except GovernanceError:
        return None, ()
    if state is None:
        return None, ()
    return state.state.value, tuple(state.reason_codes)


def _explanation(
    *,
    slug: str,
    report: dict[str, Any] | None,
    governance_state: str | None,
    reason_codes: tuple[str, ...],
    missing_artifacts: list[str],
) -> str:
    if report is None:
        return f"{slug}: missing validation_report.json; run quant validate {slug}."
    failures: list[str] = []
    if report.get("gate_bootstrap_lower") is False:
        failures.append("failed bootstrap lower-5% gate")
    if report.get("gate_deflated_sharpe") is False:
        failures.append("failed deflated Sharpe gate")
    if report.get("gate_probabilistic_sharpe") is False:
        failures.append("failed probabilistic Sharpe gate")
    if report.get("gate_regime") is False:
        failures.append("failed regime gate")
    if report.get("gate_holdout") is False:
        failures.append("failed holdout gate")
    if missing_artifacts:
        failures.append("missing " + ", ".join(missing_artifacts))
    if not failures:
        failures.append("validation gates passed")
    if governance_state:
        return f"{slug}: governance={governance_state}; " + "; ".join(failures)
    if reason_codes:
        return f"{slug}: reasons={', '.join(reason_codes)}; " + "; ".join(failures)
    return f"{slug}: " + "; ".join(failures)


def build_validation_audit(data_dir: Path, slug: str, *, repo_dir: Path) -> ValidationAudit:
    backtest_dir = data_dir / "backtests" / slug
    report_path = backtest_dir / "validation_report.json"
    chosen_path = backtest_dir / "chosen_params.json"
    walkforward_path = backtest_dir / "walkforward.parquet"
    missing: list[str] = []

    report = _read_report(report_path)
    validation_report_hash = _optional_hash(report_path, "validation_report.json", missing)
    chosen_hash = None
    walkforward_hash = None
    if report is not None:
        chosen_hash = _optional_hash(chosen_path, "chosen_params.json", missing)
        walkforward_hash = _optional_hash(walkforward_path, "walkforward.parquet", missing)

    governance_state, reason_codes = _state_reason(data_dir, slug)
    data_range = (
        None if report is None else str(report.get("data_start")),
        None if report is None else str(report.get("data_end")),
    )
    validation_command = None
    bootstrap_seed = None
    bootstrap_resamples = None
    if report is not None:
        validation_command = (
            str(report["validation_command"])
            if isinstance(report.get("validation_command"), str)
            else str(report.get("provenance", ""))
        )
        bootstrap_seed = _optional_int(report, "bootstrap_seed")
        bootstrap_resamples = _optional_int(report, "bootstrap_resamples")

    return ValidationAudit(
        strategy_slug=slug,
        git_sha=_git_sha(repo_dir),
        validation_command=validation_command,
        data_range=data_range,
        bootstrap_seed=bootstrap_seed,
        bootstrap_resamples=bootstrap_resamples,
        chosen_params_hash=chosen_hash,
        walkforward_parquet_hash=walkforward_hash,
        validation_report_hash=validation_report_hash,
        governance_state=governance_state,
        reason_codes=reason_codes,
        missing_artifacts=missing,
        explanation=_explanation(
            slug=slug,
            report=report,
            governance_state=governance_state,
            reason_codes=reason_codes,
            missing_artifacts=missing,
        ),
    )
