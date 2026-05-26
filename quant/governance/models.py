"""Typed models for strategy governance artifacts and decisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any


class GovernanceError(RuntimeError):
    """Raised when governance artifacts are missing, stale, or malformed."""


class GovernanceState(StrEnum):
    LIVE = "live"
    QUARANTINED = "quarantined"
    RESEARCH = "research"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class GovernancePolicy:
    max_validation_age_days: int = 30
    require_deflated_sharpe: bool = True
    require_probabilistic_sharpe: bool = True
    require_bootstrap_lower: bool = True
    require_regime: bool = True
    require_holdout: bool = True


@dataclass(frozen=True)
class ValidationEvidence:
    slug: str
    run_date: date
    data_start: date
    data_end: date
    gate_deflated_sharpe: bool
    gate_probabilistic_sharpe: bool
    gate_bootstrap_lower: bool
    gate_regime: bool
    gate_holdout: bool
    deflated_sharpe: float
    probabilistic_sharpe: float
    bootstrap_total_return_p05: float | None
    n_positive_regimes: int
    n_tested_regimes: int
    holdout_total_return: float | None
    chosen_params_path: str
    walkforward_path: str
    provenance: str
    manual_block: bool = False
    manual_block_reason: str | None = None

    def gate_map(self) -> dict[str, bool]:
        return {
            "deflated_sharpe": self.gate_deflated_sharpe,
            "probabilistic_sharpe": self.gate_probabilistic_sharpe,
            "bootstrap_lower": self.gate_bootstrap_lower,
            "regime": self.gate_regime,
            "holdout": self.gate_holdout,
        }

    def artifact_paths(self) -> tuple[Path, Path]:
        return Path(self.chosen_params_path), Path(self.walkforward_path)


@dataclass(frozen=True)
class StrategyState:
    slug: str
    state: GovernanceState
    evaluated_at: datetime
    validation_age_days: int | None
    reason_codes: list[str] = field(default_factory=list)
    reason: str = ""
    code_enabled_live: bool = False
    manual_block: bool = False

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["state"] = self.state.value
        payload["evaluated_at"] = self.evaluated_at.isoformat()
        return payload
