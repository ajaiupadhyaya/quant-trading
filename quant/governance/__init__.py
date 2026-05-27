"""Strategy governance: evidence-gated paper-trading eligibility."""

from quant.governance.allocation import AllocationConfig, allocate_capital
from quant.governance.audit import ValidationAudit, build_validation_audit, hash_file
from quant.governance.drift import DriftConfig, DriftRow, drift_flag, summarize_drift
from quant.governance.models import (
    GovernanceError,
    GovernancePolicy,
    GovernanceState,
    StrategyState,
    ValidationEvidence,
)
from quant.governance.policy import classify_strategy
from quant.governance.refresh import (
    build_governance_artifacts,
    validation_report_path,
    validation_report_to_evidence,
)
from quant.governance.store import (
    governance_dir,
    load_strategy_states,
    load_validation_manifest,
    strategy_states_path,
    validation_manifest_path,
    write_strategy_states,
    write_validation_manifest,
)

__all__ = [
    "AllocationConfig",
    "DriftConfig",
    "DriftRow",
    "GovernanceError",
    "GovernancePolicy",
    "GovernanceState",
    "StrategyState",
    "ValidationAudit",
    "ValidationEvidence",
    "allocate_capital",
    "build_governance_artifacts",
    "build_validation_audit",
    "classify_strategy",
    "drift_flag",
    "governance_dir",
    "hash_file",
    "load_strategy_states",
    "load_validation_manifest",
    "strategy_states_path",
    "summarize_drift",
    "validation_manifest_path",
    "validation_report_path",
    "validation_report_to_evidence",
    "write_strategy_states",
    "write_validation_manifest",
]
