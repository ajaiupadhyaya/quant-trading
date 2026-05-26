"""Pure strategy-governance classification rules."""

from __future__ import annotations

from datetime import date, datetime

from quant.governance.models import (
    GovernancePolicy,
    GovernanceState,
    StrategyState,
    ValidationEvidence,
)
from quant.strategies.base import StrategySpec


def classify_strategy(
    *,
    spec: StrategySpec,
    evidence: ValidationEvidence | None,
    policy: GovernancePolicy,
    asof: date,
) -> StrategyState:
    reason_codes: list[str] = []
    reason_parts: list[str] = []
    validation_age_days: int | None = None

    if not spec.enabled_live:
        return StrategyState(
            slug=spec.slug,
            state=GovernanceState.RESEARCH,
            evaluated_at=datetime.combine(asof, datetime.min.time()),
            validation_age_days=None,
            reason_codes=["not_live_capable"],
            reason="StrategySpec.enabled_live is false; research only.",
            code_enabled_live=False,
        )

    if evidence is None:
        return StrategyState(
            slug=spec.slug,
            state=GovernanceState.QUARANTINED,
            evaluated_at=datetime.combine(asof, datetime.min.time()),
            validation_age_days=None,
            reason_codes=["missing_validation"],
            reason="No validation evidence exists for this live-capable strategy.",
            code_enabled_live=True,
        )

    validation_age_days = (asof - evidence.run_date).days
    if validation_age_days < 0:
        reason_codes.append("future_validation_date")
        reason_parts.append(f"Validation run date {evidence.run_date} is after {asof}.")
    if validation_age_days > policy.max_validation_age_days:
        reason_codes.append("stale_validation")
        reason_parts.append(
            f"Validation is {validation_age_days} days old; limit is "
            f"{policy.max_validation_age_days} days."
        )

    gate_requirements = {
        "deflated_sharpe": policy.require_deflated_sharpe,
        "probabilistic_sharpe": policy.require_probabilistic_sharpe,
        "bootstrap_lower": policy.require_bootstrap_lower,
        "regime": policy.require_regime,
        "holdout": policy.require_holdout,
    }
    for gate, required in gate_requirements.items():
        if required and not evidence.gate_map()[gate]:
            reason_codes.append(f"failed_gate_{gate}")
            reason_parts.append(f"Required gate failed: {gate}.")

    chosen_path, walkforward_path = evidence.artifact_paths()
    if not chosen_path.exists():
        reason_codes.append("missing_chosen_params")
        reason_parts.append(f"Missing chosen params artifact: {chosen_path}.")
    if not walkforward_path.exists():
        reason_codes.append("missing_walkforward")
        reason_parts.append(f"Missing walk-forward artifact: {walkforward_path}.")

    if evidence.manual_block:
        reason_codes.append("manual_block")
        block_reason = evidence.manual_block_reason or "manual block is active"
        reason_parts.append(f"Manual block: {block_reason}.")

    state = GovernanceState.LIVE if not reason_codes else GovernanceState.QUARANTINED
    reason = (
        "Fresh validation evidence passes all required gates."
        if state is GovernanceState.LIVE
        else " ".join(reason_parts)
    )
    return StrategyState(
        slug=spec.slug,
        state=state,
        evaluated_at=datetime.combine(asof, datetime.min.time()),
        validation_age_days=validation_age_days,
        reason_codes=reason_codes,
        reason=reason,
        code_enabled_live=spec.enabled_live,
        manual_block=evidence.manual_block,
    )
