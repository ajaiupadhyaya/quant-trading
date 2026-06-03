"""Pure strategy-governance classification rules."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime

from quant.governance.models import (
    GovernancePolicy,
    GovernanceState,
    StrategyState,
    ValidationEvidence,
)
from quant.strategies.base import StrategySpec

# Evidence-schema shield: how long (calendar days, anchored on the first shielded
# refresh) an incumbent LIVE strategy may be retained across a genuine methodology
# bump before the shield fails safe (quarantine). Mirrors the 30-day
# max_validation_age_days staleness philosophy; staleness itself cannot bound the
# shielded window because run_date is re-stamped on every automated re-validation.
# See docs/superpowers/specs/2026-06-02-evidence-schema-shield-design.md.
MAX_SHIELD_CALENDAR_DAYS: int = 30

_GATE_FAILURE_PREFIX = "failed_gate_"


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
    esv = evidence.evidence_schema_version if evidence is not None else 1

    if not spec.enabled_live:
        return StrategyState(
            slug=spec.slug,
            state=GovernanceState.RESEARCH,
            evaluated_at=datetime.combine(asof, datetime.min.time()),
            validation_age_days=None,
            reason_codes=["not_live_capable"],
            reason="StrategySpec.enabled_live is false; research only.",
            code_enabled_live=False,
            evidence_schema_version=esv,
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
            evidence_schema_version=esv,
        )

    if evidence.slug != spec.slug:
        reason_codes.append("evidence_slug_mismatch")
        reason_parts.append(
            f"Validation evidence slug {evidence.slug!r} does not match strategy {spec.slug!r}."
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
        evidence_schema_version=esv,
    )


def apply_schema_shield(
    provisional: StrategyState,
    *,
    evidence: ValidationEvidence | None,
    asof: date,
    prior_state: StrategyState | None,
) -> StrategyState:
    """Evidence-schema-version shield (Phase 0).

    Pure post-pass over a provisional classification. Retains an already-LIVE
    incumbent across a *genuine methodology bump* whose only quarantine cause is
    gate failures, for a bounded calendar window, loudly. It can ONLY salvage an
    incumbent (never promote), never overrides a non-gate quarantine (manual
    block, staleness, missing artifacts, slug mismatch), and fails safe when the
    calendar wall is exhausted. Returns ``provisional`` unchanged when inert.

    See docs/superpowers/specs/2026-06-02-evidence-schema-shield-design.md.
    """
    # 1. Only ever salvage a QUARANTINED provisional decision (never promote).
    if provisional.state is not GovernanceState.QUARANTINED:
        return provisional
    # 2. Need evidence to read its schema version.
    if evidence is None:
        return provisional
    # 3. Incumbent-only: prior persisted decision must have been LIVE.
    if prior_state is None or prior_state.state is not GovernanceState.LIVE:
        return provisional
    # 4. Quarantine cause must be gate-failures ONLY (no manual_block, staleness,
    #    future date, slug mismatch, missing artifacts, missing_validation, ...).
    gate_failures = [c for c in provisional.reason_codes if c.startswith(_GATE_FAILURE_PREFIX)]
    non_gate = [c for c in provisional.reason_codes if not c.startswith(_GATE_FAILURE_PREFIX)]
    if not gate_failures or non_gate:
        return provisional
    # 5. Genuine schema bump required (strict >); same-schema decay / downgrades
    #    quarantine normally — this is the self-disarm against masking alpha decay.
    if evidence.evidence_schema_version <= prior_state.evidence_schema_version:
        return provisional

    blessed = prior_state.evidence_schema_version
    first_at = prior_state.shield_first_at or asof
    days_shielded = (asof - first_at).days

    # 6. Calendar wall: fail safe (quarantine) after the bounded window.
    if days_shielded >= MAX_SHIELD_CALENDAR_DAYS:
        return replace(
            provisional,
            reason_codes=["shield_backstop_exhausted", *provisional.reason_codes],
            reason=(
                f"Evidence-schema shield backstop exhausted: incumbent retained "
                f"{days_shielded}d (limit {MAX_SHIELD_CALENDAR_DAYS}d) across schema "
                f"{blessed}->{evidence.evidence_schema_version}; quarantining (fail-safe). "
                f"Original: {provisional.reason}"
            ),
            shielded=False,
            shield_consecutive=prior_state.shield_consecutive,
            evidence_schema_version=blessed,
            shield_first_at=first_at,
        )

    # FIRE: retain LIVE, loudly, keeping the BLESSED schema version so subsequent
    # refreshes keep recognising the unresolved bump until re-bless or the wall.
    return replace(
        provisional,
        state=GovernanceState.LIVE,
        reason_codes=["schema_shield_retained_live", *provisional.reason_codes],
        reason=(
            f"Incumbent retained LIVE by evidence-schema shield: methodology "
            f"{blessed}->{evidence.evidence_schema_version} caused gate failures "
            f"({', '.join(gate_failures)}); requires human re-bless under the new "
            f"schema (shielded day {days_shielded}/{MAX_SHIELD_CALENDAR_DAYS})."
        ),
        shielded=True,
        shield_consecutive=prior_state.shield_consecutive + 1,
        evidence_schema_version=blessed,
        shield_first_at=first_at,
    )
