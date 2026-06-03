"""Tests for the evidence-schema-version shield (Phase 0, step 1).

Spec: docs/superpowers/specs/2026-06-02-evidence-schema-shield-design.md
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

from quant.backtest.validation import EVIDENCE_SCHEMA_VERSION
from quant.governance.models import GovernanceState, StrategyState, ValidationEvidence
from quant.governance.policy import MAX_SHIELD_CALENDAR_DAYS, apply_schema_shield

ASOF = date(2026, 6, 6)


def _evidence(*, schema: int = 2, slug: str = "defensive-etf-allocation") -> ValidationEvidence:
    """A FAILING (deflated-sharpe) evidence at the given schema version."""
    return ValidationEvidence(
        slug=slug,
        run_date=ASOF,
        data_start=date(2010, 1, 1),
        data_end=ASOF,
        gate_deflated_sharpe=False,
        gate_probabilistic_sharpe=True,
        gate_bootstrap_lower=True,
        gate_regime=True,
        gate_holdout=True,
        deflated_sharpe=0.246,
        probabilistic_sharpe=0.98,
        bootstrap_total_return_p05=0.05,
        n_positive_regimes=3,
        n_tested_regimes=3,
        holdout_total_return=0.2,
        chosen_params_path="x/chosen_params.json",
        walkforward_path="x/walkforward.parquet",
        provenance="unit test",
        evidence_schema_version=schema,
    )


def _prior_live(
    *,
    blessed_schema: int = 1,
    shielded: bool = False,
    shield_first_at: date | None = None,
    shield_consecutive: int = 0,
) -> StrategyState:
    return StrategyState(
        slug="defensive-etf-allocation",
        state=GovernanceState.LIVE,
        evaluated_at=datetime(2026, 6, 1),
        validation_age_days=0,
        reason_codes=[],
        reason="ok",
        code_enabled_live=True,
        shielded=shielded,
        shield_consecutive=shield_consecutive,
        evidence_schema_version=blessed_schema,
        shield_first_at=shield_first_at,
    )


def _provisional(
    state: GovernanceState, reason_codes: list[str], *, schema: int = 2
) -> StrategyState:
    return StrategyState(
        slug="defensive-etf-allocation",
        state=state,
        evaluated_at=datetime.combine(ASOF, datetime.min.time()),
        validation_age_days=0,
        reason_codes=list(reason_codes),
        reason="provisional",
        code_enabled_live=True,
        evidence_schema_version=schema,
    )


# --- models + constant -------------------------------------------------------


def test_evidence_schema_version_constant_is_int() -> None:
    assert isinstance(EVIDENCE_SCHEMA_VERSION, int)
    assert EVIDENCE_SCHEMA_VERSION == 1


def test_strategy_state_new_fields_default() -> None:
    state = StrategyState(
        slug="x",
        state=GovernanceState.LIVE,
        evaluated_at=datetime(2026, 6, 2),
        validation_age_days=0,
        reason_codes=[],
        reason="ok",
        code_enabled_live=True,
    )
    assert state.shielded is False
    assert state.shield_consecutive == 0
    assert state.evidence_schema_version == 1
    assert state.shield_first_at is None


def test_strategy_state_to_json_dict_serializes_shield_first_at() -> None:
    state = StrategyState(
        slug="x",
        state=GovernanceState.LIVE,
        evaluated_at=datetime(2026, 6, 2),
        validation_age_days=0,
        reason_codes=["schema_shield_retained_live"],
        reason="shielded",
        code_enabled_live=True,
        shielded=True,
        shield_consecutive=1,
        evidence_schema_version=1,
        shield_first_at=date(2026, 6, 2),
    )
    payload = state.to_json_dict()
    assert payload["shielded"] is True
    assert payload["shield_consecutive"] == 1
    assert payload["evidence_schema_version"] == 1
    assert payload["shield_first_at"] == "2026-06-02"


def test_strategy_state_to_json_dict_shield_first_at_none() -> None:
    state = StrategyState(
        slug="x",
        state=GovernanceState.LIVE,
        evaluated_at=datetime(2026, 6, 2),
        validation_age_days=0,
        reason_codes=[],
        reason="ok",
        code_enabled_live=True,
    )
    assert state.to_json_dict()["shield_first_at"] is None


# --- apply_schema_shield predicate ------------------------------------------


def test_shield_retains_live_on_bump_gate_failure() -> None:
    prov = _provisional(GovernanceState.QUARANTINED, ["failed_gate_deflated_sharpe"])
    out = apply_schema_shield(
        prov, evidence=_evidence(schema=2), asof=ASOF, prior_state=_prior_live(blessed_schema=1)
    )
    assert out.state is GovernanceState.LIVE
    assert out.reason_codes[0] == "schema_shield_retained_live"
    assert "failed_gate_deflated_sharpe" in out.reason_codes
    assert out.shielded is True
    assert out.shield_consecutive == 1
    assert out.evidence_schema_version == 1  # blessed version kept, not the failing v2
    assert out.shield_first_at == ASOF


def test_shield_carries_first_at_and_increments_consecutive() -> None:
    first = ASOF - timedelta(days=7)
    prior = _prior_live(
        blessed_schema=1, shielded=True, shield_first_at=first, shield_consecutive=1
    )
    out = apply_schema_shield(
        _provisional(GovernanceState.QUARANTINED, ["failed_gate_deflated_sharpe"]),
        evidence=_evidence(schema=2),
        asof=ASOF,
        prior_state=prior,
    )
    assert out.state is GovernanceState.LIVE
    assert out.shielded is True
    assert out.shield_first_at == first
    assert out.shield_consecutive == 2


def test_shield_calendar_wall_denies_after_max_days() -> None:
    first = ASOF - timedelta(days=MAX_SHIELD_CALENDAR_DAYS)
    prior = _prior_live(
        blessed_schema=1, shielded=True, shield_first_at=first, shield_consecutive=4
    )
    out = apply_schema_shield(
        _provisional(GovernanceState.QUARANTINED, ["failed_gate_deflated_sharpe"]),
        evidence=_evidence(schema=2),
        asof=ASOF,
        prior_state=prior,
    )
    assert out.state is GovernanceState.QUARANTINED
    assert out.reason_codes[0] == "shield_backstop_exhausted"
    assert out.shielded is False


def test_shield_fires_one_day_before_wall() -> None:
    first = ASOF - timedelta(days=MAX_SHIELD_CALENDAR_DAYS - 1)
    prior = _prior_live(blessed_schema=1, shielded=True, shield_first_at=first)
    out = apply_schema_shield(
        _provisional(GovernanceState.QUARANTINED, ["failed_gate_deflated_sharpe"]),
        evidence=_evidence(schema=2),
        asof=ASOF,
        prior_state=prior,
    )
    assert out.state is GovernanceState.LIVE
    assert out.shielded is True


def test_shield_inert_when_provisional_live() -> None:
    prov = _provisional(GovernanceState.LIVE, [])
    out = apply_schema_shield(
        prov, evidence=_evidence(schema=2), asof=ASOF, prior_state=_prior_live(blessed_schema=1)
    )
    assert out.state is GovernanceState.LIVE
    assert out.shielded is False
    assert out.reason_codes == []


def test_shield_does_not_fire_same_schema_gate_failure() -> None:
    out = apply_schema_shield(
        _provisional(GovernanceState.QUARANTINED, ["failed_gate_deflated_sharpe"], schema=1),
        evidence=_evidence(schema=1),
        asof=ASOF,
        prior_state=_prior_live(blessed_schema=1),
    )
    assert out.state is GovernanceState.QUARANTINED
    assert out.shielded is False
    assert "schema_shield_retained_live" not in out.reason_codes


def test_shield_does_not_fire_on_schema_downgrade() -> None:
    out = apply_schema_shield(
        _provisional(GovernanceState.QUARANTINED, ["failed_gate_deflated_sharpe"], schema=1),
        evidence=_evidence(schema=1),
        asof=ASOF,
        prior_state=_prior_live(blessed_schema=2),
    )
    assert out.state is GovernanceState.QUARANTINED


def test_shield_blocked_by_non_gate_reason() -> None:
    out = apply_schema_shield(
        _provisional(
            GovernanceState.QUARANTINED, ["failed_gate_deflated_sharpe", "stale_validation"]
        ),
        evidence=_evidence(schema=2),
        asof=ASOF,
        prior_state=_prior_live(blessed_schema=1),
    )
    assert out.state is GovernanceState.QUARANTINED
    assert out.shielded is False


def test_shield_never_promotes_quarantined_incumbent() -> None:
    prior = StrategyState(
        slug="defensive-etf-allocation",
        state=GovernanceState.QUARANTINED,
        evaluated_at=datetime(2026, 6, 1),
        validation_age_days=0,
        reason_codes=["failed_gate_deflated_sharpe"],
        reason="q",
        code_enabled_live=True,
        evidence_schema_version=1,
    )
    out = apply_schema_shield(
        _provisional(GovernanceState.QUARANTINED, ["failed_gate_deflated_sharpe"]),
        evidence=_evidence(schema=2),
        asof=ASOF,
        prior_state=prior,
    )
    assert out.state is GovernanceState.QUARANTINED


def test_shield_never_fires_without_prior() -> None:
    out = apply_schema_shield(
        _provisional(GovernanceState.QUARANTINED, ["failed_gate_deflated_sharpe"]),
        evidence=_evidence(schema=2),
        asof=ASOF,
        prior_state=None,
    )
    assert out.state is GovernanceState.QUARANTINED


def test_shield_inert_when_evidence_none() -> None:
    out = apply_schema_shield(
        _provisional(GovernanceState.QUARANTINED, ["missing_validation"]),
        evidence=None,
        asof=ASOF,
        prior_state=_prior_live(blessed_schema=1),
    )
    assert out.state is GovernanceState.QUARANTINED
    assert out.shielded is False
