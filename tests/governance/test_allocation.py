"""Tests for governance capital allocation."""

from __future__ import annotations

from datetime import datetime

from quant.governance.allocation import AllocationConfig, allocate_capital
from quant.governance.models import GovernanceState, StrategyState, ValidationEvidence


def _state(slug: str, state: GovernanceState) -> StrategyState:
    return StrategyState(
        slug=slug,
        state=state,
        evaluated_at=datetime(2026, 5, 27),
        validation_age_days=0,
        reason_codes=[] if state is GovernanceState.LIVE else ["failed_gate_bootstrap_lower"],
        reason="",
        code_enabled_live=True,
    )


def _evidence(slug: str, dsr: float) -> ValidationEvidence:
    from datetime import date

    return ValidationEvidence(
        slug=slug,
        run_date=date(2026, 5, 27),
        data_start=date(2010, 1, 1),
        data_end=date(2026, 5, 26),
        gate_deflated_sharpe=True,
        gate_probabilistic_sharpe=True,
        gate_bootstrap_lower=True,
        gate_regime=True,
        gate_holdout=True,
        deflated_sharpe=dsr,
        probabilistic_sharpe=0.9,
        bootstrap_total_return_p05=0.02,
        n_positive_regimes=4,
        n_tested_regimes=4,
        holdout_total_return=0.1,
        chosen_params_path="chosen.json",
        walkforward_path="wf.parquet",
        provenance="test",
    )


def test_allocation_never_assigns_to_quarantined_strategy() -> None:
    weights = allocate_capital(
        {
            "baseline": _state("baseline", GovernanceState.LIVE),
            "trend": _state("trend", GovernanceState.QUARANTINED),
        },
        evidence_by_slug={"baseline": _evidence("baseline", 0.5), "trend": _evidence("trend", 2.0)},
        config=AllocationConfig(mode="dsr-weighted"),
    )

    assert weights == {"baseline": 1.0}


def test_equal_live_uses_cap_and_renormalizes() -> None:
    weights = allocate_capital(
        {
            "a": _state("a", GovernanceState.LIVE),
            "b": _state("b", GovernanceState.LIVE),
            "c": _state("c", GovernanceState.LIVE),
        },
        evidence_by_slug={},
        config=AllocationConfig(mode="equal-live", max_weight=0.40),
    )

    assert weights == {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}


def test_dsr_weighted_prefers_stronger_evidence() -> None:
    weights = allocate_capital(
        {
            "a": _state("a", GovernanceState.LIVE),
            "b": _state("b", GovernanceState.LIVE),
        },
        evidence_by_slug={"a": _evidence("a", 0.4), "b": _evidence("b", 0.8)},
        config=AllocationConfig(mode="dsr-weighted", max_weight=0.80),
    )

    assert weights["b"] > weights["a"]
    assert abs(sum(weights.values()) - 1.0) < 1e-12


def test_dsr_weighted_respects_minimum_for_live_strategies_when_feasible() -> None:
    weights = allocate_capital(
        {
            "a": _state("a", GovernanceState.LIVE),
            "b": _state("b", GovernanceState.LIVE),
            "c": _state("c", GovernanceState.LIVE),
        },
        evidence_by_slug={
            "a": _evidence("a", 10.0),
            "b": _evidence("b", 0.01),
            "c": _evidence("c", 0.01),
        },
        config=AllocationConfig(mode="dsr-weighted", max_weight=0.90, min_weight=0.05),
    )

    assert weights["b"] >= 0.05
    assert weights["c"] >= 0.05
    assert abs(sum(weights.values()) - 1.0) < 1e-12
