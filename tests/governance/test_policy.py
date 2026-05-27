"""Tests for evidence-gated strategy governance policy."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from quant.governance.models import GovernancePolicy, GovernanceState, ValidationEvidence
from quant.governance.policy import classify_strategy
from quant.strategies.base import StrategySpec


def _spec(*, enabled_live: bool = True) -> StrategySpec:
    return StrategySpec(
        slug="trend",
        name="Trend",
        description="",
        universe=["SPY"],
        rebalance_frequency="monthly",
        enabled_live=enabled_live,
    )


def _evidence(**overrides: object) -> ValidationEvidence:
    values: dict[str, object] = {
        "slug": "trend",
        "run_date": date(2026, 5, 20),
        "data_start": date(2010, 1, 1),
        "data_end": date(2026, 5, 19),
        "gate_deflated_sharpe": True,
        "gate_probabilistic_sharpe": True,
        "gate_bootstrap_lower": True,
        "gate_regime": True,
        "gate_holdout": True,
        "deflated_sharpe": 0.54,
        "probabilistic_sharpe": 0.99,
        "bootstrap_total_return_p05": 0.12,
        "n_positive_regimes": 3,
        "n_tested_regimes": 3,
        "holdout_total_return": 0.21,
        "chosen_params_path": "data/backtests/trend/chosen_params.json",
        "walkforward_path": "data/backtests/trend/walkforward.parquet",
        "provenance": "unit test",
    }
    values.update(overrides)
    return ValidationEvidence(**values)


def test_live_when_enabled_fresh_all_gates_pass_and_artifacts_exist(tmp_path: Path) -> None:
    chosen = tmp_path / "chosen_params.json"
    walkforward = tmp_path / "walkforward.parquet"
    chosen.write_text("{}")
    walkforward.write_text("fake parquet")
    evidence = _evidence(
        chosen_params_path=str(chosen),
        walkforward_path=str(walkforward),
    )
    state = classify_strategy(
        spec=_spec(),
        evidence=evidence,
        policy=GovernancePolicy(max_validation_age_days=30),
        asof=date(2026, 5, 26),
    )
    assert state.state is GovernanceState.LIVE
    assert state.reason_codes == []


def test_missing_evidence_quarantines_live_capable_strategy() -> None:
    state = classify_strategy(
        spec=_spec(),
        evidence=None,
        policy=GovernancePolicy(max_validation_age_days=30),
        asof=date(2026, 5, 26),
    )
    assert state.state is GovernanceState.QUARANTINED
    assert "missing_validation" in state.reason_codes


def test_mismatched_evidence_slug_quarantines_live_capable_strategy(tmp_path: Path) -> None:
    chosen = tmp_path / "chosen_params.json"
    walkforward = tmp_path / "walkforward.parquet"
    chosen.write_text("{}")
    walkforward.write_text("fake parquet")
    evidence = _evidence(
        slug="risk-parity",
        chosen_params_path=str(chosen),
        walkforward_path=str(walkforward),
    )
    state = classify_strategy(
        spec=_spec(),
        evidence=evidence,
        policy=GovernancePolicy(max_validation_age_days=30),
        asof=date(2026, 5, 26),
    )
    assert state.state is GovernanceState.QUARANTINED
    assert "evidence_slug_mismatch" in state.reason_codes


def test_disabled_strategy_is_research_even_without_evidence() -> None:
    state = classify_strategy(
        spec=_spec(enabled_live=False),
        evidence=None,
        policy=GovernancePolicy(max_validation_age_days=30),
        asof=date(2026, 5, 26),
    )
    assert state.state is GovernanceState.RESEARCH
    assert "not_live_capable" in state.reason_codes


def test_stale_evidence_quarantines_strategy(tmp_path: Path) -> None:
    chosen = tmp_path / "chosen_params.json"
    walkforward = tmp_path / "walkforward.parquet"
    chosen.write_text("{}")
    walkforward.write_text("fake parquet")
    evidence = _evidence(
        run_date=date(2026, 4, 1),
        chosen_params_path=str(chosen),
        walkforward_path=str(walkforward),
    )
    state = classify_strategy(
        spec=_spec(),
        evidence=evidence,
        policy=GovernancePolicy(max_validation_age_days=30),
        asof=date(2026, 5, 26),
    )
    assert state.state is GovernanceState.QUARANTINED
    assert "stale_validation" in state.reason_codes


def test_failed_gate_quarantines_strategy(tmp_path: Path) -> None:
    chosen = tmp_path / "chosen_params.json"
    walkforward = tmp_path / "walkforward.parquet"
    chosen.write_text("{}")
    walkforward.write_text("fake parquet")
    evidence = _evidence(
        gate_regime=False,
        chosen_params_path=str(chosen),
        walkforward_path=str(walkforward),
    )
    state = classify_strategy(
        spec=_spec(),
        evidence=evidence,
        policy=GovernancePolicy(max_validation_age_days=30),
        asof=date(2026, 5, 26),
    )
    assert state.state is GovernanceState.QUARANTINED
    assert "failed_gate_regime" in state.reason_codes


def test_manual_block_quarantines_otherwise_passing_strategy(tmp_path: Path) -> None:
    chosen = tmp_path / "chosen_params.json"
    walkforward = tmp_path / "walkforward.parquet"
    chosen.write_text("{}")
    walkforward.write_text("fake parquet")
    evidence = _evidence(
        chosen_params_path=str(chosen),
        walkforward_path=str(walkforward),
        manual_block=True,
        manual_block_reason="paper drawdown review",
    )
    state = classify_strategy(
        spec=_spec(),
        evidence=evidence,
        policy=GovernancePolicy(max_validation_age_days=30),
        asof=date(2026, 5, 26),
    )
    assert state.state is GovernanceState.QUARANTINED
    assert "manual_block" in state.reason_codes
    assert "paper drawdown review" in state.reason


def test_missing_chosen_params_quarantines_strategy(tmp_path: Path) -> None:
    walkforward = tmp_path / "walkforward.parquet"
    walkforward.write_text("fake parquet")
    evidence = _evidence(
        chosen_params_path=str(tmp_path / "missing.json"),
        walkforward_path=str(walkforward),
    )
    state = classify_strategy(
        spec=_spec(),
        evidence=evidence,
        policy=GovernancePolicy(max_validation_age_days=30),
        asof=date(2026, 5, 26),
    )
    assert state.state is GovernanceState.QUARANTINED
    assert "missing_chosen_params" in state.reason_codes
