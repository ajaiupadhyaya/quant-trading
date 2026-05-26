"""Tests for governance JSON artifact stores."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import pytest

from quant.governance.models import (
    GovernanceError,
    GovernanceState,
    StrategyState,
    ValidationEvidence,
)
from quant.governance.store import (
    load_strategy_states,
    load_validation_manifest,
    write_strategy_states,
    write_validation_manifest,
)


def _evidence(slug: str = "trend") -> ValidationEvidence:
    return ValidationEvidence(
        slug=slug,
        run_date=date(2026, 5, 20),
        data_start=date(2010, 1, 1),
        data_end=date(2026, 5, 19),
        gate_deflated_sharpe=True,
        gate_probabilistic_sharpe=True,
        gate_bootstrap_lower=True,
        gate_regime=True,
        gate_holdout=True,
        deflated_sharpe=0.54,
        probabilistic_sharpe=0.99,
        bootstrap_total_return_p05=0.12,
        n_positive_regimes=3,
        n_tested_regimes=3,
        holdout_total_return=0.21,
        chosen_params_path="data/backtests/trend/chosen_params.json",
        walkforward_path="data/backtests/trend/walkforward.parquet",
        provenance="unit test",
    )


def test_validation_manifest_round_trips_deterministically(tmp_path: Path) -> None:
    path = tmp_path / "validation_manifest.json"
    write_validation_manifest(path, {"trend": _evidence()})
    first = path.read_text()
    write_validation_manifest(path, {"trend": _evidence()})
    second = path.read_text()
    loaded = load_validation_manifest(path)
    assert first == second
    assert loaded["trend"].run_date == date(2026, 5, 20)
    assert json.loads(first)["strategies"]["trend"]["slug"] == "trend"


def test_strategy_states_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "strategy_states.json"
    state = StrategyState(
        slug="trend",
        state=GovernanceState.LIVE,
        evaluated_at=datetime(2026, 5, 26),
        validation_age_days=6,
        reason_codes=[],
        reason="ok",
        code_enabled_live=True,
    )
    write_strategy_states(path, {"trend": state})
    loaded = load_strategy_states(path)
    assert loaded["trend"].state is GovernanceState.LIVE
    assert loaded["trend"].validation_age_days == 6


def test_missing_manifest_raises_governance_error(tmp_path: Path) -> None:
    with pytest.raises(GovernanceError, match="Missing governance artifact"):
        load_validation_manifest(tmp_path / "missing.json")


def test_malformed_manifest_raises_governance_error(tmp_path: Path) -> None:
    path = tmp_path / "validation_manifest.json"
    path.write_text("{not-json")
    with pytest.raises(GovernanceError, match="Malformed governance artifact"):
        load_validation_manifest(path)
