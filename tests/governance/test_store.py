"""Tests for governance JSON artifact stores."""

from __future__ import annotations

import json
import math
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


def _evidence(slug: str = "trend", **overrides: object) -> ValidationEvidence:
    values: dict[str, object] = {
        "slug": slug,
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
    return ValidationEvidence(
        slug=str(values["slug"]),
        run_date=values["run_date"],  # type: ignore[arg-type]
        data_start=values["data_start"],  # type: ignore[arg-type]
        data_end=values["data_end"],  # type: ignore[arg-type]
        gate_deflated_sharpe=bool(values["gate_deflated_sharpe"]),
        gate_probabilistic_sharpe=bool(values["gate_probabilistic_sharpe"]),
        gate_bootstrap_lower=bool(values["gate_bootstrap_lower"]),
        gate_regime=bool(values["gate_regime"]),
        gate_holdout=bool(values["gate_holdout"]),
        deflated_sharpe=float(values["deflated_sharpe"]),
        probabilistic_sharpe=float(values["probabilistic_sharpe"]),
        bootstrap_total_return_p05=values["bootstrap_total_return_p05"],  # type: ignore[arg-type]
        n_positive_regimes=int(values["n_positive_regimes"]),
        n_tested_regimes=int(values["n_tested_regimes"]),
        holdout_total_return=values["holdout_total_return"],  # type: ignore[arg-type]
        chosen_params_path=str(values["chosen_params_path"]),
        walkforward_path=str(values["walkforward_path"]),
        provenance=str(values["provenance"]),
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


def test_manifest_rejects_string_bool_gate(tmp_path: Path) -> None:
    path = tmp_path / "validation_manifest.json"
    write_validation_manifest(path, {"trend": _evidence()})
    payload = json.loads(path.read_text())
    payload["strategies"]["trend"]["gate_deflated_sharpe"] = "false"
    path.write_text(json.dumps(payload))

    with pytest.raises(GovernanceError, match="Malformed governance artifact"):
        load_validation_manifest(path)


def test_write_validation_manifest_rejects_non_finite_metric(tmp_path: Path) -> None:
    path = tmp_path / "validation_manifest.json"
    evidence = _evidence(deflated_sharpe=math.nan)

    with pytest.raises(GovernanceError, match="non-finite"):
        write_validation_manifest(path, {"trend": evidence})


def test_strategy_states_reject_string_bool(tmp_path: Path) -> None:
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
    payload = json.loads(path.read_text())
    payload["strategies"]["trend"]["code_enabled_live"] = "true"
    path.write_text(json.dumps(payload))

    with pytest.raises(GovernanceError, match="Malformed governance artifact"):
        load_strategy_states(path)


def test_strategy_states_reject_non_list_reason_codes(tmp_path: Path) -> None:
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
    payload = json.loads(path.read_text())
    payload["strategies"]["trend"]["reason_codes"] = "ok"
    path.write_text(json.dumps(payload))

    with pytest.raises(GovernanceError, match="Malformed governance artifact"):
        load_strategy_states(path)
