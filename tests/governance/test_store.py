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
    load_allocation,
    load_strategy_states,
    load_validation_manifest,
    write_allocation,
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
        "evidence_schema_version": 1,
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
        evidence_schema_version=int(values["evidence_schema_version"]),  # type: ignore[call-overload]
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


def test_allocation_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "allocation.json"
    write_allocation(path, {"trend": 0.6, "defensive-etf-allocation": 0.4})
    loaded = load_allocation(path)
    assert loaded == {"defensive-etf-allocation": 0.4, "trend": 0.6}


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


# --- evidence-schema shield: new-field IO + atomicity ------------------------


def test_strategy_states_round_trips_shield_fields_nondefault(tmp_path: Path) -> None:
    path = tmp_path / "strategy_states.json"
    state = StrategyState(
        slug="defensive-etf-allocation",
        state=GovernanceState.LIVE,
        evaluated_at=datetime(2026, 6, 6),
        validation_age_days=0,
        reason_codes=["schema_shield_retained_live", "failed_gate_deflated_sharpe"],
        reason="shielded",
        code_enabled_live=True,
        shielded=True,
        shield_consecutive=2,
        evidence_schema_version=1,
        shield_first_at=date(2026, 6, 6),
    )
    write_strategy_states(path, {"defensive-etf-allocation": state})
    loaded = load_strategy_states(path)["defensive-etf-allocation"]
    assert loaded.shielded is True
    assert loaded.shield_consecutive == 2
    assert loaded.evidence_schema_version == 1
    assert loaded.shield_first_at == date(2026, 6, 6)


def test_legacy_strategy_states_loads_shield_defaults(tmp_path: Path) -> None:
    """A pre-shield version:1 file (no new keys) loads with safe defaults."""
    path = tmp_path / "strategy_states.json"
    legacy = {
        "version": 1,
        "strategies": {
            "defensive-etf-allocation": {
                "slug": "defensive-etf-allocation",
                "state": "live",
                "evaluated_at": "2026-05-27T00:00:00",
                "validation_age_days": 0,
                "reason_codes": [],
                "reason": "Fresh validation evidence passes all required gates.",
                "code_enabled_live": True,
                "manual_block": False,
            }
        },
    }
    path.write_text(json.dumps(legacy))
    loaded = load_strategy_states(path)["defensive-etf-allocation"]
    assert loaded.state is GovernanceState.LIVE
    assert loaded.shielded is False
    assert loaded.shield_consecutive == 0
    assert loaded.evidence_schema_version == 1
    assert loaded.shield_first_at is None


def test_strategy_states_reject_bool_as_shield_consecutive(tmp_path: Path) -> None:
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
    payload["strategies"]["trend"]["shield_consecutive"] = True  # bool, not int
    path.write_text(json.dumps(payload))
    with pytest.raises(GovernanceError, match="Malformed governance artifact"):
        load_strategy_states(path)


def test_strategy_states_reject_malformed_shield_first_at(tmp_path: Path) -> None:
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
    payload["strategies"]["trend"]["shield_first_at"] = "not-a-date"
    path.write_text(json.dumps(payload))
    with pytest.raises(GovernanceError, match="Malformed governance artifact"):
        load_strategy_states(path)


def test_validation_manifest_round_trips_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "validation_manifest.json"
    write_validation_manifest(path, {"trend": _evidence(evidence_schema_version=2)})
    assert load_validation_manifest(path)["trend"].evidence_schema_version == 2


def test_legacy_manifest_loads_schema_version_default(tmp_path: Path) -> None:
    path = tmp_path / "validation_manifest.json"
    write_validation_manifest(path, {"trend": _evidence()})
    payload = json.loads(path.read_text())
    del payload["strategies"]["trend"]["evidence_schema_version"]
    path.write_text(json.dumps(payload))
    assert load_validation_manifest(path)["trend"].evidence_schema_version == 1


def test_manifest_rejects_bool_as_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "validation_manifest.json"
    write_validation_manifest(path, {"trend": _evidence()})
    payload = json.loads(path.read_text())
    payload["strategies"]["trend"]["evidence_schema_version"] = True
    path.write_text(json.dumps(payload))
    with pytest.raises(GovernanceError, match="Malformed governance artifact"):
        load_validation_manifest(path)


def test_strategy_states_write_is_atomic_on_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the rename fails mid-write, the prior file is left intact (not truncated)."""
    import quant.util.atomic as atomic_mod

    path = tmp_path / "strategy_states.json"
    good = StrategyState(
        slug="trend",
        state=GovernanceState.LIVE,
        evaluated_at=datetime(2026, 5, 26),
        validation_age_days=6,
        reason_codes=[],
        reason="ok",
        code_enabled_live=True,
    )
    write_strategy_states(path, {"trend": good})
    original = path.read_text()

    def boom(src: object, dst: object) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(atomic_mod.os, "replace", boom)
    with pytest.raises(OSError):
        write_strategy_states(path, {"trend": good})
    # Prior file is byte-identical; the failed rename never truncated it.
    assert path.read_text() == original
