"""Tests for generating governance artifacts from validation sidecars."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from quant.governance.models import GovernanceError, GovernancePolicy, GovernanceState
from quant.governance.refresh import build_governance_artifacts, validation_report_to_evidence
from quant.governance.store import strategy_states_path
from quant.strategies import REGISTRY
from quant.strategies.base import StrategySpec


def _spec_cls(slug: str = "trend") -> type:
    return type(
        "S",
        (),
        {
            "spec": StrategySpec(
                slug=slug,
                name=slug,
                description="",
                universe=["SPY"],
                rebalance_frequency="monthly",
                enabled_live=True,
            )
        },
    )


def _write_sidecar(
    data_dir: Path,
    slug: str,
    *,
    gate_deflated_sharpe: bool = True,
    schema: int = 1,
) -> None:
    """Write artifacts + a sidecar at a given gate/schema (keeps artifact files present)."""
    out = data_dir / "backtests" / slug
    out.mkdir(parents=True, exist_ok=True)
    (out / "chosen_params.json").write_text(json.dumps({"latest": {"x": 1}, "windows": []}))
    (out / "walkforward.parquet").write_text("fake parquet")
    (out / "validation_report.json").write_text(
        json.dumps(
            {
                "slug": slug,
                "run_date": "2026-06-06",
                "data_start": "2010-01-01",
                "data_end": "2026-06-05",
                "gate_deflated_sharpe": gate_deflated_sharpe,
                "gate_probabilistic_sharpe": True,
                "gate_bootstrap_lower": True,
                "gate_regime": True,
                "gate_holdout": True,
                "deflated_sharpe": 0.54 if gate_deflated_sharpe else 0.246,
                "probabilistic_sharpe": 0.99,
                "bootstrap_total_return_p05": 0.12,
                "n_positive_regimes": 3,
                "n_tested_regimes": 3,
                "holdout_total_return": 0.21,
                "provenance": f"uv run quant validate {slug}",
                "evidence_schema_version": schema,
            }
        )
    )


def _write_validation_artifacts(data_dir: Path, slug: str, *, gate_regime: bool = True) -> None:
    out = data_dir / "backtests" / slug
    out.mkdir(parents=True, exist_ok=True)
    (out / "chosen_params.json").write_text(json.dumps({"latest": {"x": 1}, "windows": []}))
    (out / "walkforward.parquet").write_text("fake parquet")
    (out / "validation_report.json").write_text(
        json.dumps(
            {
                "slug": slug,
                "run_date": "2026-05-20",
                "data_start": "2010-01-01",
                "data_end": "2026-05-19",
                "gate_deflated_sharpe": True,
                "gate_probabilistic_sharpe": True,
                "gate_bootstrap_lower": True,
                "gate_regime": gate_regime,
                "gate_holdout": True,
                "deflated_sharpe": 0.54,
                "probabilistic_sharpe": 0.99,
                "bootstrap_total_return_p05": 0.12,
                "n_positive_regimes": 3,
                "n_tested_regimes": 3,
                "holdout_total_return": 0.21,
                "provenance": "uv run quant validate trend",
            }
        )
    )


def test_validation_report_to_evidence_adds_artifact_paths(tmp_data_dir: Path) -> None:
    _write_validation_artifacts(tmp_data_dir, "trend")
    evidence = validation_report_to_evidence(tmp_data_dir, "trend")
    assert evidence is not None
    assert evidence.slug == "trend"
    assert evidence.chosen_params_path.endswith("chosen_params.json")
    assert evidence.walkforward_path.endswith("walkforward.parquet")


def test_build_governance_artifacts_classifies_registry(tmp_data_dir: Path) -> None:
    _write_validation_artifacts(tmp_data_dir, "trend")
    states = build_governance_artifacts(
        data_dir=tmp_data_dir,
        registry={
            "trend": type(
                "S",
                (),
                {
                    "spec": StrategySpec(
                        slug="trend",
                        name="Trend",
                        description="",
                        universe=["SPY"],
                        rebalance_frequency="monthly",
                        enabled_live=True,
                    )
                },
            )
        },
        policy=GovernancePolicy(max_validation_age_days=30),
        asof=date(2026, 5, 26),
    )
    assert states["trend"].state is GovernanceState.LIVE
    assert (tmp_data_dir / "governance" / "validation_manifest.json").exists()
    assert (tmp_data_dir / "governance" / "strategy_states.json").exists()
    allocation = json.loads((tmp_data_dir / "governance" / "allocation.json").read_text())
    assert allocation["allocations"] == {"trend": 1.0}


def test_failed_validation_report_quarantines(tmp_data_dir: Path) -> None:
    _write_validation_artifacts(tmp_data_dir, "trend", gate_regime=False)
    states = build_governance_artifacts(
        data_dir=tmp_data_dir,
        registry={"trend": type("S", (), {"spec": REGISTRY["trend"].spec})},
        policy=GovernancePolicy(max_validation_age_days=30),
        asof=date(2026, 5, 26),
    )
    assert states["trend"].state is GovernanceState.QUARANTINED
    assert "failed_gate_regime" in states["trend"].reason_codes


def test_validation_report_rejects_non_finite_required_numeric_field(
    tmp_data_dir: Path,
) -> None:
    _write_validation_artifacts(tmp_data_dir, "trend")
    path = tmp_data_dir / "backtests" / "trend" / "validation_report.json"
    payload = path.read_text().replace('"deflated_sharpe": 0.54', '"deflated_sharpe": NaN')
    path.write_text(payload)

    with pytest.raises(GovernanceError, match="Malformed validation report"):
        validation_report_to_evidence(tmp_data_dir, "trend")


def test_validation_report_rejects_non_finite_nullable_numeric_field(
    tmp_data_dir: Path,
) -> None:
    _write_validation_artifacts(tmp_data_dir, "trend")
    path = tmp_data_dir / "backtests" / "trend" / "validation_report.json"
    payload = path.read_text().replace(
        '"bootstrap_total_return_p05": 0.12',
        '"bootstrap_total_return_p05": Infinity',
    )
    path.write_text(payload)

    with pytest.raises(GovernanceError, match="Malformed validation report"):
        validation_report_to_evidence(tmp_data_dir, "trend")


# --- evidence-schema shield (e2e through build_governance_artifacts) ---------


def _build(data_dir: Path, slug: str = "trend", asof: date = date(2026, 6, 6)) -> dict:
    return build_governance_artifacts(
        data_dir=data_dir,
        registry={slug: _spec_cls(slug)},
        policy=GovernancePolicy(max_validation_age_days=30),
        asof=asof,
    )


def test_validation_report_reads_schema_version(tmp_data_dir: Path) -> None:
    _write_sidecar(tmp_data_dir, "trend", schema=2)
    evidence = validation_report_to_evidence(tmp_data_dir, "trend")
    assert evidence is not None
    assert evidence.evidence_schema_version == 2


def test_validation_report_schema_version_defaults_when_absent(tmp_data_dir: Path) -> None:
    _write_validation_artifacts(tmp_data_dir, "trend")  # legacy sidecar, no schema key
    evidence = validation_report_to_evidence(tmp_data_dir, "trend")
    assert evidence is not None
    assert evidence.evidence_schema_version == 1


def test_build_no_shield_when_gates_pass(tmp_data_dir: Path) -> None:
    _write_sidecar(tmp_data_dir, "trend", gate_deflated_sharpe=True, schema=1)
    first = _build(tmp_data_dir)
    second = _build(tmp_data_dir)  # idempotent
    for states in (first, second):
        assert states["trend"].state is GovernanceState.LIVE
        assert states["trend"].shielded is False
        assert states["trend"].reason_codes == []
        assert states["trend"].evidence_schema_version == 1


def test_build_shield_fires_on_bump_gate_failure(tmp_data_dir: Path) -> None:
    # Refresh 1: passing sidecar @schema1 -> incumbent persisted LIVE@1.
    _write_sidecar(tmp_data_dir, "trend", gate_deflated_sharpe=True, schema=1)
    assert _build(tmp_data_dir)["trend"].state is GovernanceState.LIVE
    # Refresh 2: corrected math lands -> failing sidecar @schema2 (the bump).
    _write_sidecar(tmp_data_dir, "trend", gate_deflated_sharpe=False, schema=2)
    states = _build(tmp_data_dir)
    s = states["trend"]
    assert s.state is GovernanceState.LIVE  # NOT silently quarantined
    assert s.shielded is True
    assert s.reason_codes[0] == "schema_shield_retained_live"
    assert "failed_gate_deflated_sharpe" in s.reason_codes
    assert s.evidence_schema_version == 1  # blessed version retained
    assert s.shield_first_at == date(2026, 6, 6)
    # Persisted to disk.
    persisted = json.loads(strategy_states_path(tmp_data_dir).read_text())
    assert persisted["strategies"]["trend"]["shielded"] is True


def test_build_absent_prior_quarantines_failing_bump(tmp_data_dir: Path) -> None:
    # No prior strategy_states.json + a failing @schema2 sidecar -> no incumbent.
    _write_sidecar(tmp_data_dir, "trend", gate_deflated_sharpe=False, schema=2)
    states = _build(tmp_data_dir)
    assert states["trend"].state is GovernanceState.QUARANTINED
    assert states["trend"].shielded is False


def test_build_fails_loud_on_malformed_prior_states(tmp_data_dir: Path) -> None:
    _write_sidecar(tmp_data_dir, "trend", gate_deflated_sharpe=False, schema=2)
    path = strategy_states_path(tmp_data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not-json")  # present-but-malformed
    with pytest.raises(GovernanceError):
        _build(tmp_data_dir)
