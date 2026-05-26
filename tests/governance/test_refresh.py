"""Tests for generating governance artifacts from validation sidecars."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from quant.governance.models import GovernancePolicy, GovernanceState
from quant.governance.refresh import build_governance_artifacts, validation_report_to_evidence
from quant.strategies import REGISTRY
from quant.strategies.base import StrategySpec


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
