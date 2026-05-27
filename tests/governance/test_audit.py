"""Tests for validation reproducibility audits."""

from __future__ import annotations

import json
from pathlib import Path

from quant.governance.audit import build_validation_audit, hash_file


def _write_report(data_dir: Path, slug: str) -> Path:
    out = data_dir / "backtests" / slug
    out.mkdir(parents=True, exist_ok=True)
    (out / "chosen_params.json").write_text('{"latest":{"lookback":20},"windows":[]}\n')
    (out / "walkforward.parquet").write_bytes(b"fake parquet bytes")
    report = {
        "slug": slug,
        "run_date": "2026-05-26",
        "data_start": "2010-01-01",
        "data_end": "2026-05-26",
        "gate_deflated_sharpe": True,
        "gate_probabilistic_sharpe": True,
        "gate_bootstrap_lower": False,
        "gate_regime": True,
        "gate_holdout": True,
        "deflated_sharpe": 0.51,
        "probabilistic_sharpe": 0.95,
        "bootstrap_total_return_p05": -0.04,
        "n_positive_regimes": 4,
        "n_tested_regimes": 4,
        "holdout_total_return": 0.21,
        "validation_command": (
            "quant validate trend --start 2010-01-01 --end 2026-05-26 "
            "--bootstrap-resamples 5000 --bootstrap-seed 11"
        ),
        "bootstrap_resamples": 5000,
        "bootstrap_seed": 11,
        "provenance": "quant validate trend --start 2010-01-01 --end 2026-05-26",
    }
    report_path = out / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report_path


def test_hash_file_is_content_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    path.write_text('{"a":1}\n')

    first = hash_file(path)
    second = hash_file(path)

    assert first == second
    assert len(first) == 64


def test_build_validation_audit_records_reproducibility_metadata(tmp_data_dir: Path) -> None:
    report_path = _write_report(tmp_data_dir, "trend")

    audit = build_validation_audit(tmp_data_dir, "trend", repo_dir=Path.cwd())

    assert audit.strategy_slug == "trend"
    assert audit.data_range == ("2010-01-01", "2026-05-26")
    assert audit.bootstrap_resamples == 5000
    assert audit.bootstrap_seed == 11
    assert audit.validation_command.endswith("--bootstrap-resamples 5000 --bootstrap-seed 11")
    assert audit.validation_report_hash == hash_file(report_path)
    assert audit.chosen_params_hash == hash_file(
        tmp_data_dir / "backtests" / "trend" / "chosen_params.json"
    )
    assert audit.walkforward_parquet_hash == hash_file(
        tmp_data_dir / "backtests" / "trend" / "walkforward.parquet"
    )
    assert audit.missing_artifacts == []
    assert "failed bootstrap lower-5% gate" in audit.explanation


def test_build_validation_audit_reports_missing_artifacts(tmp_data_dir: Path) -> None:
    audit = build_validation_audit(tmp_data_dir, "pairs", repo_dir=Path.cwd())

    assert audit.strategy_slug == "pairs"
    assert audit.validation_report_hash is None
    assert audit.chosen_params_hash is None
    assert audit.walkforward_parquet_hash is None
    assert audit.missing_artifacts == ["validation_report.json"]
    assert "missing validation_report.json" in audit.explanation
