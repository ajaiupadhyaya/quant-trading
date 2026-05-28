"""Tests for the append-only research experiment registry."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from quant.research.registry import (
    ExperimentRecord,
    append_experiment,
    compare_experiments,
    leaderboard,
    list_experiments,
)


def _record(run_id: str, *, strategy: str = "trend", dsr: float = 0.4) -> ExperimentRecord:
    return ExperimentRecord(
        run_id=run_id,
        created_at=datetime(2026, 5, 28, 14, 0, tzinfo=UTC),
        strategy=strategy,
        kind="validation",
        git_sha="abc123",
        command="quant validate trend",
        params={"lookback": 252},
        metrics={"dsr": dsr, "psr": 0.9},
        gates={"overall": dsr > 0.3},
        artifacts={"validation_report": "data/backtests/trend/validation_report.json"},
        data_snapshot_id="snap-1",
        wall_time_seconds=12.5,
    )


def test_experiment_registry_appends_jsonl_deterministically(tmp_path: Path) -> None:
    path = tmp_path / "experiments.jsonl"

    append_experiment(path, _record("run-a"))
    append_experiment(path, _record("run-b", strategy="momentum", dsr=0.8))

    rows = list_experiments(path)
    assert [row.run_id for row in rows] == ["run-a", "run-b"]
    assert rows[0].metrics["dsr"] == 0.4
    assert path.read_text().count("\n") == 2


def test_compare_and_leaderboard_rank_by_metric(tmp_path: Path) -> None:
    path = tmp_path / "experiments.jsonl"
    append_experiment(path, _record("run-a", dsr=0.4))
    append_experiment(path, _record("run-b", strategy="momentum", dsr=0.8))

    comparison = compare_experiments(path, "run-a", "run-b")
    assert comparison["metric_delta"]["dsr"] == 0.4

    rows = leaderboard(path, metric="dsr")
    assert [row.run_id for row in rows] == ["run-b", "run-a"]
