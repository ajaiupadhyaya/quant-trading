"""Append-only research experiment registry."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

ExperimentKind = Literal["backtest", "validation", "research", "paper"]


@dataclass(frozen=True)
class ExperimentRecord:
    run_id: str
    created_at: datetime
    strategy: str
    kind: ExperimentKind
    git_sha: str
    command: str
    params: dict[str, object]
    metrics: dict[str, float]
    gates: dict[str, bool]
    artifacts: dict[str, str]
    data_snapshot_id: str | None
    wall_time_seconds: float

    def to_json_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["created_at"] = self.created_at.isoformat()
        return payload

    @classmethod
    def from_json_dict(cls, payload: dict[str, object]) -> ExperimentRecord:
        params = payload.get("params", {})
        metrics = payload.get("metrics", {})
        gates = payload.get("gates", {})
        artifacts = payload.get("artifacts", {})
        if not isinstance(params, dict):
            params = {}
        if not isinstance(metrics, dict):
            metrics = {}
        if not isinstance(gates, dict):
            gates = {}
        if not isinstance(artifacts, dict):
            artifacts = {}
        wall_time = payload["wall_time_seconds"]
        if not isinstance(wall_time, int | float | str):
            raise ValueError("wall_time_seconds must be numeric")
        return cls(
            run_id=str(payload["run_id"]),
            created_at=datetime.fromisoformat(str(payload["created_at"])),
            strategy=str(payload["strategy"]),
            kind=cast(ExperimentKind, payload["kind"]),
            git_sha=str(payload["git_sha"]),
            command=str(payload["command"]),
            params={str(k): v for k, v in params.items()},
            metrics={str(k): float(v) for k, v in metrics.items()},
            gates={str(k): bool(v) for k, v in gates.items()},
            artifacts={str(k): str(v) for k, v in artifacts.items()},
            data_snapshot_id=(
                None if payload.get("data_snapshot_id") is None else str(payload["data_snapshot_id"])
            ),
            wall_time_seconds=float(wall_time),
        )


def append_experiment(path: Path, record: ExperimentRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record.to_json_dict(), sort_keys=True, allow_nan=False) + "\n")


def list_experiments(path: Path) -> list[ExperimentRecord]:
    if not path.exists():
        return []
    rows: list[ExperimentRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, dict):
            raise ValueError(f"Malformed experiment row in {path}")
        rows.append(ExperimentRecord.from_json_dict(raw))
    return rows


def _find(path: Path, run_id: str) -> ExperimentRecord:
    for row in list_experiments(path):
        if row.run_id == run_id:
            return row
    raise KeyError(f"Unknown experiment run_id {run_id!r}")


def compare_experiments(path: Path, left_run_id: str, right_run_id: str) -> dict[str, object]:
    left = _find(path, left_run_id)
    right = _find(path, right_run_id)
    shared = sorted(set(left.metrics) & set(right.metrics))
    return {
        "left": left.run_id,
        "right": right.run_id,
        "metric_delta": {metric: right.metrics[metric] - left.metrics[metric] for metric in shared},
        "gate_changes": {
            gate: (left.gates.get(gate), right.gates.get(gate))
            for gate in sorted(set(left.gates) | set(right.gates))
            if left.gates.get(gate) != right.gates.get(gate)
        },
    }


def leaderboard(path: Path, *, metric: str) -> list[ExperimentRecord]:
    return sorted(
        [row for row in list_experiments(path) if metric in row.metrics],
        key=lambda row: row.metrics[metric],
        reverse=True,
    )
