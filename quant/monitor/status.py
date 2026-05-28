"""On-disk status artifact for the monitoring daemon: data/ops/monitor_status.json."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from quant.monitor.guardrails import GuardrailOutcome, Severity


@dataclass(frozen=True)
class MonitorStatus:
    version: int
    at: datetime
    worst_severity: Severity
    halt_triggered_this_tick: bool
    halt_active: bool
    outcomes: list[GuardrailOutcome]
    heartbeat: str


def monitor_status_path(data_dir: Path) -> Path:
    return data_dir / "ops" / "monitor_status.json"


def write_status(data_dir: Path, status: MonitorStatus) -> Path:
    path = monitor_status_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": status.version,
        "at": status.at.isoformat(),
        "worst_severity": status.worst_severity,
        "halt_triggered_this_tick": status.halt_triggered_this_tick,
        "halt_active": status.halt_active,
        "outcomes": [
            {"name": o.name, "severity": o.severity, "detail": o.detail} for o in status.outcomes
        ],
        "heartbeat": status.heartbeat,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def read_status(data_dir: Path) -> MonitorStatus | None:
    path = monitor_status_path(data_dir)
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    outcomes = [
        GuardrailOutcome(name=str(o["name"]), severity=o["severity"], detail=str(o["detail"]))
        for o in raw.get("outcomes", [])
    ]
    return MonitorStatus(
        version=int(raw["version"]),
        at=datetime.fromisoformat(str(raw["at"])),
        worst_severity=raw["worst_severity"],
        halt_triggered_this_tick=bool(raw["halt_triggered_this_tick"]),
        halt_active=bool(raw["halt_active"]),
        outcomes=outcomes,
        heartbeat=str(raw["heartbeat"]),
    )
