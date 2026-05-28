from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from quant.monitor.guardrails import GuardrailOutcome
from quant.monitor.status import (
    MonitorStatus,
    monitor_status_path,
    read_status,
    write_status,
)


def test_status_path(tmp_path: Path) -> None:
    assert monitor_status_path(tmp_path) == tmp_path / "ops" / "monitor_status.json"


def test_write_then_read_roundtrip(tmp_path: Path) -> None:
    status = MonitorStatus(
        version=1,
        at=datetime(2026, 5, 28, 14, 32, 5, tzinfo=UTC),
        worst_severity="halt",
        halt_triggered_this_tick=True,
        halt_active=True,
        outcomes=[
            GuardrailOutcome("drift", "halt", "halt_candidate: account@20d z=-2.50"),
            GuardrailOutcome("account_drawdown", "ok", "drawdown -1.00% within -25.00%"),
        ],
        heartbeat="14:32:05 | equity $100,000 dd -1.0% | drift halt | ...",
    )
    write_status(tmp_path, status)
    loaded = read_status(tmp_path)
    assert loaded is not None
    assert loaded.version == 1
    assert loaded.at == status.at
    assert loaded.worst_severity == "halt"
    assert loaded.halt_triggered_this_tick is True
    assert loaded.halt_active is True
    assert [(o.name, o.severity, o.detail) for o in loaded.outcomes] == [
        (o.name, o.severity, o.detail) for o in status.outcomes
    ]
    assert loaded.heartbeat == status.heartbeat


def test_read_status_absent_returns_none(tmp_path: Path) -> None:
    assert read_status(tmp_path) is None
