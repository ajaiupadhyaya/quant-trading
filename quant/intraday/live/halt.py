"""Sleeve-scoped halt artifact. Distinct from quant.governance.halt (global). Stops
ONLY the intraday loop; the daily system is unaffected. Fail-closed on corruption."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class SleeveHaltState:
    active: bool
    reason: str
    updated_at: datetime


def sleeve_halt_path(data_dir: Path) -> Path:
    return data_dir / "intraday" / "live" / "sleeve_halt.json"


def load_sleeve_halt(data_dir: Path) -> SleeveHaltState:
    path = sleeve_halt_path(data_dir)
    if not path.exists():
        return SleeveHaltState(False, "not halted", datetime.fromtimestamp(0, UTC))
    try:
        obj = json.loads(path.read_text())
        if not isinstance(obj, dict):
            raise ValueError("sleeve halt artifact is not a JSON object")
        return SleeveHaltState(
            active=bool(obj["active"]),
            reason=str(obj["reason"]),
            updated_at=datetime.fromisoformat(obj["updated_at"]),
        )
    except (ValueError, KeyError, OSError) as exc:
        # Fail closed: an unreadable halt artifact must read as HALTED.
        return SleeveHaltState(True, f"corrupt sleeve halt artifact: {exc}", datetime.now(UTC))


def _write(data_dir: Path, state: SleeveHaltState) -> None:
    path = sleeve_halt_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "active": state.active,
        "reason": state.reason,
        "updated_at": state.updated_at.isoformat(),
    }))


def set_sleeve_halt(
    data_dir: Path, *, reason: str, created_at: datetime | None = None
) -> SleeveHaltState:
    st = SleeveHaltState(True, reason, created_at or datetime.now(UTC))
    _write(data_dir, st)
    return st


def clear_sleeve_halt(data_dir: Path, *, reason: str) -> SleeveHaltState:
    st = SleeveHaltState(False, reason, datetime.now(UTC))
    _write(data_dir, st)
    return st
