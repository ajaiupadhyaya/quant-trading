"""Emergency halt/resume artifact for fail-closed paper operations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class HaltState:
    active: bool
    reason: str
    updated_at: datetime


def halt_path(data_dir: Path) -> Path:
    return data_dir / "governance" / "halt.json"


def load_halt(data_dir: Path) -> HaltState:
    path = halt_path(data_dir)
    if not path.exists():
        return HaltState(
            active=False, reason="not halted", updated_at=datetime.fromtimestamp(0, UTC)
        )
    # Fail closed: any unreadable/malformed halt artifact is treated as ACTIVE.
    # A corrupt halt.json must never read as "not halted" — that would let
    # trading proceed through a halt the artifact was meant to enforce.
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("halt artifact is not a JSON object")
        return HaltState(
            active=bool(raw.get("active", False)),
            reason=str(raw.get("reason", "")),
            updated_at=datetime.fromisoformat(str(raw["updated_at"])),
        )
    except (ValueError, KeyError, OSError) as exc:
        return HaltState(
            active=True, reason=f"corrupt halt artifact: {exc}", updated_at=datetime.now(UTC)
        )


def set_halt(data_dir: Path, *, reason: str, created_at: datetime | None = None) -> HaltState:
    state = HaltState(
        active=True,
        reason=reason,
        updated_at=created_at or datetime.now(UTC).replace(microsecond=0),
    )
    _write(data_dir, state)
    return state


def clear_halt(data_dir: Path, *, reason: str) -> HaltState:
    state = HaltState(
        active=False,
        reason=reason,
        updated_at=datetime.now(UTC).replace(microsecond=0),
    )
    _write(data_dir, state)
    return state


def _write(data_dir: Path, state: HaltState) -> None:
    from quant.util.atomic import write_json_atomic

    write_json_atomic(
        halt_path(data_dir),
        {
            "active": state.active,
            "reason": state.reason,
            "updated_at": state.updated_at.isoformat(),
        },
    )
