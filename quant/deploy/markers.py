"""Once-per-session idempotency markers + per-day run-ledger.

A marker file per (job, session-date) is written ONLY after a job's command
chain completes (the timing-critical pre-submit marker is the one exception,
written by quant rebalance itself before order submission). The scheduler keys
on session-date — not "today" — so a missed prior session is representable.
Markers are host-local run state (git-ignored), not committed artifacts.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import date, datetime
from pathlib import Path

from quant.util.atomic import write_json_atomic


def _scheduler_dir(data_dir: Path) -> Path:
    return data_dir / "ops" / "scheduler"


def marker_path(data_dir: Path, job_name: str, session_date: date) -> Path:
    return _scheduler_dir(data_dir) / f"{job_name}.{session_date.isoformat()}.json"


def write_marker(
    data_dir: Path,
    job_name: str,
    session_date: date,
    *,
    kind: str,
    fired_at_utc: datetime,
    exit_code: int,
    duration_s: float,
) -> Path:
    path = marker_path(data_dir, job_name, session_date)
    write_json_atomic(
        path,
        {
            "job": job_name,
            "session_date": session_date.isoformat(),
            "kind": kind,
            "fired_at_utc": fired_at_utc.isoformat(),
            "exit_code": exit_code,
            "duration_s": duration_s,
        },
    )
    return path


def read_markers(data_dir: Path) -> dict[str, date]:
    """Return {job_name: latest session-date with a marker}."""
    d = _scheduler_dir(data_dir)
    if not d.exists():
        return {}
    latest: dict[str, date] = {}
    for f in d.glob("*.json"):
        try:
            raw = json.loads(f.read_text(encoding="utf-8"))
            job = str(raw["job"])
            sd = date.fromisoformat(str(raw["session_date"]))
        except (ValueError, KeyError, OSError):
            continue
        if job not in latest or sd > latest[job]:
            latest[job] = sd
    return latest


def is_marked(markers: Mapping[str, date], job_name: str, session_date: date) -> bool:
    return markers.get(job_name) == session_date
