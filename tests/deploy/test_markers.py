"""Session-scoped idempotency markers under data/ops/scheduler/."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

from quant.deploy.markers import marker_path, read_markers, write_marker


def test_write_then_read_latest(tmp_path: Path) -> None:
    write_marker(
        tmp_path,
        "daily-rebalance",
        date(2026, 6, 1),
        kind="FRESH",
        fired_at_utc=datetime(2026, 6, 1, 19, 55, tzinfo=UTC),
        exit_code=0,
        duration_s=4.2,
    )
    write_marker(
        tmp_path,
        "daily-rebalance",
        date(2026, 6, 2),
        kind="FRESH",
        fired_at_utc=datetime(2026, 6, 2, 19, 55, tzinfo=UTC),
        exit_code=0,
        duration_s=4.0,
    )
    assert read_markers(tmp_path)["daily-rebalance"] == date(2026, 6, 2)


def test_read_markers_empty_when_no_dir(tmp_path: Path) -> None:
    assert read_markers(tmp_path) == {}


def test_marker_write_is_atomic_no_tmp(tmp_path: Path) -> None:
    write_marker(
        tmp_path,
        "premarket-health",
        date(2026, 6, 2),
        kind="FRESH",
        fired_at_utc=datetime(2026, 6, 2, 13, 0, tzinfo=UTC),
        exit_code=0,
        duration_s=1.0,
    )
    d = tmp_path / "ops" / "scheduler"
    assert all(p.suffix == ".json" for p in d.iterdir()), list(d.iterdir())


def test_marker_path_encodes_job_and_date(tmp_path: Path) -> None:
    p = marker_path(tmp_path, "daily-rebalance", date(2026, 6, 2))
    assert p.name == "daily-rebalance.2026-06-02.json"
