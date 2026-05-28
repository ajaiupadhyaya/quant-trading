"""Tests for emergency governance halt/resume controls."""

from __future__ import annotations

from datetime import UTC, datetime

from quant.governance.halt import clear_halt, load_halt, set_halt


def test_halt_state_round_trips_and_can_resume(tmp_data_dir) -> None:
    set_halt(
        tmp_data_dir,
        reason="operator stop",
        created_at=datetime(2026, 5, 28, 15, 0, tzinfo=UTC),
    )
    state = load_halt(tmp_data_dir)
    assert state.active
    assert state.reason == "operator stop"

    clear_halt(tmp_data_dir, reason="verified healthy")
    resumed = load_halt(tmp_data_dir)
    assert not resumed.active
    assert resumed.reason == "verified healthy"
