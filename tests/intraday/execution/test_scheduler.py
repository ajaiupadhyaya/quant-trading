import pytest

from quant.intraday.execution.scheduler import ExecutionProgram
from quant.intraday.strategy import Side


def _prog():
    return ExecutionProgram(
        symbol="QQQ", side=Side.BUY, total_qty=100, child_sizes=[40, 30, 30], start_tick=5
    )


def test_slice_due_follows_schedule_by_tick_offset():
    p = _prog()
    assert p.slice_due(5) == 40
    assert p.slice_due(6) == 30
    assert p.slice_due(7) == 30


def test_slice_due_zero_before_start_and_after_end():
    p = _prog()
    assert p.slice_due(4) == 0
    assert p.slice_due(8) == 0


def test_record_fill_tracks_remaining_and_completion():
    p = _prog()
    assert not p.is_complete
    p.record_fill(40)
    p.record_fill(30)
    assert p.remaining == 30 and not p.is_complete
    p.record_fill(30)
    assert p.remaining == 0 and p.is_complete


def test_cancel_marks_complete_and_zeros_due():
    p = _prog()
    p.cancel()
    assert p.is_complete
    assert p.slice_due(5) == 0


def test_child_sizes_must_sum_to_total():
    with pytest.raises(ValueError):
        ExecutionProgram(
            symbol="QQQ", side=Side.BUY, total_qty=100, child_sizes=[40, 30], start_tick=0
        )
