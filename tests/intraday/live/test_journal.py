from datetime import UTC, datetime

from quant.intraday.live.journal import TickRecord, append_tick, read_ticks


def test_append_then_read_roundtrips(tmp_path):
    rec = TickRecord(
        ts=datetime(2026, 6, 8, 15, 0, tzinfo=UTC),
        sleeve_value=1234.5, day_pnl=12.3, round_trips=2,
        n_orders=1, halted=False, note="ok",
    )
    append_tick(tmp_path, rec)
    df = read_ticks(tmp_path)
    assert len(df) == 1
    assert df.iloc[0]["day_pnl"] == 12.3
    assert df.iloc[0]["n_orders"] == 1


def test_append_is_cumulative(tmp_path):
    for i in range(3):
        append_tick(tmp_path, TickRecord(
            ts=datetime(2026, 6, 8, 15, i, tzinfo=UTC),
            sleeve_value=0.0, day_pnl=float(i), round_trips=0,
            n_orders=0, halted=False, note=""))
    assert len(read_ticks(tmp_path)) == 3


def test_empty_and_populated_share_dtypes(tmp_path):
    from quant.intraday.live.journal import TickRecord, append_tick, read_ticks

    empty = read_ticks(tmp_path)              # absent file
    append_tick(tmp_path, TickRecord(
        ts=datetime(2026, 6, 8, 15, 0, tzinfo=UTC), sleeve_value=1.0, day_pnl=2.0,
        round_trips=1, n_orders=1, halted=True, note="x"))
    populated = read_ticks(tmp_path)
    assert dict(empty.dtypes) == dict(populated.dtypes)
    assert populated["halted"].dtype == bool
    assert populated["ts"].dt.tz is not None
