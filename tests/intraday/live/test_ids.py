from datetime import UTC, datetime

from quant.intraday.live.ids import make_sleeve_coid


def test_coid_is_namespaced_and_unique_per_seq():
    ts = datetime(2026, 6, 8, 14, 30, 0, tzinfo=UTC)
    a = make_sleeve_coid("QQQ", ts, 0)
    b = make_sleeve_coid("QQQ", ts, 1)
    assert a.startswith("sleeve:QQQ:")
    assert a != b


def test_coid_differs_by_symbol_and_time():
    ts1 = datetime(2026, 6, 8, 14, 30, 0, tzinfo=UTC)
    ts2 = datetime(2026, 6, 8, 14, 31, 0, tzinfo=UTC)
    assert make_sleeve_coid("QQQ", ts1, 0) != make_sleeve_coid("IWM", ts1, 0)
    assert make_sleeve_coid("QQQ", ts1, 0) != make_sleeve_coid("QQQ", ts2, 0)
