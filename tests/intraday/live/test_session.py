from datetime import UTC, datetime

from quant.intraday.live.session import session_state


def test_weekend_is_closed():
    # 2026-06-06 is a Saturday
    st = session_state(datetime(2026, 6, 6, 15, 0, tzinfo=UTC))
    assert st.open is False


def test_weekday_midsession_is_open():
    # 2026-06-08 Monday, 15:00 UTC == 11:00 ET (RTH)
    st = session_state(datetime(2026, 6, 8, 15, 0, tzinfo=UTC))
    assert st.open is True
    assert st.close.tzinfo is not None
