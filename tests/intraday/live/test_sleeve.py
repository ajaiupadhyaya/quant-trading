from quant.intraday.live.sleeve import Fill, SleeveLedger


def test_long_round_trip_realized_pnl():
    led = SleeveLedger()
    led.record(Fill(symbol="QQQ", qty=10, price=100.0))  # buy 10 @100
    assert led.position("QQQ") == 10
    led.record(Fill(symbol="QQQ", qty=-10, price=105.0))  # sell 10 @105
    assert led.position("QQQ") == 0
    assert led.realized_pnl == 50.0  # (105-100)*10


def test_short_round_trip_realized_pnl():
    led = SleeveLedger()
    led.record(Fill(symbol="IWM", qty=-5, price=200.0))  # short 5 @200
    led.record(Fill(symbol="IWM", qty=5, price=190.0))  # cover 5 @190
    assert led.position("IWM") == 0
    assert led.realized_pnl == 50.0  # (200-190)*5


def test_unrealized_and_sleeve_value_from_marks():
    led = SleeveLedger()
    led.record(Fill(symbol="DIA", qty=4, price=300.0))
    marks = {"DIA": 310.0}
    assert led.unrealized_pnl(marks) == 40.0
    assert led.gross_notional(marks) == 4 * 310.0


def test_day_pnl_is_realized_plus_unrealized():
    led = SleeveLedger()
    led.record(Fill(symbol="QQQ", qty=10, price=100.0))
    led.record(Fill(symbol="QQQ", qty=-4, price=110.0))  # realize (110-100)*4=40
    marks = {"QQQ": 120.0}  # 6 left, unreal (120-100)*6=120
    assert led.realized_pnl == 40.0
    assert led.unrealized_pnl(marks) == 120.0
    assert led.day_pnl(marks) == 160.0


def test_round_trips_counts_opens_only():
    led = SleeveLedger()
    led.record(Fill(symbol="QQQ", qty=10, price=100.0))  # open
    led.record(Fill(symbol="QQQ", qty=-10, price=101.0))  # close (not a new open)
    led.record(Fill(symbol="IWM", qty=-3, price=50.0))  # open short
    assert led.round_trips == 2


def test_flip_through_zero_long_to_short():
    led = SleeveLedger()
    led.record(Fill("QQQ", qty=10, price=100.0))  # long 10 @100
    led.record(Fill("QQQ", qty=-15, price=110.0))  # sell 15 -> close 10, open short 5
    assert led.position("QQQ") == -5
    assert led.realized_pnl == 100.0  # (110-100)*10, NOT *15
    assert led.round_trips == 2  # one open per leg
    marks = {"QQQ": 110.0}
    assert led.unrealized_pnl(marks) == 0.0  # new short opened at 110, marked 110


def test_missing_mark_does_not_crash():
    led = SleeveLedger()
    led.record(Fill("QQQ", qty=5, price=100.0))
    # marks omits QQQ -> falls back to avg cost, 0 unrealized, notional at avg
    assert led.unrealized_pnl({}) == 0.0
    assert led.gross_notional({}) == 5 * 100.0
    assert led.day_pnl({}) == 0.0
