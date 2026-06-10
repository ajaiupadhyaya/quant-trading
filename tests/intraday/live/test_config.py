import pytest

from quant.intraday.live.config import SleeveConfig


def test_defaults_are_tight_and_safe():
    c = SleeveConfig()
    assert c.universe == ("QQQ", "IWM", "DIA")
    assert c.notional_cap_pct == 0.10
    assert c.notional_cap_abs == 10_000.0
    assert c.per_trade_cap == 2_000.0
    assert c.max_round_trips == 20
    assert c.daily_loss_halt_pct == 0.015
    assert c.flat_by_close_minutes == 15
    assert c.tick_seconds == 60
    assert c.mean_reversion_lookback == 30
    assert c.entry_z == 2.0
    assert c.exit_z == 0.5


def test_rejects_nonpositive_caps():
    with pytest.raises(ValueError):
        SleeveConfig(per_trade_cap=0.0)
    with pytest.raises(ValueError):
        SleeveConfig(notional_cap_abs=-1.0)


def test_rejects_zero_max_round_trips():
    with pytest.raises(ValueError):
        SleeveConfig(max_round_trips=0)


def test_rejects_entry_z_not_greater_than_exit_z():
    with pytest.raises(ValueError):
        SleeveConfig(entry_z=0.4, exit_z=2.0)


def test_sleeve_allocation_is_min_of_pct_and_abs():
    c = SleeveConfig(notional_cap_pct=0.10, notional_cap_abs=10_000.0)
    assert c.sleeve_allocation(equity=50_000.0) == 5_000.0  # pct binds
    assert c.sleeve_allocation(equity=200_000.0) == 10_000.0  # abs cap binds
