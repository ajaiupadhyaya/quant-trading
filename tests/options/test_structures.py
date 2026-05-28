import pytest

from quant.options.models import HedgeConfig
from quant.options.structures import build_structure, collar, protective_put, put_spread

SPOT = 100.0


def test_protective_put_strike_and_sign():
    cfg = HedgeConfig(put_moneyness=0.05)
    s = protective_put(SPOT, cfg)
    assert len(s.legs) == 1
    assert s.legs[0].right == "put"
    assert s.legs[0].strike == pytest.approx(95.0)
    assert s.legs[0].quantity == 1.0
    assert s.spot_at_open == SPOT


def test_collar_legs():
    cfg = HedgeConfig(put_moneyness=0.05, call_moneyness=0.05)
    s = collar(SPOT, cfg)
    rights = sorted((leg.right, leg.quantity) for leg in s.legs)
    assert rights == [("call", -1.0), ("put", 1.0)]
    call = next(leg for leg in s.legs if leg.right == "call")
    assert call.strike == pytest.approx(105.0)


def test_put_spread_legs():
    cfg = HedgeConfig(put_moneyness=0.05, spread_width=0.10)
    s = put_spread(SPOT, cfg)
    longs = [leg for leg in s.legs if leg.quantity > 0]
    shorts = [leg for leg in s.legs if leg.quantity < 0]
    assert longs[0].strike == pytest.approx(95.0)
    assert shorts[0].strike == pytest.approx(85.0)


def test_collar_cheaper_than_bare_put():
    cfg = HedgeConfig()
    t, vol, r, q = 0.08, 0.2, 0.03, 0.015
    bare = protective_put(SPOT, cfg).value(SPOT, t, vol, r, q)
    col = collar(SPOT, cfg).value(SPOT, t, vol, r, q)
    assert col < bare  # short call finances the put


def test_put_spread_cheaper_than_bare_put():
    cfg = HedgeConfig()
    t, vol, r, q = 0.08, 0.2, 0.03, 0.015
    bare = protective_put(SPOT, cfg).value(SPOT, t, vol, r, q)
    spread = put_spread(SPOT, cfg).value(SPOT, t, vol, r, q)
    assert 0.0 < spread < bare


def test_build_structure_dispatch():
    cfg = HedgeConfig(structure="collar")
    s = build_structure(SPOT, cfg)
    assert len(s.legs) == 2


def test_build_structure_unknown_raises():
    cfg = HedgeConfig(structure="butterfly")
    with pytest.raises(ValueError):
        build_structure(SPOT, cfg)
