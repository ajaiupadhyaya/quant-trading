import pytest

from quant.options.models import HedgeConfig, HedgeDecision, HedgeStructure, OptionLeg


def test_structure_value_sums_signed_legs():
    legs = (OptionLeg("put", 95.0, 1.0), OptionLeg("call", 105.0, -1.0))
    struct = HedgeStructure(legs=legs, spot_at_open=100.0, expiry_index=21)
    from quant.options.pricing import bs_price

    expected = bs_price(100.0, 95.0, 0.08, 0.2, 0.03, 0.0, "put") - bs_price(
        100.0, 105.0, 0.08, 0.2, 0.03, 0.0, "call"
    )
    assert struct.value(100.0, 0.08, 0.2, 0.03, 0.0) == pytest.approx(expected)


def test_config_defaults():
    cfg = HedgeConfig()
    assert cfg.structure == "put"
    assert cfg.put_moneyness == 0.05
    assert cfg.coverage == 1.0
    assert cfg.tenor_days == 30
    assert cfg.roll_days == 21
    assert cfg.use_regime is True
    assert cfg.regime_intensity["crisis"] == 1.0


def test_config_is_frozen():
    cfg = HedgeConfig()
    with pytest.raises(Exception):
        cfg.coverage = 2.0  # type: ignore[misc]


def test_decision_record_fields():
    legs = (OptionLeg("put", 95.0, 1.0),)
    struct = HedgeStructure(legs=legs, spot_at_open=100.0, expiry_index=21)
    dec = HedgeDecision(
        structure=struct,
        contracts=2.0,
        premium=3.5,
        net_beta=0.9,
        regime_label="choppy",
        intensity=0.6,
    )
    assert dec.contracts == 2.0
    assert dec.regime_label == "choppy"
