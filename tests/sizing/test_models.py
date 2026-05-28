from __future__ import annotations

import dataclasses

from quant.sizing.models import DEFAULT_REGIME_WEIGHTS, SizingConfig, SizingDecision


def test_default_config_values() -> None:
    c = SizingConfig()
    assert c.target_vol == 0.15
    assert c.vol_lookback_days == 63
    assert c.max_leverage == 2.0
    assert c.kelly_fraction == 0.5
    assert c.kelly_cap == 1.0
    assert c.kelly_lookback_days == 252
    assert c.dd_floor == 0.20
    assert c.dd_lookback_days == 252
    assert c.use_vol_target and c.use_kelly and c.use_drawdown and c.use_regime
    assert dict(c.regime_weights) == DEFAULT_REGIME_WEIGHTS


def test_config_is_frozen() -> None:
    c = SizingConfig()
    try:
        c.target_vol = 0.10  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("SizingConfig should be frozen")


def test_two_default_configs_share_equal_weights() -> None:
    # default_factory must not leak a shared mutable across instances in a way
    # that diverges; equal value, independent identity is fine.
    assert dict(SizingConfig().regime_weights) == dict(SizingConfig().regime_weights)


def test_sizing_decision_fields() -> None:
    d = SizingDecision(gross=1.5, vol_scale=1.2, kelly=1.0, drawdown=1.0, regime=1.0)
    assert d.gross == 1.5
    assert d.vol_scale == 1.2
    assert d.kelly == 1.0
    assert d.drawdown == 1.0
    assert d.regime == 1.0
