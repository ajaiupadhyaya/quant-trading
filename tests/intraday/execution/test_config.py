import pytest

from quant.intraday.execution.config import ExecConfig


def test_defaults():
    c = ExecConfig()
    assert c.horizon_ticks == 5
    assert c.risk_aversion > 0
    assert 0.0 <= c.perm_impact_frac <= 1.0
    assert c.sigma_lookback_bars > 0
    assert c.impact_coef_bps > 0


def test_rejects_bad_values():
    with pytest.raises(ValueError):
        ExecConfig(horizon_ticks=0)
    with pytest.raises(ValueError):
        ExecConfig(risk_aversion=0.0)
    with pytest.raises(ValueError):
        ExecConfig(perm_impact_frac=1.5)
