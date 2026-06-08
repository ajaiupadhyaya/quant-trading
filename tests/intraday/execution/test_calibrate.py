import math

from quant.intraday.execution.calibrate import calibrate
from quant.intraday.execution.config import ExecConfig


def test_calibrate_returns_positive_params():
    recent_returns = [0.001, -0.002, 0.0015, -0.001, 0.0005] * 20
    sigma, eta, gamma = calibrate(
        price=400.0, slice_shares=20, adv_dollar=5_000_000_000.0,
        recent_returns=recent_returns, config=ExecConfig(),
    )
    assert sigma > 0 and math.isfinite(sigma)
    assert eta > 0 and math.isfinite(eta)
    assert gamma == ExecConfig().perm_impact_frac * eta


def test_zero_volatility_returns_small_positive_sigma():
    sigma, eta, _gamma = calibrate(
        price=400.0, slice_shares=20, adv_dollar=5_000_000_000.0,
        recent_returns=[0.0] * 100, config=ExecConfig(),
    )
    assert sigma >= 0.0
    assert eta > 0


def test_eta_scales_with_impact_coef():
    rr = [0.001, -0.001] * 50
    _, eta_lo, _ = calibrate(price=400.0, slice_shares=20, adv_dollar=5e9,
                             recent_returns=rr, config=ExecConfig(impact_coef_bps=5.0))
    _, eta_hi, _ = calibrate(price=400.0, slice_shares=20, adv_dollar=5e9,
                             recent_returns=rr, config=ExecConfig(impact_coef_bps=20.0))
    assert eta_hi > eta_lo
