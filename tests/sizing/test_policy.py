from __future__ import annotations

import math

import numpy as np

from quant.sizing.models import SizingConfig
from quant.sizing.policy import compute_gross


def _rng_returns(n: int, mu: float, sigma: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(mu, sigma, size=n)


def test_all_toggles_off_gives_unit_gross() -> None:
    cfg = SizingConfig(use_vol_target=False, use_kelly=False, use_drawdown=False, use_regime=False)
    d = compute_gross(_rng_returns(300, 0.0005, 0.01), "crisis", cfg)
    assert d.gross == 1.0
    assert d.vol_scale == 1.0 and d.kelly == 1.0 and d.drawdown == 1.0 and d.regime == 1.0


def test_disabled_component_is_unit() -> None:
    cfg = SizingConfig(use_kelly=False, use_drawdown=False, use_regime=False)
    d = compute_gross(_rng_returns(300, 0.0005, 0.01), "calm-bull", cfg)
    assert d.kelly == 1.0 and d.drawdown == 1.0 and d.regime == 1.0
    # vol_target still active -> vol_scale drives gross
    assert math.isclose(d.gross, d.vol_scale)


def test_gross_is_product_of_components() -> None:
    cfg = SizingConfig(max_leverage=100.0)  # high cap so no clamp
    d = compute_gross(_rng_returns(400, 0.0008, 0.012, seed=3), "choppy", cfg)
    expected = d.vol_scale * d.kelly * d.drawdown * d.regime
    assert math.isclose(d.gross, min(100.0, expected))


def test_crisis_regime_zeroes_gross() -> None:
    cfg = SizingConfig()
    d = compute_gross(_rng_returns(300, 0.0005, 0.01), "crisis", cfg)
    assert d.regime == 0.0
    assert d.gross == 0.0


def test_gross_clamped_to_max_leverage() -> None:
    # tiny vol -> vol_target wants huge leverage; cap binds
    cfg = SizingConfig(use_kelly=False, use_drawdown=False, use_regime=False, max_leverage=2.0)
    d = compute_gross(_rng_returns(300, 0.0001, 0.0005, seed=7), "calm-bull", cfg)
    assert d.gross <= 2.0


def test_empty_history_is_neutral() -> None:
    cfg = SizingConfig()
    d = compute_gross(np.array([]), "calm-bull", cfg)
    # vol/kelly/drawdown all no-op to neutral on empty; regime calm-bull = 1.0
    assert d.gross == 1.0
