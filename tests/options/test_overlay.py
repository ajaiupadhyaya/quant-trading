import numpy as np
import pandas as pd
import pytest

from quant.options.models import HedgeConfig
from quant.options.overlay import HedgeComparison, apply_hedge, compare_hedge, cvar, worst_day


def _series(vals, start="2020-01-01"):
    idx = pd.date_range(start, periods=len(vals), freq="B")
    return pd.Series(vals, index=idx)


def test_cvar_and_worst_day():
    r = _series([-0.05, -0.03, 0.01, 0.02, -0.10, 0.04] * 10)
    assert worst_day(r) == pytest.approx(-0.10)
    assert cvar(r, alpha=0.1) < 0.0  # negative tail mean
    assert cvar(r, alpha=0.1) <= r.mean()


def test_protective_put_reduces_drawdown_in_crash():
    rng = np.random.default_rng(1)
    calm = rng.normal(0.0005, 0.008, 200)
    crash = np.array([-0.05, -0.07, -0.06, -0.04, -0.08, -0.03])
    book = np.concatenate([calm, crash])
    spy_ret = book.copy()
    idx = pd.date_range("2020-01-01", periods=len(book), freq="B")
    spy_close = pd.Series(100.0 * np.cumprod(1 + spy_ret), index=idx)
    returns = pd.Series(book, index=idx)
    cfg = HedgeConfig(structure="put", use_regime=False, coverage=1.0)
    comp = compare_hedge(returns, spy_close, cfg)
    assert comp.hedged["max_drawdown"] >= comp.baseline["max_drawdown"]
    assert comp.hedged["cvar_5"] >= comp.baseline["cvar_5"]


def test_hedge_drags_cagr_in_calm_uptrend():
    rng = np.random.default_rng(2)
    book = rng.normal(0.0008, 0.006, 400)  # steady uptrend, no crash
    idx = pd.date_range("2020-01-01", periods=len(book), freq="B")
    spy_close = pd.Series(100.0 * np.cumprod(1 + book), index=idx)
    returns = pd.Series(book, index=idx)
    cfg = HedgeConfig(structure="put", use_regime=False)
    comp = compare_hedge(returns, spy_close, cfg)
    assert comp.hedged["cagr"] < comp.baseline["cagr"]  # insurance cost is real
    assert comp.total_premium > 0.0
    assert comp.n_rolls >= 1


def test_apply_hedge_truncation_invariance():
    rng = np.random.default_rng(3)
    book = rng.normal(0.0003, 0.01, 250)
    idx = pd.date_range("2020-01-01", periods=len(book), freq="B")
    spy_close = pd.Series(100.0 * np.cumprod(1 + book), index=idx)
    returns = pd.Series(book, index=idx)
    cfg = HedgeConfig(use_regime=False)
    full, _ = apply_hedge(returns, spy_close, cfg)
    t = 180
    trunc, _ = apply_hedge(returns.iloc[:t], spy_close.iloc[:t], cfg)
    np.testing.assert_allclose(full.iloc[:t].to_numpy(), trunc.to_numpy(), atol=0.0)


def test_comparison_is_frozen_with_expected_keys():
    rng = np.random.default_rng(4)
    book = rng.normal(0.0005, 0.008, 120)
    idx = pd.date_range("2020-01-01", periods=len(book), freq="B")
    spy_close = pd.Series(100.0 * np.cumprod(1 + book), index=idx)
    comp = compare_hedge(pd.Series(book, index=idx), spy_close, HedgeConfig(use_regime=False))
    assert isinstance(comp, HedgeComparison)
    for key in ("sharpe", "max_drawdown", "cvar_5", "worst_day", "cagr"):
        assert key in comp.hedged and key in comp.baseline
    with pytest.raises(Exception):
        comp.total_premium = 0.0  # type: ignore[misc]
