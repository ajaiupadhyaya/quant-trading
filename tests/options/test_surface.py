"""Track F — live vol-surface read (IV / term / skew from real option mids)."""

from __future__ import annotations

from datetime import date, timedelta
from types import SimpleNamespace

import pytest

import quant.options.surface as surf
from quant.options.pricing import bs_price
from quant.options.surface import (
    OptionQuote,
    VolSurfaceConfig,
    compute_vol_surface,
    live_vol_surface,
    parse_occ_symbol,
    render_vol_surface,
)

ASOF = date(2026, 6, 4)
SPOT = 750.0
CFG = VolSurfaceConfig()


def _mk(dte: int, strike: float, right: str, vol: float) -> OptionQuote:
    """A quote whose mid is the BSM price at ``vol`` (so IV recovery == vol)."""
    t = dte / 365.0
    mid = bs_price(SPOT, strike, t, vol, CFG.r, CFG.q, right)
    return OptionQuote(expiry=ASOF + timedelta(days=dte), strike=strike, right=right, mid=mid)


def test_parse_occ_symbol() -> None:
    assert parse_occ_symbol("SPY260605P00645000") == ("SPY", date(2026, 6, 5), "put", 645.0)
    assert parse_occ_symbol("SPY260918C00750000") == ("SPY", date(2026, 9, 18), "call", 750.0)
    assert parse_occ_symbol("garbage") is None


def test_recovers_atm_iv_and_term_and_skew() -> None:
    quotes = [
        _mk(28, 750.0, "call", 0.16),  # near ATM call
        _mk(28, 787.5, "call", 0.14),  # near 105% call (skew leg)
        _mk(28, 712.5, "put", 0.22),  # near 95% put (skew leg)
        _mk(88, 750.0, "call", 0.17),  # far ATM call
    ]
    v = compute_vol_surface(quotes, SPOT, ASOF, config=CFG)
    assert v.atm_iv_30d is not None and abs(v.atm_iv_30d - 0.16) < 1e-4
    assert v.atm_iv_90d is not None and abs(v.atm_iv_90d - 0.17) < 1e-4
    assert v.term_slope is not None and abs(v.term_slope - 0.01) < 1e-4
    assert v.put_skew is not None and abs(v.put_skew - 0.08) < 1e-4
    assert v.iv_regime == "normal"  # 0.16 in [0.12, 0.18)
    assert v.term_label == "contango"  # +0.01 > 0.005
    assert v.tail_label == "elevated"  # 0.08 in [0.05, 0.09)
    assert v.near_dte == 28 and v.far_dte == 88


def test_stressed_and_backwardation_and_extreme() -> None:
    quotes = [
        _mk(30, 750.0, "call", 0.32),  # ATM IV high → stressed
        _mk(30, 787.5, "call", 0.26),
        _mk(30, 712.5, "put", 0.40),  # skew 0.40-0.26 = 0.14 → extreme
        _mk(90, 750.0, "call", 0.28),  # far < near → backwardation
    ]
    v = compute_vol_surface(quotes, SPOT, ASOF, config=CFG)
    assert v.iv_regime == "stressed"  # >= 0.28
    assert v.term_label == "backwardation"  # 0.28 - 0.32 < -0.005
    assert v.tail_label == "extreme"  # >= 0.09


def test_empty_quotes_never_raises() -> None:
    v = compute_vol_surface([], SPOT, ASOF, config=CFG)
    assert v.n_quotes == 0 and v.atm_iv_30d is None
    assert v.iv_regime is None and v.term_label is None and v.tail_label is None


def test_no_spot_degrades() -> None:
    v = compute_vol_surface([_mk(30, 750.0, "call", 0.16)], None, ASOF, config=CFG)
    assert v.spot is None and v.atm_iv_30d is None


def test_single_expiry_has_no_term_slope() -> None:
    quotes = [_mk(28, 750.0, "call", 0.16), _mk(28, 712.5, "put", 0.20)]
    v = compute_vol_surface(quotes, SPOT, ASOF, config=CFG)
    assert v.atm_iv_30d is not None
    assert v.atm_iv_90d is None and v.term_slope is None and v.term_label is None


def test_dte_out_of_range_filtered() -> None:
    # 2-day (below min_dte) and 200-day (above max_dte) are dropped.
    quotes = [_mk(2, 750.0, "call", 0.16), _mk(200, 750.0, "call", 0.16)]
    v = compute_vol_surface(quotes, SPOT, ASOF, config=CFG)
    assert v.n_quotes == 0 and v.atm_iv_30d is None


def test_live_vol_surface_failopen_no_spot(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(surf, "_spot_from_cache", lambda *a, **k: None)
    v = live_vol_surface(SimpleNamespace(), ASOF)
    assert v.spot is None and v.atm_iv_30d is None  # no spot → no fetch, no raise


def test_live_vol_surface_failopen_chain_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(surf, "_spot_from_cache", lambda *a, **k: 750.0)
    monkeypatch.setattr(
        surf, "_fetch_quotes", lambda *a, **k: (_ for _ in ()).throw(RuntimeError())
    )
    v = live_vol_surface(SimpleNamespace(), ASOF)
    assert v.spot == 750.0 and v.atm_iv_30d is None  # chain down → degraded, never raises


def test_render() -> None:
    assert render_vol_surface(None) == "Vol surface: unavailable"
    quotes = [
        _mk(28, 750.0, "call", 0.16),
        _mk(28, 787.5, "call", 0.14),
        _mk(28, 712.5, "put", 0.22),
        _mk(88, 750.0, "call", 0.17),
    ]
    out = render_vol_surface(compute_vol_surface(quotes, SPOT, ASOF, config=CFG))
    assert "iv_regime=normal" in out
    assert "ATM_IV30=16.0%" in out
    assert "term=contango" in out
    assert "tail=elevated" in out
