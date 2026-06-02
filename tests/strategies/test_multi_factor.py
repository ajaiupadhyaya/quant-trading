"""Targeted tests for ``MultiFactor`` regime overlay integration.

The general smoke tests live in ``test_concrete_strategies.py``. This file is
for behavior unique to the strategy's regime-overlay plumbing (Task 1.3 of the
2026-05-25 go-live plan).
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd


def test_multi_factor_overlay_reduces_exposure_when_spy_below_200dma() -> None:
    """Overlay should cut gross notional when SPY is below its 200dma."""
    from quant.strategies.multi_factor import MEGACAP_UNIVERSE, MultiFactor

    idx = pd.date_range("2022-01-03", periods=400, freq="B")
    idx.name = "timestamp"
    rng = np.random.default_rng(7)

    # Strategy universe (megacap) — synthetic random walks, no SPY here.
    frames: dict[str, pd.DataFrame] = {}
    for sym in MEGACAP_UNIVERSE:
        close = pd.Series(
            100.0 * np.exp(np.cumsum(rng.normal(0.0006, 0.012, len(idx)))),
            index=idx,
        )
        frames[sym] = pd.DataFrame(
            {"open": close, "high": close, "low": close, "close": close, "volume": 1_000_000},
            index=idx,
        )
    bars = pd.concat(frames, axis=1)

    # Separate SPY frame, crashing through 200dma.
    spy_close = pd.Series(np.linspace(450.0, 300.0, len(idx)), index=idx)
    spy_df = pd.DataFrame(
        {"open": spy_close, "high": spy_close, "low": spy_close, "close": spy_close, "volume": 1},
        index=idx,
    )
    spy_bars = pd.concat({"SPY": spy_df}, axis=1)

    vix = pd.Series(15.0, index=idx, name="vix")  # calm VIX, SPY gate dominates

    # Avoid fundamentals/network deps in the test.
    base_params = {
        "use_fundamentals": False,
        "min_history_days": 252,
    }

    strat_off = MultiFactor(
        bars=bars,
        params={**base_params, "regime_overlay_enabled": False},
        vix=vix,
        spy_bars=spy_bars,
    )
    strat_on = MultiFactor(
        bars=bars,
        params={**base_params, "regime_overlay_enabled": True},
        vix=vix,
        spy_bars=spy_bars,
    )
    asof = idx[-1].date()
    pos_off = strat_off.target_positions(asof, 200_000.0)
    pos_on = strat_on.target_positions(asof, 200_000.0)

    if not pos_off:
        # Skip if synthetic data didn't produce any signals — not the path under test.
        import pytest

        pytest.skip("synthetic factor panel produced no picks; cannot assert overlay effect")

    last_close = bars.xs("close", axis=1, level=1).iloc[-1]
    notional_off = sum(
        abs(s) * float(last_close[sym]) for sym, s in pos_off.items() if sym in last_close
    )
    notional_on = sum(
        abs(s) * float(last_close[sym]) for sym, s in pos_on.items() if sym in last_close
    )
    # Factor is 0.5 when SPY below 200dma; allow rounding slack.
    assert notional_on <= notional_off * 0.7, f"on={notional_on:.0f} off={notional_off:.0f}"


def _facts_with_shares(equity: float, shares: float) -> dict:  # type: ignore[type-arg]
    """A minimal company-facts payload: one equity fact + one shares fact."""
    return {
        "facts": {
            "us-gaap": {
                "StockholdersEquity": {
                    "units": {"USD": [{"val": equity, "end": "2023-09-30", "filed": "2023-11-03"}]}
                },
            },
            "dei": {
                "EntityCommonStockSharesOutstanding": {
                    "units": {
                        "shares": [{"val": shares, "end": "2023-09-30", "filed": "2023-11-03"}]
                    }
                },
            },
        }
    }


def test_fundamentals_panel_uses_real_market_cap_not_price(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Two names with identical book equity AND identical price but different
    shares-outstanding must get DIFFERENT book-to-market. This only holds if the
    panel computes market cap = price * shares (the fix). The prior bug used
    price as a market-cap proxy, which would make them identical."""
    import quant.data.edgar as edgar_mod
    from quant.strategies.multi_factor import MultiFactor

    edgar_mod.fetch_company_facts.cache_clear()
    monkeypatch.setenv("QUANT_DATA_DIR", str(tmp_path))

    # Same equity, same price; AAPL has 3x the share count of MSFT -> 3x market
    # cap -> 1/3 the book-to-market.
    facts_by_cik = {
        "0000320193": _facts_with_shares(equity=60e9, shares=15e9),  # AAPL
        "0000789019": _facts_with_shares(equity=60e9, shares=5e9),  # MSFT
    }
    ticker_map = {
        "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
        "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corp."},
    }

    class _Resp:
        def __init__(self, payload: dict) -> None:  # type: ignore[type-arg]
            self._p = payload

        def json(self) -> dict:  # type: ignore[type-arg]
            return self._p

        def raise_for_status(self) -> None:
            return None

    def _fake_get(url: str, **_kw):  # type: ignore[no-untyped-def]
        if "company_tickers.json" in url:
            return _Resp(ticker_map)
        for cik, facts in facts_by_cik.items():
            if cik in url:
                return _Resp(facts)
        raise RuntimeError(f"unexpected URL {url}")

    monkeypatch.setattr("quant.data.edgar._http_get", _fake_get)

    idx = pd.date_range("2024-01-02", periods=2, freq="B")
    bars = pd.concat(
        {sym: pd.DataFrame({"close": [100.0, 100.0]}, index=idx) for sym in ("AAPL", "MSFT")},
        axis=1,
    )
    strat = MultiFactor(bars=bars, params={"use_fundamentals": True})

    prices = pd.Series({"AAPL": 100.0, "MSFT": 100.0})
    panel = strat._fundamentals_panel(date(2024, 1, 3), ["AAPL", "MSFT"], prices)

    assert "book_to_market" in panel.columns
    btm = panel["book_to_market"]
    assert btm["AAPL"] == btm["AAPL"] and btm["MSFT"] == btm["MSFT"]  # not NaN
    # 60e9 / (100 * 15e9) = 0.04  vs  60e9 / (100 * 5e9) = 0.12
    assert abs(btm["MSFT"] - 3.0 * btm["AAPL"]) < 1e-9, f"AAPL={btm['AAPL']} MSFT={btm['MSFT']}"
