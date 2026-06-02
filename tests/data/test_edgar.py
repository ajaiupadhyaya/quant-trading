"""Tests for the SEC EDGAR PIT fundamentals pipeline.

Tests use mocked HTTP responses so they're hermetic / fast / never hit SEC.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from quant.data.edgar import (
    asset_growth_yoy,
    book_to_market,
    cik_for_ticker,
    fetch_company_facts,
    get_facts_asof,
    gross_profitability,
    market_cap_asof,
)


def _mock_ticker_map() -> dict[str, dict[str, object]]:
    return {
        "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
        "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corp."},
    }


def _mock_companyfacts() -> dict[str, object]:
    """Synthetic AAPL company-facts payload with the concepts we care about."""
    return {
        "facts": {
            "us-gaap": {
                "Assets": {
                    "units": {
                        "USD": [
                            {"val": 300_000_000_000, "end": "2022-09-24", "filed": "2022-10-28"},
                            {"val": 350_000_000_000, "end": "2023-09-30", "filed": "2023-11-03"},
                            {"val": 365_000_000_000, "end": "2024-09-28", "filed": "2024-11-01"},
                        ]
                    }
                },
                "StockholdersEquity": {
                    "units": {
                        "USD": [
                            {"val": 50_000_000_000, "end": "2023-09-30", "filed": "2023-11-03"},
                            {"val": 55_000_000_000, "end": "2024-09-28", "filed": "2024-11-01"},
                        ]
                    }
                },
                "GrossProfit": {
                    "units": {
                        "USD": [
                            {"val": 170_000_000_000, "end": "2023-09-30", "filed": "2023-11-03"},
                        ]
                    }
                },
            },
            "dei": {
                "EntityCommonStockSharesOutstanding": {
                    "units": {
                        "shares": [
                            {"val": 15_550_000_000, "end": "2023-09-30", "filed": "2023-11-03"},
                            {"val": 15_000_000_000, "end": "2024-09-28", "filed": "2024-11-01"},
                        ]
                    }
                },
            },
        }
    }


class _Resp:
    def __init__(self, payload: dict, status_code: int = 200) -> None:  # type: ignore[type-arg]
        self._payload = payload
        self.status_code = status_code

    def json(self) -> dict:  # type: ignore[type-arg]
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def _patch_http(monkeypatch, ticker_map=None, facts=None):  # type: ignore[no-untyped-def]
    ticker_map = ticker_map or _mock_ticker_map()
    facts = facts or _mock_companyfacts()

    def _fake_get(url: str, **_kw):  # type: ignore[no-untyped-def]
        if "company_tickers.json" in url:
            return _Resp(ticker_map)
        if "companyfacts" in url:
            return _Resp(facts)
        raise RuntimeError(f"unexpected URL {url}")

    monkeypatch.setattr("quant.data.edgar._http_get", _fake_get)


def test_cik_lookup_uppercase(monkeypatch, tmp_path: Path) -> None:
    _patch_http(monkeypatch)
    assert cik_for_ticker("aapl", data_dir=tmp_path) == "0000320193"


def test_cik_lookup_unknown(monkeypatch, tmp_path: Path) -> None:
    _patch_http(monkeypatch)
    assert cik_for_ticker("NOPE", data_dir=tmp_path) is None


def test_fetch_company_facts_caches(monkeypatch, tmp_path: Path) -> None:
    _patch_http(monkeypatch)
    df = fetch_company_facts("AAPL", data_dir=tmp_path)
    assert not df.empty
    assert set(df["concept"]).issuperset({"total_assets", "stockholders_equity", "gross_profit"})
    # Hitting again with the cached file should not call _http_get.
    monkeypatch.setattr(
        "quant.data.edgar._http_get",
        lambda url, **_: (_ for _ in ()).throw(AssertionError(f"unexpected fetch: {url}")),
    )
    df2 = fetch_company_facts("AAPL", data_dir=tmp_path)
    assert len(df2) == len(df)


def test_fetch_company_facts_in_process_cache_skips_parquet_read(
    monkeypatch, tmp_path: Path
) -> None:
    """The hot loop in multi-factor calls fetch_company_facts thousands of times
    per backtest. After the parquet is on disk, subsequent calls for the same
    (ticker, data_dir) must NOT touch the disk — that I/O dominated multi-factor
    validate runs at >2 hours."""
    import pandas as pd

    import quant.data.edgar as edgar_mod

    edgar_mod.fetch_company_facts.cache_clear()
    _patch_http(monkeypatch)
    df1 = fetch_company_facts("AAPL", data_dir=tmp_path)
    assert not df1.empty

    real_read_parquet = pd.read_parquet
    calls = {"n": 0}

    def counting_read_parquet(*args, **kwargs):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        return real_read_parquet(*args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", counting_read_parquet)
    for _ in range(10):
        df_n = fetch_company_facts("AAPL", data_dir=tmp_path)
        assert len(df_n) == len(df1)

    assert calls["n"] == 0, (
        f"fetch_company_facts re-read parquet {calls['n']}x after first call; "
        "expected in-process cache to suppress disk I/O"
    )


def test_get_facts_asof_is_pit_correct(monkeypatch, tmp_path: Path) -> None:
    """A fact filed AFTER asof must not appear in the PIT cut."""
    _patch_http(monkeypatch)
    fetch_company_facts("AAPL", data_dir=tmp_path)
    # 2023-10-01 is BEFORE the 2023-11-03 filing → that fact is invisible.
    pit = get_facts_asof("AAPL", date(2023, 10, 1), data_dir=tmp_path)
    assert "total_assets" in pit
    assert pit["total_assets"].period_end == date(2022, 9, 24)
    assert pit["total_assets"].filed == date(2022, 10, 28)


def test_get_facts_asof_picks_latest_available(monkeypatch, tmp_path: Path) -> None:
    _patch_http(monkeypatch)
    fetch_company_facts("AAPL", data_dir=tmp_path)
    pit = get_facts_asof("AAPL", date(2024, 12, 1), data_dir=tmp_path)
    # 2024-11-01 filing IS visible → latest = period_end 2024-09-28.
    assert pit["total_assets"].period_end == date(2024, 9, 28)
    assert pit["total_assets"].value == 365_000_000_000


def test_book_to_market_and_profitability(monkeypatch, tmp_path: Path) -> None:
    _patch_http(monkeypatch)
    fetch_company_facts("AAPL", data_dir=tmp_path)
    btm = book_to_market("AAPL", date(2024, 1, 1), market_cap=2_500_000_000_000, data_dir=tmp_path)
    assert btm is not None
    assert abs(btm - (50_000_000_000 / 2_500_000_000_000)) < 1e-12
    gp = gross_profitability("AAPL", date(2024, 1, 1), data_dir=tmp_path)
    assert gp is not None
    assert abs(gp - (170_000_000_000 / 350_000_000_000)) < 1e-12


def test_shares_outstanding_extracted_from_dei_namespace(monkeypatch, tmp_path: Path) -> None:
    """Shares outstanding lives in the `dei` namespace under the `shares` unit,
    not us-gaap/USD. fetch_company_facts must surface it as its own concept."""
    _patch_http(monkeypatch)
    df = fetch_company_facts("AAPL", data_dir=tmp_path)
    assert "shares_outstanding" in set(df["concept"])
    pit = get_facts_asof("AAPL", date(2024, 1, 1), data_dir=tmp_path)
    assert "shares_outstanding" in pit
    # 2024-11-01 filing invisible at asof 2024-01-01 → latest visible = 2023-11-03.
    assert pit["shares_outstanding"].value == 15_550_000_000
    assert pit["shares_outstanding"].unit == "shares"


def test_market_cap_asof_is_price_times_shares(monkeypatch, tmp_path: Path) -> None:
    _patch_http(monkeypatch)
    fetch_company_facts("AAPL", data_dir=tmp_path)
    mcap = market_cap_asof("AAPL", date(2024, 1, 1), price=190.0, data_dir=tmp_path)
    assert mcap is not None
    assert abs(mcap - 190.0 * 15_550_000_000) < 1.0


def test_market_cap_asof_none_when_shares_missing(monkeypatch, tmp_path: Path) -> None:
    # A payload without any shares-outstanding fact → market cap unavailable.
    facts = _mock_companyfacts()
    del facts["facts"]["dei"]  # type: ignore[index]
    _patch_http(monkeypatch, facts=facts)
    fetch_company_facts("AAPL", data_dir=tmp_path)
    assert market_cap_asof("AAPL", date(2024, 1, 1), price=190.0, data_dir=tmp_path) is None


def test_shares_outstanding_falls_back_to_weighted_average(monkeypatch, tmp_path: Path) -> None:
    """Some filers (e.g. META) omit dei:EntityCommonStockSharesOutstanding and
    only report us-gaap weighted-average share counts. Fall back to those so the
    name keeps a market cap rather than silently dropping its value factor."""
    facts = {
        "facts": {
            "us-gaap": {
                "StockholdersEquity": {
                    "units": {"USD": [{"val": 50e9, "end": "2023-09-30", "filed": "2023-11-03"}]}
                },
                "WeightedAverageNumberOfDilutedSharesOutstanding": {
                    "units": {
                        "shares": [{"val": 2.5e9, "end": "2023-09-30", "filed": "2023-11-03"}]
                    }
                },
            }
        }
    }
    _patch_http(monkeypatch, facts=facts)
    fetch_company_facts("AAPL", data_dir=tmp_path)
    mcap = market_cap_asof("AAPL", date(2024, 1, 1), price=300.0, data_dir=tmp_path)
    assert mcap is not None
    assert abs(mcap - 300.0 * 2.5e9) < 1.0


def test_market_cap_asof_none_for_nonpositive_price(monkeypatch, tmp_path: Path) -> None:
    _patch_http(monkeypatch)
    fetch_company_facts("AAPL", data_dir=tmp_path)
    assert market_cap_asof("AAPL", date(2024, 1, 1), price=0.0, data_dir=tmp_path) is None


def test_asset_growth_yoy(monkeypatch, tmp_path: Path) -> None:
    _patch_http(monkeypatch)
    fetch_company_facts("AAPL", data_dir=tmp_path)
    ag = asset_growth_yoy("AAPL", date(2024, 1, 1), data_dir=tmp_path)
    # 2023-09-30 vs 2022-09-24 → ~16.67% growth
    assert ag is not None
    assert 0.15 < ag < 0.18


def test_returns_none_when_ticker_missing(monkeypatch, tmp_path: Path) -> None:
    _patch_http(monkeypatch, ticker_map={})  # empty map
    df = fetch_company_facts("XYZ", data_dir=tmp_path)
    assert df.empty
    assert get_facts_asof("XYZ", date(2024, 1, 1), data_dir=tmp_path) == {}


def test_ticker_map_caches_locally(monkeypatch, tmp_path: Path) -> None:
    _patch_http(monkeypatch)
    cik_for_ticker("AAPL", data_dir=tmp_path)
    cache = tmp_path / "fundamentals" / "_ticker_to_cik.json"
    assert cache.exists()
    payload = json.loads(cache.read_text())
    assert payload["AAPL"] == "0000320193"
