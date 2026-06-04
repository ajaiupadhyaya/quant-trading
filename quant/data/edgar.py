"""SEC EDGAR fundamentals fetcher with point-in-time correctness.

EDGAR's ``data.sec.gov/api/xbrl/companyfacts/CIK<10-digit>.json`` endpoint
returns every reported XBRL concept for a company with the precise filing
date attached to each datapoint. We:

1. Resolve ticker -> CIK via the public ``company_tickers.json`` map (cached).
2. Fetch /companyfacts/<CIK>.json (cached per-ticker as parquet for fast reuse).
3. For each concept (book value, total assets, gross profit, revenue,
   net income), keep only rows whose ``filed`` date is on or before the
   ``as_of`` query date — this is the PIT cut that makes downstream factor
   research clean.

The Hou-Xue-Zhang factor zoo uses these in:
  * value         = book / market  (book_value / market_cap)
  * profitability = gross_profit / total_assets (Novy-Marx 2013)
  * investment    = -1 * asset growth YoY (assets_t / assets_{t-1} - 1)
  * size          = -log(market_cap)

This file owns the data plumbing; the strategy file consumes it.

Required by SEC: every request must include a User-Agent header identifying
the caller. Set EDGAR_USER_AGENT in your env or .env (defaults to a generic
research label, which is acceptable for low-volume use).
"""

from __future__ import annotations

import functools
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from quant.util.config import Settings
from quant.util.logging import logger

_EDGAR_BASE = "https://data.sec.gov"
_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
_DEFAULT_UA = "quant-trading research bot ajaiupad@gmail.com"

# Concept → (XBRL US-GAAP tags to try in order). The first tag found in the
# company's reported facts wins. EDGAR uses several different tags for the
# same economic concept depending on filing year / filer type, so we keep
# a tag preference list per concept.
_CONCEPT_TAGS: dict[str, tuple[str, ...]] = {
    "total_assets": ("Assets",),
    "stockholders_equity": (
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ),
    "revenue": (
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
    ),
    "gross_profit": ("GrossProfit",),
    "net_income": ("NetIncomeLoss",),
    # Shares outstanding is reported in the `dei` namespace under the `shares`
    # unit (and occasionally in us-gaap). We need it to turn a share PRICE into
    # a market CAP — without it, cross-sectional value/size factors degrade to
    # ranking on raw price, which is meaningless across names with wildly
    # different float (e.g. AAPL ~15B shares vs BRK-B ~1.5B).
    "shares_outstanding": (
        "EntityCommonStockSharesOutstanding",
        "CommonStockSharesOutstanding",
        # Fallbacks for filers that omit a point-in-time share count (e.g. META):
        # weighted-average diluted/basic shares are a close proxy for a mega-cap
        # and keep the name's value/size factor alive rather than dropping it.
        "WeightedAverageNumberOfDilutedSharesOutstanding",
        "WeightedAverageNumberOfSharesOutstandingBasic",
    ),
}

# Namespaces searched per concept, in order. Most concepts live in us-gaap;
# shares-outstanding lives in dei. Searching both is harmless for the rest.
_NAMESPACES: tuple[str, ...] = ("us-gaap", "dei")
# Units tried in order. USD covers the dollar concepts; `shares` covers the
# share-count concepts. USD/shares retained for any per-share fallback.
_UNITS: tuple[str, ...] = ("USD", "shares", "USD/shares")


@dataclass(frozen=True)
class FactRow:
    """One point-in-time XBRL fact.

    ``start`` is the period start for FLOW concepts (income/cash-flow statement
    items span a window); it is ``None`` for INSTANT balance-sheet concepts
    (assets, equity, shares). The window length (``period_end - start``) lets us
    distinguish a quarterly (~90d) from an annual (~365d) flow — essential for an
    honest earnings yield, since a single quarter's net income understates the
    trailing-twelve-month figure ~4x.
    """

    concept: str
    value: float
    period_end: date
    filed: date
    unit: str
    start: date | None = None


def _settings() -> Settings:
    return Settings()  # type: ignore[call-arg]


def _user_agent() -> str:
    return os.environ.get("EDGAR_USER_AGENT", _DEFAULT_UA)


def _http_get(url: str, retries: int = 3, backoff: float = 1.0) -> requests.Response:
    """GET with retries + SEC's required User-Agent header. Honours 429s."""
    headers = {"User-Agent": _user_agent(), "Accept-Encoding": "gzip, deflate"}
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=headers, timeout=20)
            if resp.status_code == 429:
                logger.warning("EDGAR 429 on {} (attempt {})", url, attempt + 1)
                time.sleep(backoff * (attempt + 2))
                continue
            resp.raise_for_status()
            return resp
        except Exception as exc:
            last_exc = exc
            time.sleep(backoff * (attempt + 1))
    raise RuntimeError(f"EDGAR GET failed after {retries} retries: {url} ({last_exc!r})")


def _ticker_to_cik_map(data_dir: Path) -> dict[str, str]:
    """Return {TICKER: 10-digit-CIK} from EDGAR's master mapping file."""
    cache_path = data_dir / "fundamentals" / "_ticker_to_cik.json"
    if cache_path.exists():
        cached: dict[str, str] = {
            str(k): str(v) for k, v in json.loads(cache_path.read_text()).items()
        }
        return cached

    logger.info("Fetching EDGAR ticker→CIK map…")
    resp = _http_get(_TICKER_MAP_URL)
    payload = resp.json()
    # payload looks like {"0": {"cik_str": 320193, "ticker": "AAPL", ...}, ...}
    out: dict[str, str] = {}
    for row in payload.values():
        ticker = str(row["ticker"]).upper()
        cik_str = f"{int(row['cik_str']):010d}"
        out[ticker] = cik_str

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(out, indent=2))
    return out


def cik_for_ticker(ticker: str, data_dir: Path | None = None) -> str | None:
    """Look up the 10-digit CIK for a ticker. Returns None if not in EDGAR's map."""
    data_dir = data_dir or _settings().data_dir
    # Normalize dashed BRK-B → BRK.B for EDGAR's map (we store both forms).
    candidates = {
        ticker.upper(),
        ticker.upper().replace("-", "."),
        ticker.upper().replace(".", "-"),
    }
    mapping = _ticker_to_cik_map(data_dir)
    for c in candidates:
        if c in mapping:
            return mapping[c]
    return None


def _facts_path(ticker: str, data_dir: Path) -> Path:
    safe = re.sub(r"[^A-Z0-9_]", "_", ticker.upper())
    # _v2 carries the period-start column; pre-v2 caches lack it and are ignored.
    return data_dir / "fundamentals" / f"{safe}_facts_v2.parquet"


def _extract_concept(facts: dict[str, Any], concept: str) -> list[FactRow]:
    """Pull one concept out of the /companyfacts payload. Returns sorted rows.

    Searches each candidate tag across both the us-gaap and dei namespaces, and
    each namespace's units in preference order (USD, then shares). The first
    (tag, namespace, unit) triple with rows wins — this lets dollar concepts
    resolve from us-gaap/USD while shares-outstanding resolves from dei/shares.
    """
    facts_root = facts.get("facts", {})
    tags = _CONCEPT_TAGS.get(concept, (concept,))
    for tag in tags:
        for ns_name in _NAMESPACES:
            ns = facts_root.get(ns_name, {})
            if tag not in ns:
                continue
            unit_map = ns[tag].get("units", {})
            for unit_name in _UNITS:
                rows = unit_map.get(unit_name, [])
                if not rows:
                    continue
                out: list[FactRow] = []
                for r in rows:
                    try:
                        start_raw = r.get("start")
                        out.append(
                            FactRow(
                                concept=concept,
                                value=float(r["val"]),
                                period_end=date.fromisoformat(str(r["end"])[:10]),
                                filed=date.fromisoformat(str(r["filed"])[:10]),
                                unit=unit_name,
                                start=(
                                    date.fromisoformat(str(start_raw)[:10])
                                    if start_raw is not None
                                    else None
                                ),
                            )
                        )
                    except Exception:  # malformed row; skip
                        continue
                out.sort(key=lambda f: (f.filed, f.period_end))
                return out
            # Otherwise try the next namespace / tag
    return []


@functools.lru_cache(maxsize=512)
def fetch_company_facts(ticker: str, data_dir: Path | None = None) -> pd.DataFrame:
    """Fetch + parse /companyfacts for ``ticker`` into a long-format DataFrame.

    Columns: ``concept`` ``value`` ``period_end`` ``filed`` ``unit``. Cached as
    parquet per ticker and also memoized in-process — multi-factor's hot loop
    calls this ~10M times per validate run, where the parquet round-trip
    dominates wall time even though the file is tiny.
    """
    data_dir = data_dir or _settings().data_dir
    path = _facts_path(ticker, data_dir)
    if path.exists():
        return pd.read_parquet(path)

    cik = cik_for_ticker(ticker, data_dir)
    if cik is None:
        logger.warning("No EDGAR CIK for ticker {}", ticker)
        empty = pd.DataFrame(columns=["concept", "value", "period_end", "filed", "unit", "start"])
        empty.to_parquet(path)
        return empty

    url = f"{_EDGAR_BASE}/api/xbrl/companyfacts/CIK{cik}.json"
    logger.info("EDGAR fetch {}: {}", ticker, url)
    facts = _http_get(url).json()

    rows: list[FactRow] = []
    for concept in _CONCEPT_TAGS:
        rows.extend(_extract_concept(facts, concept))

    df = pd.DataFrame(
        [
            {
                "concept": r.concept,
                "value": r.value,
                "period_end": r.period_end,
                "filed": r.filed,
                "unit": r.unit,
                "start": r.start,
            }
            for r in rows
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df


def get_facts_asof(
    ticker: str,
    asof: date,
    data_dir: Path | None = None,
) -> dict[str, FactRow]:
    """Return the most recent fact per concept for ``ticker`` filed on or before ``asof``.

    This is the PIT cut: any fact filed AFTER ``asof`` is invisible. The returned
    map is keyed by concept; values are the FactRow with the largest ``period_end``
    among rows whose ``filed`` ≤ ``asof``.
    """
    df = fetch_company_facts(ticker, data_dir)
    if df.empty:
        return {}
    pit = df[df["filed"] <= pd.Timestamp(asof).date()]
    if pit.empty:
        return {}
    out: dict[str, FactRow] = {}
    for concept, sub in pit.groupby("concept"):
        latest = sub.sort_values(["period_end", "filed"]).iloc[-1]
        out[str(concept)] = FactRow(
            concept=str(concept),
            value=float(latest["value"]),
            period_end=pd.Timestamp(latest["period_end"]).date(),
            filed=pd.Timestamp(latest["filed"]).date(),
            unit=str(latest["unit"]),
        )
    return out


def market_cap_asof(
    ticker: str, asof: date, price: float, data_dir: Path | None = None
) -> float | None:
    """PIT market capitalisation = ``price`` * shares-outstanding (filed ≤ asof).

    Returns None when the price is non-positive or no PIT shares-outstanding
    fact is available. This is the correct input for book-to-market and size
    factors — using raw ``price`` as a market-cap proxy is wrong across names
    because share counts differ by orders of magnitude.
    """
    if price <= 0:
        return None
    facts = get_facts_asof(ticker, asof, data_dir)
    shares = facts.get("shares_outstanding")
    if shares is None or shares.value <= 0:
        return None
    return float(price) * float(shares.value)


def book_to_market(
    ticker: str, asof: date, market_cap: float, data_dir: Path | None = None
) -> float | None:
    """Book / market ratio from PIT stockholders_equity."""
    if market_cap <= 0:
        return None
    facts = get_facts_asof(ticker, asof, data_dir)
    book = facts.get("stockholders_equity")
    if book is None or book.value <= 0:
        return None
    return float(book.value) / float(market_cap)


def gross_profitability(ticker: str, asof: date, data_dir: Path | None = None) -> float | None:
    """Novy-Marx: gross_profit / total_assets, both PIT."""
    facts = get_facts_asof(ticker, asof, data_dir)
    gp = facts.get("gross_profit")
    ta = facts.get("total_assets")
    if gp is None or ta is None or ta.value <= 0:
        return None
    return float(gp.value) / float(ta.value)


def latest_annual_flow(
    ticker: str, asof: date, concept: str, data_dir: Path | None = None
) -> float | None:
    """Most recent ANNUAL (≈365-day) value of a FLOW concept, filed ≤ ``asof``.

    Income/cash-flow concepts are reported over both quarterly (~90d) and annual
    (~365d) windows. For a valuation gauge we want the annual (trailing 10-K)
    figure — a single quarter understates it ~4x. We select the latest fact whose
    period length falls in [300, 400] days. Returns None if no annual fact exists
    or the cache predates the period-start column.
    """
    df = fetch_company_facts(ticker, data_dir)
    if df.empty or "start" not in df.columns:
        return None
    sub = df[(df["concept"] == concept) & (df["filed"] <= pd.Timestamp(asof).date())].copy()
    sub = sub.dropna(subset=["start"])
    if sub.empty:
        return None
    period_days = (pd.to_datetime(sub["period_end"]) - pd.to_datetime(sub["start"])).dt.days
    annual = sub[(period_days >= 300) & (period_days <= 400)]
    if annual.empty:
        return None
    annual = annual.sort_values(["period_end", "filed"])
    return float(annual.iloc[-1]["value"])


def earnings_yield(
    ticker: str, asof: date, market_cap: float, data_dir: Path | None = None
) -> float | None:
    """Trailing-twelve-month earnings yield = annual net_income / market_cap, PIT.

    The inverse of the P/E ratio and the cleanest absolute valuation gauge with a
    market-level interpretation (≈5% is fairly valued; higher = cheaper). Uses the
    latest ANNUAL net income (not a single quarter — that would understate it ~4x)
    and the PIT market cap (price * shares-outstanding, from ``market_cap_asof``).
    Can be negative when the trailing year was a net loss.
    """
    if market_cap <= 0:
        return None
    ni = latest_annual_flow(ticker, asof, "net_income", data_dir)
    if ni is None:
        return None
    return ni / float(market_cap)


def gross_profitability_ttm(ticker: str, asof: date, data_dir: Path | None = None) -> float | None:
    """Novy-Marx quality with an ANNUAL gross profit numerator (PIT).

    Like ``gross_profitability`` but uses the latest annual gross profit rather
    than whatever single (often quarterly) fact sorts last — so the absolute
    level is comparable to the ~0.2-0.5 range the quality label anchors expect.
    """
    gp = latest_annual_flow(ticker, asof, "gross_profit", data_dir)
    if gp is None:
        return None
    facts = get_facts_asof(ticker, asof, data_dir)
    ta = facts.get("total_assets")
    if ta is None or ta.value <= 0:
        return None
    return gp / float(ta.value)


def asset_growth_yoy(
    ticker: str,
    asof: date,
    data_dir: Path | None = None,
) -> float | None:
    """One-year asset growth (PIT). Hou-Xue-Zhang investment factor: negate before ranking."""
    df = fetch_company_facts(ticker, data_dir)
    if df.empty:
        return None
    sub = df[(df["concept"] == "total_assets") & (df["filed"] <= pd.Timestamp(asof).date())]
    if sub.empty:
        return None
    sub = sub.sort_values(["period_end", "filed"])
    latest = sub.iloc[-1]
    target_prior = pd.Timestamp(latest["period_end"]) - pd.Timedelta(days=365)
    prior = sub[sub["period_end"] <= target_prior.date()]
    if prior.empty:
        return None
    prior_row = prior.iloc[-1]
    if float(prior_row["value"]) <= 0:
        return None
    return float(latest["value"]) / float(prior_row["value"]) - 1.0
