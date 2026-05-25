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
}


@dataclass(frozen=True)
class FactRow:
    """One point-in-time XBRL fact."""

    concept: str
    value: float
    period_end: date
    filed: date
    unit: str


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
    return data_dir / "fundamentals" / f"{safe}_facts.parquet"


def _extract_concept(facts: dict[str, Any], concept: str) -> list[FactRow]:
    """Pull one concept out of the /companyfacts payload. Returns sorted rows."""
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    tags = _CONCEPT_TAGS.get(concept, (concept,))
    for tag in tags:
        if tag not in us_gaap:
            continue
        unit_map = us_gaap[tag].get("units", {})
        # Prefer USD; fall back to USD/shares for per-share concepts (unused here).
        for unit_name in ("USD", "USD/shares"):
            rows = unit_map.get(unit_name, [])
            if not rows:
                continue
            out: list[FactRow] = []
            for r in rows:
                try:
                    out.append(
                        FactRow(
                            concept=concept,
                            value=float(r["val"]),
                            period_end=date.fromisoformat(str(r["end"])[:10]),
                            filed=date.fromisoformat(str(r["filed"])[:10]),
                            unit=unit_name,
                        )
                    )
                except Exception:  # malformed row; skip
                    continue
            out.sort(key=lambda f: (f.filed, f.period_end))
            return out
        # Otherwise try the next tag
    return []


def fetch_company_facts(ticker: str, data_dir: Path | None = None) -> pd.DataFrame:
    """Fetch + parse /companyfacts for ``ticker`` into a long-format DataFrame.

    Columns: ``concept`` ``value`` ``period_end`` ``filed`` ``unit``. Cached as
    parquet per ticker. Subsequent calls hit cache.
    """
    data_dir = data_dir or _settings().data_dir
    path = _facts_path(ticker, data_dir)
    if path.exists():
        return pd.read_parquet(path)

    cik = cik_for_ticker(ticker, data_dir)
    if cik is None:
        logger.warning("No EDGAR CIK for ticker {}", ticker)
        empty = pd.DataFrame(columns=["concept", "value", "period_end", "filed", "unit"])
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
