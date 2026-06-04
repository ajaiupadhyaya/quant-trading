"""Live equity-index volatility surface read (roadmap track F).

Alpaca returns SPY option *quotes* but no IV/greeks on this tier, so we recover
implied vol ourselves from real option mids with the BSM solver in ``pricing.py``
— turning the existing options toolkit into a live vol-surface signal. We
summarise the surface into the three dimensions a vol desk watches:

  * LEVEL — ATM implied vol at ~30 days (a real-chain VIX analogue).
  * TERM  — ATM IV(~90d) - ATM IV(~30d): contango (calm) vs backwardation (stress).
  * SKEW  — IV(95% put) - IV(105% call) at ~30d: the equity crash premium. Its
            STEEPNESS is a tail-risk gauge (puts bid relative to calls).

These feed a vol regime, a term-structure label, and a tail label. Everything is
advisory: it enriches MarketState + the Claude analyst and actuates nothing. The
read is bounded + fail-open — a bad chain/quote degrades a field to ``None``, it
never raises. Distinct from the signals engine's realised-vol/VIX block (this is
forward-looking implied vol from real options).
"""

from __future__ import annotations

import concurrent.futures
import math
import re
from dataclasses import dataclass
from datetime import date
from typing import Any

from quant.options.pricing import implied_vol
from quant.util.logging import logger

# OCC symbol: root (1-6 alpha) + YYMMDD + C/P + strike (8 digits, thousandths).
_OCC_RE = re.compile(r"^([A-Z]+)(\d{6})([CP])(\d{8})\Z")


@dataclass(frozen=True)
class OptionQuote:
    """One contract's parsed quote (mid of bid/ask)."""

    expiry: date
    strike: float
    right: str  # "call" | "put"
    mid: float


@dataclass(frozen=True)
class VolSurfaceConfig:
    """Pricing inputs + label anchors. Advisory; tuned to SPY norms."""

    underlying: str = "SPY"
    r: float = 0.045  # risk-free (≈3m T-bill)
    q: float = 0.013  # SPY dividend yield
    near_dte: int = 30  # target near tenor (days)
    far_dte: int = 90  # target far tenor
    min_dte: int = 5
    max_dte: int = 130
    moneyness_band: float = 0.10  # fetch strikes within ±this of spot
    skew_put_money: float = 0.95  # OTM put strike for the skew leg
    skew_call_money: float = 1.05  # OTM call strike for the skew leg
    # vol-regime anchors on ATM 30d IV
    iv_calm: float = 0.12
    iv_elevated: float = 0.18
    iv_stressed: float = 0.28
    # term-structure anchors on (far - near) IV
    term_flat_band: float = 0.005
    # tail anchors on the 95/105 put skew (vol points)
    skew_elevated: float = 0.05
    skew_extreme: float = 0.09


@dataclass(frozen=True)
class VolSurface:
    """The summarised vol surface for one ``asof`` date. Any field may be None."""

    asof: str  # ISO
    spot: float | None
    near_dte: int | None
    far_dte: int | None
    atm_iv_30d: float | None  # ATM IV at the ~near tenor
    atm_iv_90d: float | None  # ATM IV at the ~far tenor
    term_slope: float | None  # atm_iv_90d - atm_iv_30d
    put_skew: float | None  # IV(95% put) - IV(105% call) at the near tenor
    iv_regime: str | None  # "calm" | "normal" | "elevated" | "stressed"
    term_label: str | None  # "contango" | "flat" | "backwardation"
    tail_label: str | None  # "benign" | "elevated" | "extreme"
    n_quotes: int
    n_expiries: int


def parse_occ_symbol(symbol: str) -> tuple[str, date, str, float] | None:
    """Parse an OCC option symbol → (underlying, expiry, right, strike) or None."""
    m = _OCC_RE.match(symbol.strip().upper())
    if m is None:
        return None
    try:
        yy, mm, dd = int(m.group(2)[:2]), int(m.group(2)[2:4]), int(m.group(2)[4:6])
        expiry = date(2000 + yy, mm, dd)
        right = "call" if m.group(3) == "C" else "put"
        strike = int(m.group(4)) / 1000.0
    except (ValueError, TypeError):
        return None
    return m.group(1), expiry, right, strike


def _finite(x: Any) -> float | None:
    try:
        v = float(x) if x is not None else None
    except (TypeError, ValueError):
        return None
    return v if (v is not None and math.isfinite(v)) else None


def _nearest(quotes: list[OptionQuote], target_strike: float) -> OptionQuote | None:
    return min(quotes, key=lambda x: abs(x.strike - target_strike)) if quotes else None


def _iv(q: OptionQuote, spot: float, asof: date, cfg: VolSurfaceConfig) -> float | None:
    t = (q.expiry - asof).days / 365.0
    return _finite(implied_vol(q.mid, spot, q.strike, t, cfg.r, cfg.q, q.right))


def compute_vol_surface(
    quotes: list[OptionQuote],
    spot: float | None,
    asof: date,
    *,
    config: VolSurfaceConfig | None = None,
) -> VolSurface:
    """Pure: parsed quotes + spot → the summarised surface. Solves IV via BSM.

    Picks the expiry nearest the near/far DTE targets, computes ATM IV at each,
    the term slope, and a 95/105 put-skew at the near tenor.
    """
    cfg = config or VolSurfaceConfig()
    spot = _finite(spot)

    def _empty(n_q: int, n_e: int) -> VolSurface:
        return VolSurface(
            asof.isoformat(), spot, None, None, None, None, None, None, None, None, None, n_q, n_e
        )

    usable = [
        q
        for q in quotes
        if cfg.min_dte <= (q.expiry - asof).days <= cfg.max_dte and q.mid > 0 and q.strike > 0
    ]
    expiries = sorted({q.expiry for q in usable})
    if spot is None or spot <= 0 or not expiries:
        return _empty(len(usable), len(expiries))

    near = min(expiries, key=lambda e: abs((e - asof).days - cfg.near_dte))
    far = min(expiries, key=lambda e: abs((e - asof).days - cfg.far_dte))

    def _atm_iv(expiry: date) -> float | None:
        calls = [q for q in usable if q.expiry == expiry and q.right == "call"]
        atm = _nearest(calls, spot) or _nearest([q for q in usable if q.expiry == expiry], spot)
        return _iv(atm, spot, asof, cfg) if atm is not None else None

    atm_iv_30d = _atm_iv(near)
    atm_iv_90d = _atm_iv(far) if far != near else None
    term_slope = (
        (atm_iv_90d - atm_iv_30d) if (atm_iv_30d is not None and atm_iv_90d is not None) else None
    )

    # 95/105 put skew at the near tenor.
    near_puts = [q for q in usable if q.expiry == near and q.right == "put"]
    near_calls = [q for q in usable if q.expiry == near and q.right == "call"]
    put_leg = _nearest(near_puts, spot * cfg.skew_put_money)
    call_leg = _nearest(near_calls, spot * cfg.skew_call_money)
    iv_put = _iv(put_leg, spot, asof, cfg) if put_leg is not None else None
    iv_call = _iv(call_leg, spot, asof, cfg) if call_leg is not None else None
    put_skew = (iv_put - iv_call) if (iv_put is not None and iv_call is not None) else None

    iv_regime: str | None = None
    if atm_iv_30d is not None:
        iv_regime = (
            "calm"
            if atm_iv_30d < cfg.iv_calm
            else "normal"
            if atm_iv_30d < cfg.iv_elevated
            else "elevated"
            if atm_iv_30d < cfg.iv_stressed
            else "stressed"
        )

    term_label: str | None = None
    if term_slope is not None:
        term_label = (
            "backwardation"
            if term_slope < -cfg.term_flat_band
            else "contango"
            if term_slope > cfg.term_flat_band
            else "flat"
        )

    tail_label: str | None = None
    if put_skew is not None:
        tail_label = (
            "extreme"
            if put_skew >= cfg.skew_extreme
            else "elevated"
            if put_skew >= cfg.skew_elevated
            else "benign"
        )

    return VolSurface(
        asof=asof.isoformat(),
        spot=spot,
        near_dte=(near - asof).days,
        far_dte=(far - asof).days if far != near else None,
        atm_iv_30d=atm_iv_30d,
        atm_iv_90d=atm_iv_90d,
        term_slope=term_slope,
        put_skew=put_skew,
        iv_regime=iv_regime,
        term_label=term_label,
        tail_label=tail_label,
        n_quotes=len(usable),
        n_expiries=len(expiries),
    )


def _with_timeout(fn: Any, seconds: float) -> Any:
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(fn).result(timeout=seconds)


def _spot_from_cache(settings: Any, symbol: str) -> float | None:
    """Latest cached close for the underlying (bounded, fail-open)."""
    try:
        import pandas as pd

        from quant.data import bars

        path = bars._cache_path(symbol, getattr(settings, "data_dir", None))
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        s = df["close"].dropna() if "close" in df.columns else None
        return _finite(s.iloc[-1]) if s is not None and len(s) else None
    except Exception as exc:  # fail-open
        logger.info("options.surface: spot load skipped ({!r})", exc)
        return None


def _fetch_quotes(
    settings: Any, asof: date, spot: float, cfg: VolSurfaceConfig
) -> list[OptionQuote]:
    """Fetch a near-the-money slice of the chain and parse to OptionQuotes."""
    from datetime import timedelta

    from alpaca.data.historical.option import OptionHistoricalDataClient
    from alpaca.data.requests import OptionChainRequest

    client = OptionHistoricalDataClient(
        api_key=getattr(settings, "alpaca_api_key", None),
        secret_key=getattr(settings, "alpaca_secret_key", None),
    )
    req = OptionChainRequest(
        underlying_symbol=cfg.underlying,
        strike_price_gte=spot * (1.0 - cfg.moneyness_band),
        strike_price_lte=spot * (1.0 + cfg.moneyness_band),
        expiration_date_lte=(asof + timedelta(days=cfg.max_dte)),
    )
    chain = _with_timeout(lambda: client.get_option_chain(req), 20.0)
    out: list[OptionQuote] = []
    for sym, snap in (chain or {}).items():
        parsed = parse_occ_symbol(sym)
        if parsed is None:
            continue
        _, expiry, right, strike = parsed
        quote = getattr(snap, "latest_quote", None)
        bid = _finite(getattr(quote, "bid_price", None))
        ask = _finite(getattr(quote, "ask_price", None))
        if bid is None or ask is None or bid <= 0 or ask <= 0 or ask < bid:
            continue
        out.append(OptionQuote(expiry=expiry, strike=strike, right=right, mid=(bid + ask) / 2.0))
    return out


def live_vol_surface(
    settings: Any, asof: date, *, config: VolSurfaceConfig | None = None
) -> VolSurface:
    """Bounded, fail-open vol-surface read from the live SPY chain. Never raises."""
    cfg = config or VolSurfaceConfig()
    spot = _spot_from_cache(settings, cfg.underlying)
    quotes: list[OptionQuote] = []
    if spot is not None and spot > 0:
        try:
            quotes = _with_timeout(lambda: _fetch_quotes(settings, asof, spot, cfg), 25.0)
        except Exception as exc:  # fail-open: no chain → empty read
            logger.info("options.surface: chain fetch skipped ({!r})", exc)
            quotes = []
    return compute_vol_surface(quotes, spot, asof, config=cfg)


def render_vol_surface(v: VolSurface | None) -> str:
    """Terse one-liner for the Claude prompt + CLI + logs."""
    if v is None:
        return "Vol surface: unavailable"
    if v.atm_iv_30d is None:
        return "Vol surface: no coverage"
    bits: list[str] = []
    if v.iv_regime:
        bits.append(f"iv_regime={v.iv_regime}")
    bits.append(f"ATM_IV30={v.atm_iv_30d:.1%}")
    if v.term_slope is not None and v.term_label:
        bits.append(f"term={v.term_label}({v.term_slope:+.1%})")
    if v.put_skew is not None and v.tail_label:
        bits.append(f"tail={v.tail_label}(skew {v.put_skew:+.1%})")
    return "Vol surface: " + ", ".join(bits)
