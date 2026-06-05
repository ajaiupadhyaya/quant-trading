"""Macro / business-cycle nowcast (roadmap track C).

A deterministic, fail-open read of the macro-economic backdrop built from free
FRED series — the broad-economy companion to the per-day market signals and the
event-risk calendar. It blends the recession/cycle toolkit a discretionary macro
desk watches:

  * Yield curve   — 10y-3m (the Fed's recession-model curve) + 10y-2y. Inversion
                    leads recessions by ~12-18 months.
  * Credit        — BAA-AAA default-risk spread + ICE BofA HY OAS. Widening = the
                    bond market pricing stress before equities do.
  * Conditions    — Chicago Fed NFCI (>0 = tighter than average).
  * Labour        — initial jobless claims vs their trailing-year low.
  * Inflation     — 10-year breakeven.
  * Recession     — the Sahm Rule real-time indicator (≥0.5 = recession onset).

These collapse into a single ``recession_risk`` in [0,1] (coverage-normalised,
like the signals composite), a low/elevated/high label, a coincident
``recession_signal`` (Sahm), and an expansion/late-cycle/contraction cycle phase.

HONEST FRAMING: leading indicators (curve) and coincident ones (Sahm, claims)
have very different lead times; ``recession_risk`` is a blended "how stressed /
late-cycle is the backdrop" gauge, not a calibrated recession probability. It is
advisory only — it feeds MarketState + the Claude analyst and actuates nothing.
All values are in FRED-native units (percent for rates/spreads, level for claims).
"""

from __future__ import annotations

import concurrent.futures
import math
from dataclasses import dataclass
from datetime import date
from typing import Any

from quant.util.logging import logger


@dataclass(frozen=True)
class MacroNowcastConfig:
    """Thresholds + label anchors. Advisory; tuned to post-1990 norms."""

    inversion_pct: float = 0.0  # 10y-3m below this (percent) = inverted
    hy_oas_stress: float = 5.0  # HY OAS (percent) at/above this = credit stress
    hy_oas_high: float = 8.0  # acute stress
    nfci_high: float = 1.0  # NFCI scale for the conditions sub-score
    claims_elevated: float = 0.20  # claims this far above the trailing-year low = labour weakening
    sahm_trigger: float = 0.50  # Sahm Rule recession threshold
    risk_high: float = 0.60  # recession_risk label anchors
    risk_elevated: float = 0.30
    min_components: int = 2  # need this many inputs before labelling


@dataclass(frozen=True)
class MacroNowcast:
    """The macro backdrop for one ``asof`` date. Any field may be None."""

    asof: str  # ISO
    term_spread_10y3m: float | None
    term_spread_10y2y: float | None
    credit_spread_baa_aaa: float | None
    hy_oas: float | None
    financial_conditions: float | None  # NFCI
    initial_claims: float | None
    claims_vs_year_low: float | None  # claims / trailing-52w low - 1
    breakeven_10y: float | None
    sahm: float | None
    recession_signal: bool  # Sahm >= trigger (coincident)
    recession_risk: float | None  # blended leading+coincident composite [0,1]
    recession_risk_label: str | None  # "low" | "elevated" | "high"
    cycle_label: str | None  # "expansion" | "late-cycle" | "contraction"
    n_components: int  # how many sub-scores fed the composite


def _finite(x: Any) -> float | None:
    try:
        v = float(x) if x is not None else None
    except (TypeError, ValueError):
        return None
    return v if (v is not None and math.isfinite(v)) else None


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def compute_macro_nowcast(
    asof: date,
    *,
    t10y3m: float | None = None,
    t10y2y: float | None = None,
    baa: float | None = None,
    aaa: float | None = None,
    hy_oas: float | None = None,
    nfci: float | None = None,
    claims: float | None = None,
    claims_year_low: float | None = None,
    breakeven10: float | None = None,
    sahm: float | None = None,
    config: MacroNowcastConfig | None = None,
) -> MacroNowcast:
    """Pure: blend macro inputs into one recession/cycle read. No I/O."""
    cfg = config or MacroNowcastConfig()
    t10y3m = _finite(t10y3m)
    t10y2y = _finite(t10y2y)
    baa = _finite(baa)
    aaa = _finite(aaa)
    hy_oas = _finite(hy_oas)
    nfci = _finite(nfci)
    claims = _finite(claims)
    claims_year_low = _finite(claims_year_low)
    breakeven10 = _finite(breakeven10)
    sahm = _finite(sahm)

    credit_spread = (baa - aaa) if (baa is not None and aaa is not None) else None
    claims_vs_low = (
        (claims / claims_year_low - 1.0)
        if (claims is not None and claims_year_low is not None and claims_year_low > 0)
        else None
    )

    # Each sub-score is in [0,1]; missing inputs simply drop out of the mean.
    sub: list[float] = []
    if t10y3m is not None:
        sub.append(_clip01((cfg.inversion_pct - t10y3m) / 1.0))  # inverted 100bps → 1.0
    if hy_oas is not None:
        sub.append(_clip01((hy_oas - cfg.hy_oas_stress) / (cfg.hy_oas_high - cfg.hy_oas_stress)))
    if nfci is not None:
        sub.append(_clip01(nfci / cfg.nfci_high))
    if claims_vs_low is not None:
        sub.append(_clip01(claims_vs_low / cfg.claims_elevated))
    if sahm is not None:
        sub.append(_clip01(sahm / cfg.sahm_trigger))

    n_components = len(sub)
    recession_risk = (sum(sub) / n_components) if n_components else None

    recession_signal = sahm is not None and sahm >= cfg.sahm_trigger

    recession_risk_label: str | None = None
    if recession_risk is not None and n_components >= cfg.min_components:
        recession_risk_label = (
            "high"
            if recession_risk >= cfg.risk_high
            else ("elevated" if recession_risk >= cfg.risk_elevated else "low")
        )

    cycle_label: str | None = None
    if recession_signal:
        cycle_label = "contraction"
    elif recession_risk is not None and n_components >= cfg.min_components:
        late = (
            recession_risk >= cfg.risk_elevated
            or (t10y3m is not None and t10y3m < cfg.inversion_pct)
            or (hy_oas is not None and hy_oas >= cfg.hy_oas_stress)
        )
        cycle_label = "late-cycle" if late else "expansion"

    return MacroNowcast(
        asof=asof.isoformat(),
        term_spread_10y3m=t10y3m,
        term_spread_10y2y=t10y2y,
        credit_spread_baa_aaa=credit_spread,
        hy_oas=hy_oas,
        financial_conditions=nfci,
        initial_claims=claims,
        claims_vs_year_low=claims_vs_low,
        breakeven_10y=breakeven10,
        sahm=sahm,
        recession_signal=recession_signal,
        recession_risk=recession_risk,
        recession_risk_label=recession_risk_label,
        cycle_label=cycle_label,
        n_components=n_components,
    )


def _with_timeout(fn: Any, seconds: float) -> Any:
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(fn).result(timeout=seconds)


def live_macro_nowcast(
    settings: Any, asof: date, *, config: MacroNowcastConfig | None = None
) -> MacroNowcast:
    """Bounded, fail-open FRED reads → nowcast. Never raises."""

    def _series(key: str) -> Any | None:
        try:
            from quant.data import macro

            code = macro.FRED_SERIES.get(key, key)
            s = _with_timeout(lambda: macro.get_series(code), 8.0)
            return s.dropna() if s is not None else None
        except Exception as exc:  # one series failing must not sink the read
            logger.info("macro.nowcast: series {} skipped ({!r})", key, exc)
            return None

    def _last(key: str) -> float | None:
        s = _series(key)
        return _finite(s.iloc[-1]) if s is not None and len(s) else None

    claims_series = _series("claims")
    claims_last: float | None = None
    claims_year_low: float | None = None
    if claims_series is not None and len(claims_series):
        claims_last = _finite(claims_series.iloc[-1])
        claims_year_low = _finite(claims_series.iloc[-52:].min())  # ~trailing year (weekly)

    return compute_macro_nowcast(
        asof,
        t10y3m=_last("term_10y3m"),
        t10y2y=_last("term_10y2y"),
        baa=_last("baa"),
        aaa=_last("aaa"),
        hy_oas=_last("hy_oas"),
        nfci=_last("nfci"),
        claims=claims_last,
        claims_year_low=claims_year_low,
        breakeven10=_last("breakeven10"),
        sahm=_last("sahm"),
        config=config,
    )


def render_macro_nowcast(n: MacroNowcast | None) -> str:
    """Terse one-liner for the Claude prompt + CLI + logs."""
    if n is None:
        return "Macro nowcast: unavailable"
    bits: list[str] = []
    if n.cycle_label:
        bits.append(f"cycle={n.cycle_label}")
    if n.recession_risk is not None:
        lbl = f"={n.recession_risk_label}" if n.recession_risk_label else ""
        bits.append(f"recession_risk{lbl}({n.recession_risk:.2f})")
    if n.term_spread_10y3m is not None:
        bits.append(f"10y3m={n.term_spread_10y3m:+.2f}")
    if n.hy_oas is not None:
        bits.append(f"HY_OAS={n.hy_oas:.1f}%")
    if n.credit_spread_baa_aaa is not None:
        bits.append(f"BAA-AAA={n.credit_spread_baa_aaa:.2f}")
    if n.claims_vs_year_low is not None:
        bits.append(f"claims={n.claims_vs_year_low:+.0%}>yr-low")
    if n.sahm is not None:
        bits.append(f"sahm={n.sahm:.2f}")
    return "Macro nowcast: " + (", ".join(bits) if bits else "n/a")
