# Options/Greeks Engine + Protective Hedging Overlay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an analytic Black-Scholes-Merton options/Greeks core and a point-in-time, observed-only protective hedging overlay (SPY beta hedge: protective put / collar / put-spread) that honestly compares baseline-vs-hedged book returns.

**Architecture:** New `quant/options/` package mirroring `quant/sizing/`: pure pricing functions → models → structure builders → PIT beta → composing policy → returns-overlay backtest → one `quant hedge` CLI group with registry logging. No new dependencies (scipy/numpy/pandas already present). Observed-only — never touches live allocation.

**Tech Stack:** Python 3.12, numpy, scipy (`norm`, `brentq`), pandas, Click, Rich, pytest, Hypothesis, mypy strict, ruff.

---

## File Structure

- Create `quant/options/__init__.py` — public API exports
- Create `quant/options/pricing.py` — `bs_price`, `bs_greeks`, `implied_vol`, `Greeks`
- Create `quant/options/models.py` — `OptionLeg`, `HedgeStructure`, `HedgeConfig`, `HedgeDecision`
- Create `quant/options/structures.py` — `protective_put`, `collar`, `put_spread`, `build_structure`
- Create `quant/options/beta.py` — `rolling_beta`
- Create `quant/options/policy.py` — `build_hedge`
- Create `quant/options/overlay.py` — `cvar`, `worst_day`, `apply_hedge`, `compare_hedge`, `HedgeComparison`, `HedgeLedger`
- Create tests under `tests/options/`
- Modify `quant/cli.py` — add `quant hedge` group (`price`, `compare`)
- Modify `README.md` — document the pillar

---

## Task 1: Pricing core — `pricing.py`

**Files:**
- Create: `quant/options/__init__.py` (empty for now)
- Create: `quant/options/pricing.py`
- Test: `tests/options/__init__.py`, `tests/options/test_pricing.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/options/test_pricing.py
import math

import numpy as np
import pytest

from quant.options.pricing import Greeks, bs_greeks, bs_price, implied_vol

S, K, T, VOL, R, Q = 100.0, 100.0, 1.0, 0.20, 0.03, 0.0


def test_atm_call_known_value():
    # Textbook ATM call, S=K=100, vol=20%, r=3%, q=0, T=1y ≈ 9.413
    price = bs_price(S, K, T, VOL, R, Q, "call")
    assert price == pytest.approx(9.4134, abs=1e-3)


def test_put_call_parity():
    c = bs_price(S, K, T, VOL, R, Q, "call")
    p = bs_price(S, K, T, VOL, R, Q, "put")
    lhs = c - p
    rhs = S * math.exp(-Q * T) - K * math.exp(-R * T)
    assert lhs == pytest.approx(rhs, abs=1e-9)


def test_delta_matches_finite_difference():
    h = 1e-4
    up = bs_price(S + h, K, T, VOL, R, Q, "call")
    dn = bs_price(S - h, K, T, VOL, R, Q, "call")
    fd_delta = (up - dn) / (2 * h)
    assert bs_greeks(S, K, T, VOL, R, Q, "call").delta == pytest.approx(fd_delta, abs=1e-5)


def test_gamma_matches_finite_difference():
    h = 1e-2
    up = bs_price(S + h, K, T, VOL, R, Q, "call")
    mid = bs_price(S, K, T, VOL, R, Q, "call")
    dn = bs_price(S - h, K, T, VOL, R, Q, "call")
    fd_gamma = (up - 2 * mid + dn) / (h * h)
    assert bs_greeks(S, K, T, VOL, R, Q, "call").gamma == pytest.approx(fd_gamma, abs=1e-4)


def test_vega_matches_finite_difference():
    h = 1e-4
    up = bs_price(S, K, T, VOL + h, R, Q, "call")
    dn = bs_price(S, K, T, VOL - h, R, Q, "call")
    fd_vega = (up - dn) / (2 * h)
    assert bs_greeks(S, K, T, VOL, R, Q, "call").vega == pytest.approx(fd_vega, abs=1e-3)


def test_put_delta_negative():
    assert bs_greeks(S, K, T, VOL, R, Q, "put").delta < 0.0


def test_implied_vol_round_trip():
    price = bs_price(S, 95.0, T, 0.25, R, Q, "put")
    iv = implied_vol(price, S, 95.0, T, R, Q, "put")
    assert iv == pytest.approx(0.25, abs=1e-4)


def test_at_expiry_returns_intrinsic():
    assert bs_price(110.0, 100.0, 0.0, VOL, R, Q, "call") == pytest.approx(10.0)
    assert bs_price(90.0, 100.0, 0.0, VOL, R, Q, "call") == pytest.approx(0.0)
    assert bs_price(90.0, 100.0, 0.0, VOL, R, Q, "put") == pytest.approx(10.0)


def test_nonfinite_inputs_return_nan():
    assert math.isnan(bs_price(float("nan"), K, T, VOL, R, Q, "call"))
    assert math.isnan(bs_price(S, K, T, -0.1, R, Q, "call"))


def test_greeks_is_frozen_dataclass():
    g = bs_greeks(S, K, T, VOL, R, Q, "call")
    assert isinstance(g, Greeks)
    with pytest.raises(Exception):
        g.delta = 0.0  # type: ignore[misc]


def test_implied_vol_unreachable_price_is_nan():
    # Price above no-arb upper bound for a call (>= S) -> nan
    assert math.isnan(implied_vol(200.0, S, K, T, R, Q, "call"))
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `uv run pytest tests/options/test_pricing.py -q`
Expected: FAIL (module `quant.options.pricing` not found).

- [ ] **Step 3: Implement `pricing.py`**

```python
# quant/options/pricing.py
"""Black-Scholes-Merton pricing + Greeks + implied vol.

Pure functions, no I/O, no state. Continuous dividend yield ``q``. Every
function degrades to ``nan`` on non-finite / non-positive inputs (callers
guard) and never raises on finite inputs — matching quant/sizing/components.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.optimize import brentq  # type: ignore[import-untyped]
from scipy.stats import norm  # type: ignore[import-untyped]


@dataclass(frozen=True)
class Greeks:
    """First-order Greeks (theta/rho per year, vega per 1.00 vol point)."""

    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


def _valid(spot: float, strike: float, vol: float, t_years: float) -> bool:
    return (
        math.isfinite(spot)
        and math.isfinite(strike)
        and math.isfinite(vol)
        and math.isfinite(t_years)
        and spot > 0.0
        and strike > 0.0
        and vol > 0.0
    )


def _d1_d2(spot: float, strike: float, t_years: float, vol: float, r: float, q: float):
    sqrt_t = math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (r - q + 0.5 * vol * vol) * t_years) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    return d1, d2


def bs_price(
    spot: float, strike: float, t_years: float, vol: float, r: float, q: float, right: str
) -> float:
    """Black-Scholes-Merton price. ``right`` in {"call","put"}."""
    if t_years <= 0.0:
        if right == "call":
            return max(0.0, spot - strike)
        return max(0.0, strike - spot)
    if not _valid(spot, strike, vol, t_years):
        return float("nan")
    d1, d2 = _d1_d2(spot, strike, t_years, vol, r, q)
    disc_q = math.exp(-q * t_years)
    disc_r = math.exp(-r * t_years)
    if right == "call":
        return spot * disc_q * norm.cdf(d1) - strike * disc_r * norm.cdf(d2)
    return strike * disc_r * norm.cdf(-d2) - spot * disc_q * norm.cdf(-d1)


def bs_greeks(
    spot: float, strike: float, t_years: float, vol: float, r: float, q: float, right: str
) -> Greeks:
    """Delta, gamma, vega, theta (per year), rho. nan-filled on bad input."""
    if not _valid(spot, strike, vol, t_years) or t_years <= 0.0:
        nan = float("nan")
        return Greeks(nan, nan, nan, nan, nan)
    d1, d2 = _d1_d2(spot, strike, t_years, vol, r, q)
    sqrt_t = math.sqrt(t_years)
    disc_q = math.exp(-q * t_years)
    disc_r = math.exp(-r * t_years)
    pdf_d1 = norm.pdf(d1)
    gamma = disc_q * pdf_d1 / (spot * vol * sqrt_t)
    vega = spot * disc_q * pdf_d1 * sqrt_t
    if right == "call":
        delta = disc_q * norm.cdf(d1)
        theta = (
            -spot * disc_q * pdf_d1 * vol / (2 * sqrt_t)
            - r * strike * disc_r * norm.cdf(d2)
            + q * spot * disc_q * norm.cdf(d1)
        )
        rho = strike * t_years * disc_r * norm.cdf(d2)
    else:
        delta = -disc_q * norm.cdf(-d1)
        theta = (
            -spot * disc_q * pdf_d1 * vol / (2 * sqrt_t)
            + r * strike * disc_r * norm.cdf(-d2)
            - q * spot * disc_q * norm.cdf(-d1)
        )
        rho = -strike * t_years * disc_r * norm.cdf(-d2)
    return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)


def implied_vol(
    price: float, spot: float, strike: float, t_years: float, r: float, q: float, right: str
) -> float:
    """Brent-solve implied vol on [1e-4, 5.0]; nan if price is unreachable."""
    if not (math.isfinite(price) and price > 0.0) or t_years <= 0.0 or spot <= 0.0:
        return float("nan")

    def objective(vol: float) -> float:
        return bs_price(spot, strike, t_years, vol, r, q, right) - price

    try:
        lo, hi = objective(1e-4), objective(5.0)
        if lo * hi > 0.0:  # not bracketed -> price outside model range
            return float("nan")
        return float(brentq(objective, 1e-4, 5.0, maxiter=100, xtol=1e-8))
    except (ValueError, RuntimeError):
        return float("nan")
```

Also create empty `quant/options/__init__.py` and `tests/options/__init__.py`.

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/options/test_pricing.py -q`
Expected: PASS (all 11).

- [ ] **Step 5: Commit**

```bash
git add quant/options/__init__.py quant/options/pricing.py tests/options/
git commit -m "feat(options): Black-Scholes-Merton pricing + Greeks + implied vol"
```

---

## Task 2: Models — `models.py`

**Files:**
- Create: `quant/options/models.py`
- Test: `tests/options/test_models.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/options/test_models.py
import pytest

from quant.options.models import HedgeConfig, HedgeDecision, HedgeStructure, OptionLeg


def test_structure_value_sums_signed_legs():
    legs = (OptionLeg("put", 95.0, 1.0), OptionLeg("call", 105.0, -1.0))
    struct = HedgeStructure(legs=legs, spot_at_open=100.0, expiry_index=21)
    # long put minus short call; just assert it is finite and matches manual sum
    from quant.options.pricing import bs_price

    expected = bs_price(100.0, 95.0, 0.08, 0.2, 0.03, 0.0, "put") - bs_price(
        100.0, 105.0, 0.08, 0.2, 0.03, 0.0, "call"
    )
    assert struct.value(100.0, 0.08, 0.2, 0.03, 0.0) == pytest.approx(expected)


def test_config_defaults():
    cfg = HedgeConfig()
    assert cfg.structure == "put"
    assert cfg.put_moneyness == 0.05
    assert cfg.coverage == 1.0
    assert cfg.tenor_days == 30
    assert cfg.roll_days == 21
    assert cfg.use_regime is True
    assert cfg.regime_intensity["crisis"] == 1.0


def test_config_is_frozen():
    cfg = HedgeConfig()
    with pytest.raises(Exception):
        cfg.coverage = 2.0  # type: ignore[misc]


def test_decision_record_fields():
    legs = (OptionLeg("put", 95.0, 1.0),)
    struct = HedgeStructure(legs=legs, spot_at_open=100.0, expiry_index=21)
    dec = HedgeDecision(
        structure=struct,
        contracts=2.0,
        premium=3.5,
        net_beta=0.9,
        regime_label="choppy",
        intensity=0.6,
    )
    assert dec.contracts == 2.0
    assert dec.regime_label == "choppy"
```

- [ ] **Step 2: Run, verify fail.** `uv run pytest tests/options/test_models.py -q` → FAIL.

- [ ] **Step 3: Implement `models.py`**

```python
# quant/options/models.py
"""Dataclasses for the hedging overlay: legs, structures, config, decisions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from quant.options.pricing import bs_price

DEFAULT_REGIME_INTENSITY: dict[str, float] = {"calm-bull": 0.25, "choppy": 0.6, "crisis": 1.0}


def _default_regime_intensity() -> Mapping[str, float]:
    return MappingProxyType(dict(DEFAULT_REGIME_INTENSITY))


@dataclass(frozen=True)
class OptionLeg:
    """One leg: right in {"call","put"}, strike, signed quantity (+long/-short)."""

    right: str
    strike: float
    quantity: float


@dataclass(frozen=True)
class HedgeStructure:
    """A built multi-leg structure struck against ``spot_at_open``."""

    legs: tuple[OptionLeg, ...]
    spot_at_open: float
    expiry_index: int

    def value(self, spot: float, t_years: float, vol: float, r: float, q: float) -> float:
        """Sum of signed-leg Black-Scholes values at the given market state."""
        total = 0.0
        for leg in self.legs:
            total += leg.quantity * bs_price(spot, leg.strike, t_years, vol, r, q, leg.right)
        return total


@dataclass(frozen=True)
class HedgeConfig:
    """Knobs for the hedging overlay. All defaults intentional."""

    structure: str = "put"  # "put" | "collar" | "put_spread"
    put_moneyness: float = 0.05
    call_moneyness: float = 0.05
    spread_width: float = 0.10
    coverage: float = 1.0
    tenor_days: int = 30
    roll_days: int = 21
    vol_lookback_days: int = 21
    risk_free: float = 0.03
    div_yield: float = 0.015
    beta_lookback_days: int = 63
    use_regime: bool = True
    regime_intensity: Mapping[str, float] = field(default_factory=_default_regime_intensity)


@dataclass(frozen=True)
class HedgeDecision:
    """A single roll's record for introspection/serialization."""

    structure: HedgeStructure
    contracts: float
    premium: float
    net_beta: float
    regime_label: str | None
    intensity: float
```

- [ ] **Step 4: Run, verify pass.** `uv run pytest tests/options/test_models.py -q` → PASS.

- [ ] **Step 5: Commit**

```bash
git add quant/options/models.py tests/options/test_models.py
git commit -m "feat(options): hedge models (leg, structure, config, decision)"
```

---

## Task 3: Structure builders — `structures.py`

**Files:**
- Create: `quant/options/structures.py`
- Test: `tests/options/test_structures.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/options/test_structures.py
import pytest

from quant.options.models import HedgeConfig
from quant.options.structures import build_structure, collar, protective_put, put_spread

SPOT = 100.0


def test_protective_put_strike_and_sign():
    cfg = HedgeConfig(put_moneyness=0.05)
    s = protective_put(SPOT, cfg)
    assert len(s.legs) == 1
    assert s.legs[0].right == "put"
    assert s.legs[0].strike == pytest.approx(95.0)
    assert s.legs[0].quantity == 1.0
    assert s.spot_at_open == SPOT


def test_collar_legs():
    cfg = HedgeConfig(put_moneyness=0.05, call_moneyness=0.05)
    s = collar(SPOT, cfg)
    rights = sorted((leg.right, leg.quantity) for leg in s.legs)
    assert rights == [("call", -1.0), ("put", 1.0)]
    call = next(leg for leg in s.legs if leg.right == "call")
    assert call.strike == pytest.approx(105.0)


def test_put_spread_legs():
    cfg = HedgeConfig(put_moneyness=0.05, spread_width=0.10)
    s = put_spread(SPOT, cfg)
    longs = [leg for leg in s.legs if leg.quantity > 0]
    shorts = [leg for leg in s.legs if leg.quantity < 0]
    assert longs[0].strike == pytest.approx(95.0)
    assert shorts[0].strike == pytest.approx(85.0)


def test_collar_cheaper_than_bare_put():
    cfg = HedgeConfig()
    t, vol, r, q = 0.08, 0.2, 0.03, 0.015
    bare = protective_put(SPOT, cfg).value(SPOT, t, vol, r, q)
    col = collar(SPOT, cfg).value(SPOT, t, vol, r, q)
    assert col < bare  # short call finances the put


def test_put_spread_cheaper_than_bare_put():
    cfg = HedgeConfig()
    t, vol, r, q = 0.08, 0.2, 0.03, 0.015
    bare = protective_put(SPOT, cfg).value(SPOT, t, vol, r, q)
    spread = put_spread(SPOT, cfg).value(SPOT, t, vol, r, q)
    assert 0.0 < spread < bare


def test_build_structure_dispatch():
    cfg = HedgeConfig(structure="collar")
    s = build_structure(SPOT, cfg)
    assert len(s.legs) == 2


def test_build_structure_unknown_raises():
    cfg = HedgeConfig(structure="butterfly")
    with pytest.raises(ValueError):
        build_structure(SPOT, cfg)
```

- [ ] **Step 2: Run, verify fail.** → FAIL.

- [ ] **Step 3: Implement `structures.py`**

```python
# quant/options/structures.py
"""Pure builders mapping (spot, config) -> a HedgeStructure (quantity = 1 unit)."""

from __future__ import annotations

from quant.options.models import HedgeConfig, HedgeStructure, OptionLeg

# expiry_index is filled by the overlay at roll time; builders use a placeholder.
_PLACEHOLDER_EXPIRY = 0


def protective_put(spot: float, cfg: HedgeConfig) -> HedgeStructure:
    strike = spot * (1.0 - cfg.put_moneyness)
    return HedgeStructure(
        legs=(OptionLeg("put", strike, 1.0),),
        spot_at_open=spot,
        expiry_index=_PLACEHOLDER_EXPIRY,
    )


def collar(spot: float, cfg: HedgeConfig) -> HedgeStructure:
    put_strike = spot * (1.0 - cfg.put_moneyness)
    call_strike = spot * (1.0 + cfg.call_moneyness)
    return HedgeStructure(
        legs=(OptionLeg("put", put_strike, 1.0), OptionLeg("call", call_strike, -1.0)),
        spot_at_open=spot,
        expiry_index=_PLACEHOLDER_EXPIRY,
    )


def put_spread(spot: float, cfg: HedgeConfig) -> HedgeStructure:
    high_strike = spot * (1.0 - cfg.put_moneyness)
    low_strike = spot * (1.0 - cfg.put_moneyness - cfg.spread_width)
    return HedgeStructure(
        legs=(OptionLeg("put", high_strike, 1.0), OptionLeg("put", low_strike, -1.0)),
        spot_at_open=spot,
        expiry_index=_PLACEHOLDER_EXPIRY,
    )


_BUILDERS = {"put": protective_put, "collar": collar, "put_spread": put_spread}


def build_structure(spot: float, cfg: HedgeConfig) -> HedgeStructure:
    """Dispatch on cfg.structure. Raises ValueError on unknown structure name."""
    try:
        builder = _BUILDERS[cfg.structure]
    except KeyError as exc:
        raise ValueError(f"unknown hedge structure: {cfg.structure!r}") from exc
    return builder(spot, cfg)
```

- [ ] **Step 4: Run, verify pass.** → PASS.

- [ ] **Step 5: Commit**

```bash
git add quant/options/structures.py tests/options/test_structures.py
git commit -m "feat(options): protective-put / collar / put-spread structure builders"
```

---

## Task 4: PIT beta — `beta.py`

**Files:**
- Create: `quant/options/beta.py`
- Test: `tests/options/test_beta.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/options/test_beta.py
import numpy as np

from quant.options.beta import rolling_beta


def test_recovers_known_slope():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.01, 500)
    y = 0.8 * x + rng.normal(0, 1e-5, 500)
    assert abs(rolling_beta(y, x) - 0.8) < 0.02


def test_degenerate_returns_neutral():
    assert rolling_beta(np.array([]), np.array([])) == 1.0
    assert rolling_beta(np.array([0.01]), np.array([0.01])) == 1.0
    assert rolling_beta(np.array([0.01, 0.02]), np.array([0.0, 0.0])) == 1.0  # zero var


def test_clamped_to_range():
    x = np.array([0.001, -0.001, 0.001, -0.001])
    y = 10.0 * x  # beta 10 -> clamp to 3
    assert rolling_beta(y, x) == 3.0
```

- [ ] **Step 2: Run, verify fail.** → FAIL.

- [ ] **Step 3: Implement `beta.py`**

```python
# quant/options/beta.py
"""Point-in-time net beta of book returns vs SPY returns (OLS slope)."""

from __future__ import annotations

import math

import numpy as np

_NEUTRAL = 1.0
_MIN_BETA = 0.0
_MAX_BETA = 3.0


def rolling_beta(book_returns: np.ndarray, spy_returns: np.ndarray) -> float:
    """OLS slope of book on SPY over the supplied (trailing, PIT) window.

    Returns 1.0 (neutral) on degenerate input; clamps result to [0, 3].
    """
    book = np.asarray(book_returns, dtype=float)
    spy = np.asarray(spy_returns, dtype=float)
    n = min(book.size, spy.size)
    if n < 2:
        return _NEUTRAL
    book = book[-n:]
    spy = spy[-n:]
    mask = np.isfinite(book) & np.isfinite(spy)
    if mask.sum() < 2:
        return _NEUTRAL
    book = book[mask]
    spy = spy[mask]
    var = float(np.var(spy, ddof=1))
    if var <= 0.0 or not math.isfinite(var):
        return _NEUTRAL
    cov = float(np.cov(book, spy, ddof=1)[0, 1])
    beta = cov / var
    if not math.isfinite(beta):
        return _NEUTRAL
    return float(max(_MIN_BETA, min(_MAX_BETA, beta)))
```

- [ ] **Step 4: Run, verify pass.** → PASS.

- [ ] **Step 5: Commit**

```bash
git add quant/options/beta.py tests/options/test_beta.py
git commit -m "feat(options): PIT rolling beta of book vs SPY"
```

---

## Task 5: Policy — `policy.py`

**Files:**
- Create: `quant/options/policy.py`
- Test: `tests/options/test_policy.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/options/test_policy.py
import numpy as np

from quant.options.models import HedgeConfig
from quant.options.policy import build_hedge


def _hist(n=70, seed=0):
    rng = np.random.default_rng(seed)
    spy = rng.normal(0.0003, 0.01, n)
    book = 1.0 * spy + rng.normal(0, 1e-4, n)
    return book, spy


def test_crisis_buys_more_contracts_than_calm():
    book, spy = _hist()
    cfg = HedgeConfig()
    calm = build_hedge(100.0, book, spy, "calm-bull", cfg, 1.0, expiry_index=21)
    crisis = build_hedge(100.0, book, spy, "crisis", cfg, 1.0, expiry_index=21)
    assert crisis.contracts > calm.contracts
    assert crisis.intensity == 1.0
    assert calm.intensity == 0.25


def test_no_regime_full_intensity():
    book, spy = _hist()
    cfg = HedgeConfig(use_regime=False)
    dec = build_hedge(100.0, book, spy, "crisis", cfg, 1.0, expiry_index=21)
    assert dec.intensity == 1.0


def test_unknown_label_neutral_intensity():
    book, spy = _hist()
    cfg = HedgeConfig()
    dec = build_hedge(100.0, book, spy, "mystery", cfg, 1.0, expiry_index=21)
    assert dec.intensity == 1.0


def test_premium_positive_for_put():
    book, spy = _hist()
    cfg = HedgeConfig(structure="put")
    dec = build_hedge(100.0, book, spy, "crisis", cfg, 1.0, expiry_index=21)
    assert dec.premium > 0.0
    assert dec.contracts > 0.0


def test_contracts_scale_with_book_value():
    book, spy = _hist()
    cfg = HedgeConfig()
    small = build_hedge(100.0, book, spy, "choppy", cfg, 1.0, expiry_index=21)
    large = build_hedge(100.0, book, spy, "choppy", cfg, 2.0, expiry_index=21)
    assert large.contracts == 2 * small.contracts
```

- [ ] **Step 2: Run, verify fail.** → FAIL.

- [ ] **Step 3: Implement `policy.py`**

```python
# quant/options/policy.py
"""Compose beta + regime intensity + structure into a HedgeDecision."""

from __future__ import annotations

import numpy as np

from quant.options.beta import rolling_beta
from quant.options.models import HedgeConfig, HedgeDecision, HedgeStructure
from quant.options.structures import build_structure

_TRADING_DAYS = 252.0


def _intensity(regime_label: str | None, cfg: HedgeConfig) -> float:
    if not cfg.use_regime or regime_label is None:
        return 1.0
    return float(cfg.regime_intensity.get(regime_label, 1.0))


def build_hedge(
    spot: float,
    book_returns_hist: np.ndarray,
    spy_returns_hist: np.ndarray,
    regime_label: str | None,
    cfg: HedgeConfig,
    book_value: float,
    expiry_index: int,
) -> HedgeDecision:
    """Build one roll's hedge. All inputs are strictly trailing (PIT)."""
    beta = rolling_beta(
        book_returns_hist[-cfg.beta_lookback_days :],
        spy_returns_hist[-cfg.beta_lookback_days :],
    )
    intensity = _intensity(regime_label, cfg)
    base = build_structure(spot, cfg)
    structure = HedgeStructure(
        legs=base.legs, spot_at_open=spot, expiry_index=expiry_index
    )

    contracts = cfg.coverage * intensity * beta * book_value / spot if spot > 0 else 0.0

    tenor_years = cfg.tenor_days / 365.0
    vol = _trailing_vol(spy_returns_hist[-cfg.vol_lookback_days :])
    unit_premium = structure.value(spot, tenor_years, vol, cfg.risk_free, cfg.div_yield)
    premium = contracts * unit_premium

    return HedgeDecision(
        structure=structure,
        contracts=contracts,
        premium=premium,
        net_beta=beta,
        regime_label=regime_label,
        intensity=intensity,
    )


def _trailing_vol(window: np.ndarray) -> float:
    arr = np.asarray(window, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return 0.15  # neutral fallback annualized vol
    daily = float(np.std(arr, ddof=1))
    ann = daily * np.sqrt(_TRADING_DAYS)
    return float(ann) if ann > 1e-6 else 0.15
```

- [ ] **Step 4: Run, verify pass.** → PASS.

- [ ] **Step 5: Commit**

```bash
git add quant/options/policy.py tests/options/test_policy.py
git commit -m "feat(options): hedge policy — beta + regime intensity -> sized structure"
```

---

## Task 6: Overlay — `overlay.py`

**Files:**
- Create: `quant/options/overlay.py`
- Test: `tests/options/test_overlay.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/options/test_overlay.py
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
    assert cvar(r, alpha=0.1) < worst_day(r) * 0  # negative tail mean
    assert cvar(r, alpha=0.1) <= r.mean()


def test_protective_put_reduces_drawdown_in_crash():
    # Calm then a sharp crash, book beta ~1 to SPY.
    rng = np.random.default_rng(1)
    calm = rng.normal(0.0005, 0.008, 200)
    crash = np.array([-0.05, -0.07, -0.06, -0.04, -0.08, -0.03])
    book = np.concatenate([calm, crash])
    spy_ret = book.copy()
    spy_close = pd.Series(100.0 * np.cumprod(1 + spy_ret), index=pd.date_range("2020-01-01", periods=len(book), freq="B"))
    returns = pd.Series(book, index=spy_close.index)
    cfg = HedgeConfig(structure="put", use_regime=False, coverage=1.0)
    comp = compare_hedge(returns, spy_close, cfg)
    # hedged max drawdown is less negative (closer to 0) than baseline
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
    T = 180
    trunc, _ = apply_hedge(returns.iloc[:T], spy_close.iloc[:T], cfg)
    np.testing.assert_allclose(full.iloc[:T].to_numpy(), trunc.to_numpy(), atol=0.0)


def test_comparison_is_frozen_with_expected_keys():
    rng = np.random.default_rng(4)
    book = rng.normal(0.0005, 0.008, 120)
    idx = pd.date_range("2020-01-01", periods=len(book), freq="B")
    spy_close = pd.Series(100.0 * np.cumprod(1 + book), index=idx)
    comp = compare_hedge(pd.Series(book, index=idx), spy_close, HedgeConfig(use_regime=False))
    assert isinstance(comp, HedgeComparison)
    for key in ("sharpe", "max_drawdown", "cvar_5", "worst_day", "cagr"):
        assert key in comp.hedged and key in comp.baseline
```

- [ ] **Step 2: Run, verify fail.** → FAIL.

- [ ] **Step 3: Implement `overlay.py`**

```python
# quant/options/overlay.py
"""PIT returns-overlay application of a hedge + baseline-vs-hedged comparison.

Mirrors quant/sizing/backtest.py. Observed-only: never touches live state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant.backtest.metrics import cagr, max_drawdown, sharpe, sortino, total_return, win_rate
from quant.options.models import HedgeConfig, HedgeDecision
from quant.options.policy import build_hedge
from quant.strategies._common import annualize_vol

_TRADING_DAYS = 252


def cvar(returns: pd.Series, alpha: float = 0.05) -> float:
    """Mean of the worst ``alpha`` tail of daily returns (negative number)."""
    arr = returns.to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    k = max(1, int(np.ceil(alpha * arr.size)))
    worst = np.sort(arr)[:k]
    return float(np.mean(worst))


def worst_day(returns: pd.Series) -> float:
    arr = returns.to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.min()) if arr.size else 0.0


def _as_of_label(labels: pd.Series | None, prior_ts: pd.Timestamp | None) -> str | None:
    if labels is None or prior_ts is None or labels.empty:
        return None
    eligible = labels.loc[:prior_ts]
    return None if eligible.empty else str(eligible.iloc[-1])


@dataclass(frozen=True)
class HedgeLedger:
    """Per-roll decisions plus the daily hedge-P&L path."""

    decisions: list[HedgeDecision]
    hedge_pnl: pd.Series


def apply_hedge(
    returns: pd.Series,
    spy_close: pd.Series,
    config: HedgeConfig,
    regime_labels: pd.Series | None = None,
) -> tuple[pd.Series, HedgeLedger]:
    """Apply the hedge overlay. Returns (hedged_returns, ledger), index-aligned.

    At day t the hedge is rolled every ``roll_days`` using only returns[:t] and
    spy_close[:t+1] (today's spot is transactable) and the regime label as-of
    t-1. The held structure is repriced daily; hedge P&L is added to baseline
    equity. PIT — proven by a truncation-invariance test.
    """
    index = returns.index
    n = len(returns)
    r = returns.to_numpy(dtype=float)
    spy = spy_close.reindex(index).to_numpy(dtype=float)

    baseline_equity = np.cumprod(1.0 + np.nan_to_num(r))
    hedge_pnl_daily = np.zeros(n, dtype=float)

    decisions: list[HedgeDecision] = []
    held: HedgeDecision | None = None
    prev_value = 0.0  # per-contract structure value yesterday
    tenor_years = config.tenor_days / 365.0
    bars_to_expiry = max(1, round(config.tenor_days * (_TRADING_DAYS / 365.0)))

    book_ret_hist: list[float] = []
    spy_ret_hist: list[float] = []

    for t in range(n):
        spot = spy[t]
        # update trailing return histories with yesterday's realized return
        if t > 0:
            book_ret_hist.append(r[t - 1])
            prev_spy = spy[t - 1]
            spy_ret_hist.append((spot / prev_spy - 1.0) if prev_spy > 0 else 0.0)

        is_roll = (t % config.roll_days == 0) or held is None
        # remaining time for the held structure (in years), shrinking each day
        if held is not None:
            bars_left = max(0, held.structure.expiry_index - t)
            t_left = bars_left / _TRADING_DAYS
        else:
            t_left = tenor_years

        if is_roll and np.isfinite(spot) and spot > 0:
            label = _as_of_label(regime_labels, index[t - 1] if t > 0 else None)
            vol_hist = np.asarray(spy_ret_hist, dtype=float)
            book_hist = np.asarray(book_ret_hist, dtype=float)
            book_value = float(baseline_equity[t])
            new_dec = build_hedge(
                spot, book_hist, vol_hist, label, config, book_value, expiry_index=t + bars_to_expiry
            )
            # realize close of the prior structure at today's spot before opening new
            if held is not None:
                vol_now = _vol_now(spy_ret_hist, config)
                close_val = held.structure.value(spot, t_left, vol_now, config.risk_free, config.div_yield)
                hedge_pnl_daily[t] += held.contracts * (close_val - prev_value)
            # pay the premium for the new structure today (cost drag)
            hedge_pnl_daily[t] -= new_dec.premium
            held = new_dec
            prev_value = new_dec.structure.value(
                spot, tenor_years, _vol_now(spy_ret_hist, config), config.risk_free, config.div_yield
            )
            decisions.append(new_dec)
        elif held is not None and np.isfinite(spot) and spot > 0:
            vol_now = _vol_now(spy_ret_hist, config)
            cur_val = held.structure.value(spot, t_left, vol_now, config.risk_free, config.div_yield)
            hedge_pnl_daily[t] += held.contracts * (cur_val - prev_value)
            prev_value = cur_val

    hedge_pnl = pd.Series(hedge_pnl_daily, index=index, name="hedge_pnl")
    hedged_equity = baseline_equity + np.cumsum(hedge_pnl_daily)
    hedged_equity = np.maximum(hedged_equity, 1e-9)  # guard div-by-zero
    hedged_ret_vals = np.empty(n, dtype=float)
    hedged_ret_vals[0] = hedged_equity[0] - 1.0
    hedged_ret_vals[1:] = hedged_equity[1:] / hedged_equity[:-1] - 1.0
    hedged = pd.Series(hedged_ret_vals, index=index, name="hedged_returns")
    return hedged, HedgeLedger(decisions=decisions, hedge_pnl=hedge_pnl)


def _vol_now(spy_ret_hist: list[float], config: HedgeConfig) -> float:
    arr = np.asarray(spy_ret_hist[-config.vol_lookback_days :], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return 0.15
    daily = float(np.std(arr, ddof=1))
    ann = daily * np.sqrt(_TRADING_DAYS)
    return ann if ann > 1e-6 else 0.15


def _metrics(returns: pd.Series) -> dict[str, float]:
    return {
        "total_return": total_return(returns),
        "cagr": cagr(returns),
        "sharpe": sharpe(returns),
        "sortino": sortino(returns),
        "max_drawdown": max_drawdown(returns),
        "ann_vol": annualize_vol(returns),
        "win_rate": win_rate(returns),
        "cvar_5": cvar(returns, 0.05),
        "worst_day": worst_day(returns),
    }


@dataclass(frozen=True)
class HedgeComparison:
    """Baseline vs hedged metrics plus a hedge-cost summary."""

    baseline: dict[str, float]
    hedged: dict[str, float]
    total_premium: float
    premium_drag_annual: float
    n_rolls: int
    mean_contracts: float
    config: HedgeConfig


def compare_hedge(
    returns: pd.Series,
    spy_close: pd.Series,
    config: HedgeConfig,
    regime_labels: pd.Series | None = None,
) -> HedgeComparison:
    """Compute baseline and hedged metrics + hedge-cost summary."""
    hedged, ledger = apply_hedge(returns, spy_close, config, regime_labels)
    premiums = [d.premium for d in ledger.decisions]
    total_premium = float(sum(premiums))
    n_rolls = len(ledger.decisions)
    mean_contracts = float(np.mean([d.contracts for d in ledger.decisions])) if n_rolls else 0.0
    years = max(1e-9, len(returns) / _TRADING_DAYS)
    premium_drag_annual = total_premium / years
    return HedgeComparison(
        baseline=_metrics(returns),
        hedged=_metrics(hedged),
        total_premium=total_premium,
        premium_drag_annual=premium_drag_annual,
        n_rolls=n_rolls,
        mean_contracts=mean_contracts,
        config=config,
    )
```

- [ ] **Step 4: Run, verify pass.** `uv run pytest tests/options/test_overlay.py -q` → PASS. If the crash-drawdown test is flaky on magnitude, keep the crash sharp enough that the OTM put goes in-the-money (it does at −30% cumulative).

- [ ] **Step 5: Commit**

```bash
git add quant/options/overlay.py tests/options/test_overlay.py
git commit -m "feat(options): PIT hedge overlay + tail metrics + comparison"
```

---

## Task 7: Public API + property test — `__init__.py`

**Files:**
- Modify: `quant/options/__init__.py`
- Test: `tests/options/test_property.py`

- [ ] **Step 1: Write the property test**

```python
# tests/options/test_property.py
import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from quant.options import HedgeConfig, apply_hedge


@settings(max_examples=30, deadline=None)
@given(
    seed=st.integers(0, 10_000),
    n=st.integers(60, 300),
    trunc=st.integers(40, 59),
)
def test_truncation_invariance(seed, n, trunc):
    rng = np.random.default_rng(seed)
    book = rng.normal(0.0003, 0.012, n)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    spy_close = pd.Series(100.0 * np.cumprod(1 + book), index=idx)
    returns = pd.Series(book, index=idx)
    cfg = HedgeConfig(use_regime=False)
    full, _ = apply_hedge(returns, spy_close, cfg)
    part, _ = apply_hedge(returns.iloc[:trunc], spy_close.iloc[:trunc], cfg)
    np.testing.assert_allclose(full.iloc[:trunc].to_numpy(), part.to_numpy(), atol=0.0)
```

- [ ] **Step 2: Run, verify fail.** → FAIL (import error on `quant.options` exports).

- [ ] **Step 3: Implement `__init__.py`**

```python
# quant/options/__init__.py
"""Options/Greeks engine + protective hedging overlay — an observed, comparison-only signal."""

from quant.options.beta import rolling_beta
from quant.options.models import (
    DEFAULT_REGIME_INTENSITY,
    HedgeConfig,
    HedgeDecision,
    HedgeStructure,
    OptionLeg,
)
from quant.options.overlay import (
    HedgeComparison,
    HedgeLedger,
    apply_hedge,
    compare_hedge,
    cvar,
    worst_day,
)
from quant.options.policy import build_hedge
from quant.options.pricing import Greeks, bs_greeks, bs_price, implied_vol
from quant.options.structures import build_structure, collar, protective_put, put_spread

__all__ = [
    "DEFAULT_REGIME_INTENSITY",
    "Greeks",
    "HedgeComparison",
    "HedgeConfig",
    "HedgeDecision",
    "HedgeLedger",
    "HedgeStructure",
    "OptionLeg",
    "apply_hedge",
    "bs_greeks",
    "bs_price",
    "build_hedge",
    "build_structure",
    "collar",
    "compare_hedge",
    "cvar",
    "implied_vol",
    "protective_put",
    "put_spread",
    "rolling_beta",
    "worst_day",
]
```

- [ ] **Step 4: Run, verify pass.** `uv run pytest tests/options/ -q` → all PASS.

- [ ] **Step 5: Commit**

```bash
git add quant/options/__init__.py tests/options/test_property.py
git commit -m "feat(options): public API exports + truncation-invariance property test"
```

---

## Task 8: CLI — `quant hedge` group

**Files:**
- Modify: `quant/cli.py` (add group after the `sizing` group, ~line 1631)
- Test: `tests/test_cli_hedge.py`

- [ ] **Step 1: Write failing CLI tests**

```python
# tests/test_cli_hedge.py
from click.testing import CliRunner

from quant.cli import cli


def test_hedge_price_prints_greeks():
    runner = CliRunner()
    res = runner.invoke(
        cli, ["hedge", "price", "--spot", "500", "--strike", "480", "--days", "30", "--vol", "0.2"]
    )
    assert res.exit_code == 0, res.output
    assert "delta" in res.output.lower()
    assert "price" in res.output.lower()


def test_hedge_price_with_mark_shows_iv():
    runner = CliRunner()
    res = runner.invoke(
        cli,
        ["hedge", "price", "--spot", "500", "--strike", "480", "--days", "30",
         "--vol", "0.2", "--mark", "8.0", "--right", "put"],
    )
    assert res.exit_code == 0, res.output
    assert "implied" in res.output.lower()
```

(The `hedge compare` path is exercised by the overlay tests; a CLI integration test for
compare requires cached bars, so it is covered by the offline `compare_hedge` tests in
Task 6 rather than a network-dependent CLI test.)

- [ ] **Step 2: Run, verify fail.** → FAIL (no `hedge` command).

- [ ] **Step 3: Implement the CLI group**

Insert after the `sizing` group block (after `sizing_compare`, before `def strategies()`,
around `quant/cli.py:1631`):

```python
@cli.group(help="Options/Greeks engine + protective hedging overlay — observed-only.")
def hedge() -> None:
    pass


@hedge.command("price", help="Black-Scholes price + Greeks (and implied vol if --mark given).")
@click.option("--spot", required=True, type=float)
@click.option("--strike", required=True, type=float)
@click.option("--days", required=True, type=float, help="Calendar days to expiry.")
@click.option("--vol", default=0.20, show_default=True, type=float, help="Annualized vol.")
@click.option("--right", default="put", show_default=True, type=click.Choice(["put", "call"]))
@click.option("--rate", default=0.03, show_default=True, type=float)
@click.option("--div", default=0.015, show_default=True, type=float)
@click.option("--mark", default=None, type=float, help="Market price -> solve implied vol.")
def hedge_price(
    spot: float, strike: float, days: float, vol: float, right: str,
    rate: float, div: float, mark: float | None,
) -> None:
    from quant.options import bs_greeks, bs_price, implied_vol

    t_years = days / 365.0
    price = bs_price(spot, strike, t_years, vol, rate, div, right)
    g = bs_greeks(spot, strike, t_years, vol, rate, div, right)
    table = Table(title=f"{right.capitalize()} {strike:g} / {days:g}d on spot {spot:g}")
    table.add_column("Field")
    table.add_column("Value", justify="right")
    table.add_row("price", f"{price:.4f}")
    table.add_row("delta", f"{g.delta:.4f}")
    table.add_row("gamma", f"{g.gamma:.6f}")
    table.add_row("vega", f"{g.vega:.4f}")
    table.add_row("theta", f"{g.theta:.4f}")
    table.add_row("rho", f"{g.rho:.4f}")
    if mark is not None:
        iv = implied_vol(mark, spot, strike, t_years, rate, div, right)
        table.add_row("implied vol", f"{iv:.4f}")
    console.print(table)


@hedge.command("compare", help="Compare a strategy's returns with vs without the SPY hedge overlay.")
@click.argument("strategy")
@click.option("--start", default="2018-01-01", show_default=True)
@click.option("--end", default=None, help="History end (YYYY-MM-DD). Default: today.")
@click.option("--structure", default="put", show_default=True,
              type=click.Choice(["put", "collar", "put_spread"]))
@click.option("--put-moneyness", default=0.05, show_default=True, type=float)
@click.option("--call-moneyness", default=0.05, show_default=True, type=float)
@click.option("--spread-width", default=0.10, show_default=True, type=float)
@click.option("--coverage", default=1.0, show_default=True, type=float)
@click.option("--tenor-days", default=30, show_default=True, type=int)
@click.option("--roll-days", default=21, show_default=True, type=int)
@click.option("--no-regime", is_flag=True, default=False)
def hedge_compare(
    strategy: str, start: str, end: str | None, structure: str,
    put_moneyness: float, call_moneyness: float, spread_width: float,
    coverage: float, tenor_days: int, roll_days: int, no_regime: bool,
) -> None:
    from quant.options import HedgeConfig, compare_hedge
    from quant.research.registry import ExperimentRecord, append_experiment

    _require_strategy(strategy)
    settings = Settings()  # type: ignore[call-arg]
    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end) if end else date.today()

    console.print(f"[bold]Backtesting {strategy} {start_date}..{end_date}...[/bold]")
    result = _run_single_backtest(strategy, start_date, end_date)
    returns = result.returns
    if returns.empty:
        raise click.ClickException(f"Backtest for {strategy!r} produced no returns.")

    spy_bars = get_bars(BarRequest(symbols=["SPY"], start=start_date, end=end_date))
    if spy_bars.empty:
        raise click.ClickException("No SPY bars cached for the hedge underlying.")
    spy_close = _spy_close_series(spy_bars)

    labels = _load_regime_labels()
    if labels is None and not no_regime:
        console.print("[yellow]No regime series found; hedge intensity will be neutral.[/yellow]")

    config = HedgeConfig(
        structure=structure, put_moneyness=put_moneyness, call_moneyness=call_moneyness,
        spread_width=spread_width, coverage=coverage, tenor_days=tenor_days,
        roll_days=roll_days, use_regime=not no_regime,
    )
    comp = compare_hedge(returns, spy_close, config, regime_labels=labels)

    table = Table(title=f"Hedge comparison — {strategy} ({structure})")
    table.add_column("Metric")
    table.add_column("Baseline", justify="right")
    table.add_column("Hedged", justify="right")
    rows = [
        ("Sharpe", "sharpe"), ("Sortino", "sortino"), ("Max drawdown", "max_drawdown"),
        ("CVaR 5%", "cvar_5"), ("Worst day", "worst_day"), ("Ann vol", "ann_vol"),
        ("CAGR", "cagr"), ("Total return", "total_return"),
    ]
    for label_text, key in rows:
        table.add_row(label_text, f"{comp.baseline[key]:.4f}", f"{comp.hedged[key]:.4f}")
    console.print(table)
    console.print(
        f"Hedge cost — {comp.n_rolls} rolls, total premium {comp.total_premium:.4f}, "
        f"~{comp.premium_drag_annual:.4f}/yr, mean contracts {comp.mean_contracts:.3f}"
    )

    gates = {
        "gate_maxdd_improved": comp.hedged["max_drawdown"] >= comp.baseline["max_drawdown"],
        "gate_cvar_improved": comp.hedged["cvar_5"] >= comp.baseline["cvar_5"],
    }
    append_experiment(
        settings.data_dir / "research" / "experiments.jsonl",
        ExperimentRecord(
            run_id=f"hedge-{uuid.uuid4().hex[:12]}",
            created_at=datetime.now(UTC).replace(microsecond=0),
            strategy=strategy,
            kind="research",
            git_sha=_git_sha(),
            command=f"quant hedge compare {strategy} --structure {structure}",
            params={
                "structure": structure, "put_moneyness": put_moneyness,
                "call_moneyness": call_moneyness, "spread_width": spread_width,
                "coverage": coverage, "tenor_days": tenor_days, "roll_days": roll_days,
                "use_regime": not no_regime,
            },
            metrics={f"hedged_{k}": v for k, v in comp.hedged.items()}
            | {"total_premium": comp.total_premium, "premium_drag_annual": comp.premium_drag_annual},
            gates=gates,
            artifacts={},
            data_snapshot_id=None,
            wall_time_seconds=0.0,
        ),
    )
```

Add the `_spy_close_series` helper near `_load_regime_labels` (around `quant/cli.py:1521`):

```python
def _spy_close_series(spy_bars: pd.DataFrame) -> pd.Series:
    """Extract a clean SPY close series from a bars frame (MultiIndex or flat)."""
    if isinstance(spy_bars.columns, pd.MultiIndex):
        close = spy_bars["SPY"]["close"] if "SPY" in spy_bars.columns.get_level_values(0) else spy_bars.xs("close", axis=1, level=-1).iloc[:, 0]
    else:
        close = spy_bars["close"] if "close" in spy_bars.columns else spy_bars.iloc[:, 0]
    return close.sort_index().astype(float)
```

> Note: confirm the exact column shape `get_bars` returns for a single symbol by reading
> `quant/data/bars.py` before finalizing `_spy_close_series`; adapt the extraction to match.

- [ ] **Step 4: Run, verify pass.** `uv run pytest tests/test_cli_hedge.py -q` → PASS.

- [ ] **Step 5: Smoke-test compare against real cached bars (manual)**

Run: `uv run quant hedge compare trend --start 2018-01-01 --end 2024-12-31`
Expected: prints a baseline-vs-hedged table + hedge-cost line; writes a `hedge-…` record.

- [ ] **Step 6: Commit**

```bash
git add quant/cli.py tests/test_cli_hedge.py
git commit -m "feat(options): quant hedge CLI (price + compare) with registry logging"
```

---

## Task 9: Docs + full-suite green

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add a README section** documenting the pillar (mirror the sizing/monitor sections): what `quant hedge price` and `quant hedge compare` do, the observed-only stance, the analytic-BS approach, and that hedging is expected to trade Sharpe for tail protection.

- [ ] **Step 2: Run the full suite + linters**

```bash
uv run pytest -q
uv run mypy quant
uv run ruff check quant tests
uv run ruff format --check quant tests
```
Expected: all green; suite count ≥ 502 + new tests.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs(options): document the options/Greeks engine + hedging overlay"
```

---

## Self-Review Notes

- **Spec coverage:** pricing (T1), models (T2), structures (T3), beta (T4), policy w/ regime
  intensity (T5), overlay + tail metrics + PIT (T6), exports + property test (T7), CLI
  price+compare + registry gates (T8), docs (T9). All §3–§5 spec items mapped.
- **Type consistency:** `HedgeConfig`/`HedgeStructure`/`HedgeDecision`/`HedgeComparison`
  signatures consistent across policy/overlay/CLI. `build_hedge` signature
  `(spot, book_hist, spy_hist, label, cfg, book_value, expiry_index)` used identically in
  policy tests and overlay.
- **Deferred (spec §6):** live Alpaca recommend, live wiring, vol skew, American exercise,
  per-name — none implemented here, by design.
- **Open verification:** `_spy_close_series` column shape must be confirmed against
  `quant/data/bars.py` at execution time (flagged inline in Task 8).
```
