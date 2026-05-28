# Position-Sizing Engine Implementation Plan (Pillar 4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a composable, point-in-time position-sizing overlay (`quant/sizing/`) that transforms a strategy's daily returns via a gross-exposure scalar (vol-targeting + fractional Kelly + drawdown throttle + regime multiplier), plus a backtest-comparison harness and a `quant sizing compare` CLI. Observed-first; not wired to live allocation.

**Architecture:** Pure component functions → frozen `SizingConfig`/`SizingDecision` → `compute_gross` composition → returns-overlay `apply_sizing`/`compare_sizing` → CLI. PIT invariant: the gross scalar for day `t` uses only returns before `t` and yesterday's regime label.

**Tech Stack:** Python 3.12, numpy, pandas, Click, pytest, hypothesis. uv-managed. mypy-strict, ruff lint+format. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-05-28-position-sizing-engine-design.md`

**Tooling note:** Use `uv run <cmd>` for ALL commands (e.g. `uv run pytest`, `uv run ruff check .`, `uv run ruff format .`, `uv run mypy quant`, `uv run quant ...`). Do NOT call `.venv/bin/...` or bare `python`/`pip`.

---

## File Structure

- Create `quant/sizing/__init__.py` — curated public exports.
- Create `quant/sizing/components.py` — four pure component functions.
- Create `quant/sizing/models.py` — `SizingConfig`, `SizingDecision`.
- Create `quant/sizing/policy.py` — `compute_gross`.
- Create `quant/sizing/backtest.py` — `apply_sizing`, `compare_sizing`, `SizingComparison`.
- Create `tests/sizing/__init__.py` + test modules per source file.
- Modify `quant/cli.py` — add `sizing` group + `compare` command.

---

### Task 1: Component functions

**Files:**
- Create: `quant/sizing/__init__.py` (empty docstring for now)
- Create: `quant/sizing/components.py`
- Test: `tests/sizing/__init__.py`, `tests/sizing/test_components.py`

- [ ] **Step 1: Write failing tests**

Create `tests/sizing/__init__.py` (empty). Create `tests/sizing/test_components.py`:

```python
from __future__ import annotations

import math

import numpy as np

from quant.sizing.components import (
    drawdown_throttle,
    fractional_kelly,
    regime_multiplier,
    vol_target_scale,
)


def test_vol_target_scale_basic() -> None:
    # target 0.15, realized 0.30 -> 0.5, well under cap
    assert vol_target_scale(0.30, 0.15, 2.0) == 0.5


def test_vol_target_scale_clamps_to_max() -> None:
    # realized far below target would lever past the cap -> clamp
    assert vol_target_scale(0.01, 0.15, 2.0) == 2.0


def test_vol_target_scale_neutral_on_zero_vol() -> None:
    assert vol_target_scale(0.0, 0.15, 2.0) == 1.0
    assert vol_target_scale(-1.0, 0.15, 2.0) == 1.0
    assert vol_target_scale(float("nan"), 0.15, 2.0) == 1.0


def test_vol_target_scale_monotonic() -> None:
    # higher realized vol -> lower scale
    assert vol_target_scale(0.20, 0.15, 5.0) > vol_target_scale(0.40, 0.15, 5.0)


def test_fractional_kelly_basic() -> None:
    # mu=0.10, var=0.04 -> full kelly 2.5; half -> 1.25; cap 1.0 -> 1.0
    assert fractional_kelly(0.10, 0.04, 0.5, 1.0) == 1.0
    # smaller edge stays under cap: mu=0.01, var=0.04 -> full 0.25, half 0.125
    assert math.isclose(fractional_kelly(0.01, 0.04, 0.5, 1.0), 0.125)


def test_fractional_kelly_negative_edge_is_zero() -> None:
    assert fractional_kelly(-0.05, 0.04, 0.5, 1.0) == 0.0


def test_fractional_kelly_neutral_on_bad_variance() -> None:
    assert fractional_kelly(0.10, 0.0, 0.5, 1.0) == 0.0
    assert fractional_kelly(0.10, -1.0, 0.5, 1.0) == 0.0
    assert fractional_kelly(float("nan"), 0.04, 0.5, 1.0) == 0.0


def test_drawdown_throttle_no_drawdown() -> None:
    # steadily rising equity -> at peak -> factor 1.0
    rets = np.full(300, 0.001)
    assert drawdown_throttle(rets, 0.20) == 1.0


def test_drawdown_throttle_deep_drawdown_floors_to_zero() -> None:
    # +0% then a -25% cumulative crash with dd_floor 0.20 -> 0.0
    rets = np.concatenate([np.zeros(10), np.full(1, -0.25)])
    assert drawdown_throttle(rets, 0.20) == 0.0


def test_drawdown_throttle_partial_ramp() -> None:
    # equity rises to a peak (1.0) then falls 10% from it: 1 + (-0.10)/0.20 = 0.5.
    # The leading 0.0 establishes the peak — matching the repo convention
    # (metrics.max_drawdown / _common.drawdown_leverage_factor) where equity is
    # cumprod(1+returns) with no implicit leading-capital point, so a lone down
    # day is its own peak (dd=0).
    rets = np.array([0.0, -0.10])
    assert math.isclose(drawdown_throttle(rets, 0.20), 0.5)


def test_drawdown_throttle_neutral_on_empty_or_zero_floor() -> None:
    assert drawdown_throttle(np.array([]), 0.20) == 1.0
    assert drawdown_throttle(np.array([-0.5]), 0.0) == 1.0


def test_regime_multiplier_defaults() -> None:
    w = {"calm-bull": 1.0, "choppy": 0.5, "crisis": 0.0}
    assert regime_multiplier("calm-bull", w) == 1.0
    assert regime_multiplier("choppy", w) == 0.5
    assert regime_multiplier("crisis", w) == 0.0


def test_regime_multiplier_unknown_and_none() -> None:
    w = {"calm-bull": 1.0, "choppy": 0.5, "crisis": 0.0}
    assert regime_multiplier(None, w) == 1.0
    assert regime_multiplier("mystery", w) == 1.0
    assert regime_multiplier("mystery", w, default=0.3) == 0.3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/sizing/test_components.py -q`
Expected: FAIL — `ModuleNotFoundError: quant.sizing.components`.

- [ ] **Step 3: Implement**

Create `quant/sizing/__init__.py`:

```python
"""Composable, point-in-time position sizing — an observed, comparison-only overlay."""
```

Create `quant/sizing/components.py`:

```python
"""Pure position-sizing components.

Each function returns a finite float and degrades to a neutral value on bad
input (never raises, never returns NaN) so downstream registry serialization
stays finite. None of these touch I/O or hold state.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np


def vol_target_scale(realized_vol: float, target_vol: float, max_scale: float) -> float:
    """Leverage scalar pushing realized vol toward target, clamped to [0, max_scale].

    Returns 1.0 (neutral) when realized_vol is non-finite or <= 0.
    """
    if not math.isfinite(realized_vol) or realized_vol <= 0.0:
        return 1.0
    if not math.isfinite(target_vol) or target_vol < 0.0:
        return 1.0
    scale = target_vol / realized_vol
    return float(max(0.0, min(max_scale, scale)))


def fractional_kelly(mean_return: float, variance: float, fraction: float, cap: float) -> float:
    """Fractional Kelly fraction f = clamp(fraction * mean/variance, 0, cap).

    Long-only (negative edge -> 0.0). Returns 0.0 when variance <= 0 or any
    input is non-finite.
    """
    if not (math.isfinite(mean_return) and math.isfinite(variance) and math.isfinite(fraction)):
        return 0.0
    if variance <= 0.0:
        return 0.0
    full = mean_return / variance
    scaled = fraction * full
    return float(max(0.0, min(cap, scaled)))


def drawdown_throttle(returns_window: np.ndarray, dd_floor: float) -> float:
    """Daniel-Moskowitz exposure attenuator on a 1-D strategy-equity series.

    Builds trailing equity from returns_window, computes current drawdown vs
    trailing peak, returns the linear ramp 1 + dd/dd_floor clamped to [0, 1].
    Returns 1.0 on empty window or dd_floor <= 0.
    """
    if dd_floor <= 0.0:
        return 1.0
    arr = np.asarray(returns_window, dtype=float)
    if arr.size == 0:
        return 1.0
    arr = np.nan_to_num(arr, nan=0.0)
    equity = np.cumprod(1.0 + arr)
    peak = float(np.maximum.accumulate(equity)[-1])
    current = float(equity[-1])
    if peak <= 0.0:
        return 1.0
    dd = current / peak - 1.0  # non-positive
    if dd >= 0.0:
        return 1.0
    factor = 1.0 + dd / dd_floor
    return float(max(0.0, min(1.0, factor)))


def regime_multiplier(
    label: str | None, weights: Mapping[str, float], default: float = 1.0
) -> float:
    """Map a regime label to an exposure multiplier; unknown/None -> default."""
    if label is None:
        return default
    value = weights.get(label, default)
    if not math.isfinite(value):
        return default
    return float(value)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/sizing/test_components.py -q`
Expected: PASS (all).

- [ ] **Step 5: Lint + types**

Run: `uv run ruff check quant/sizing tests/sizing && uv run ruff format quant/sizing tests/sizing && uv run mypy quant/sizing`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add quant/sizing/__init__.py quant/sizing/components.py tests/sizing/__init__.py tests/sizing/test_components.py
git commit -m "feat(sizing): pure position-sizing components"
```

---

### Task 2: Config + decision models

**Files:**
- Create: `quant/sizing/models.py`
- Test: `tests/sizing/test_models.py`

- [ ] **Step 1: Write failing tests**

Create `tests/sizing/test_models.py`:

```python
from __future__ import annotations

import dataclasses

from quant.sizing.models import DEFAULT_REGIME_WEIGHTS, SizingConfig, SizingDecision


def test_default_config_values() -> None:
    c = SizingConfig()
    assert c.target_vol == 0.15
    assert c.vol_lookback_days == 63
    assert c.max_leverage == 2.0
    assert c.kelly_fraction == 0.5
    assert c.kelly_cap == 1.0
    assert c.kelly_lookback_days == 252
    assert c.dd_floor == 0.20
    assert c.dd_lookback_days == 252
    assert c.use_vol_target and c.use_kelly and c.use_drawdown and c.use_regime
    assert dict(c.regime_weights) == DEFAULT_REGIME_WEIGHTS


def test_config_is_frozen() -> None:
    c = SizingConfig()
    try:
        c.target_vol = 0.10  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("SizingConfig should be frozen")


def test_two_default_configs_share_equal_weights() -> None:
    # default_factory must not leak a shared mutable across instances in a way
    # that diverges; equal value, independent identity is fine.
    assert dict(SizingConfig().regime_weights) == dict(SizingConfig().regime_weights)


def test_sizing_decision_fields() -> None:
    d = SizingDecision(gross=1.5, vol_scale=1.2, kelly=1.0, drawdown=1.0, regime=1.0)
    assert d.gross == 1.5
    assert d.vol_scale == 1.2
    assert d.kelly == 1.0
    assert d.drawdown == 1.0
    assert d.regime == 1.0
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/sizing/test_models.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement**

Create `quant/sizing/models.py`:

```python
"""Configuration and decision records for the sizing engine."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

DEFAULT_REGIME_WEIGHTS: dict[str, float] = {"calm-bull": 1.0, "choppy": 0.5, "crisis": 0.0}


def _default_regime_weights() -> Mapping[str, float]:
    return MappingProxyType(dict(DEFAULT_REGIME_WEIGHTS))


@dataclass(frozen=True)
class SizingConfig:
    """Knobs for the four-component gross-exposure scalar. All defaults intentional."""

    target_vol: float = 0.15
    vol_lookback_days: int = 63
    max_leverage: float = 2.0
    use_vol_target: bool = True

    kelly_fraction: float = 0.5
    kelly_cap: float = 1.0
    kelly_lookback_days: int = 252
    use_kelly: bool = True

    dd_floor: float = 0.20
    dd_lookback_days: int = 252
    use_drawdown: bool = True

    regime_weights: Mapping[str, float] = field(default_factory=_default_regime_weights)
    use_regime: bool = True


@dataclass(frozen=True)
class SizingDecision:
    """A single day's gross scalar plus its post-toggle component breakdown."""

    gross: float
    vol_scale: float
    kelly: float
    drawdown: float
    regime: float
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/sizing/test_models.py -q`
Expected: PASS.

- [ ] **Step 5: Lint + types**

Run: `uv run ruff check quant/sizing tests/sizing && uv run ruff format quant/sizing tests/sizing && uv run mypy quant/sizing`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add quant/sizing/models.py tests/sizing/test_models.py
git commit -m "feat(sizing): SizingConfig + SizingDecision models"
```

---

### Task 3: Composition (`compute_gross`)

**Files:**
- Create: `quant/sizing/policy.py`
- Test: `tests/sizing/test_policy.py`

- [ ] **Step 1: Write failing tests**

Create `tests/sizing/test_policy.py`:

```python
from __future__ import annotations

import math

import numpy as np

from quant.sizing.models import SizingConfig
from quant.sizing.policy import compute_gross


def _rng_returns(n: int, mu: float, sigma: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(mu, sigma, size=n)


def test_all_toggles_off_gives_unit_gross() -> None:
    cfg = SizingConfig(
        use_vol_target=False, use_kelly=False, use_drawdown=False, use_regime=False
    )
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
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/sizing/test_policy.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement**

Create `quant/sizing/policy.py`:

```python
"""Compose the four sizing components into a single gross-exposure scalar."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.sizing.components import (
    drawdown_throttle,
    fractional_kelly,
    regime_multiplier,
    vol_target_scale,
)
from quant.sizing.models import SizingConfig, SizingDecision
from quant.strategies._common import annualize_vol

_TRADING_DAYS = 252


def compute_gross(
    returns_history: np.ndarray,
    regime_label: str | None,
    config: SizingConfig,
) -> SizingDecision:
    """Build the day's gross scalar from trailing returns + yesterday's regime label.

    ``returns_history`` must contain only returns strictly before the day being
    sized (PIT). Each component no-ops to its neutral value (1.0) when its
    toggle is off or there is too little history.
    """
    arr = np.asarray(returns_history, dtype=float)

    if config.use_vol_target:
        tail = arr[-config.vol_lookback_days :]
        realized = annualize_vol(pd.Series(tail), trading_days=_TRADING_DAYS)
        vol_scale = vol_target_scale(realized, config.target_vol, config.max_leverage)
    else:
        vol_scale = 1.0

    if config.use_kelly:
        ktail = arr[-config.kelly_lookback_days :]
        if ktail.size >= 2:
            mean_ann = float(np.mean(ktail)) * _TRADING_DAYS
            var_ann = float(np.var(ktail, ddof=1)) * _TRADING_DAYS
            kelly = fractional_kelly(mean_ann, var_ann, config.kelly_fraction, config.kelly_cap)
        else:
            kelly = 1.0
    else:
        kelly = 1.0

    if config.use_drawdown:
        dtail = arr[-config.dd_lookback_days :]
        drawdown = drawdown_throttle(dtail, config.dd_floor)
    else:
        drawdown = 1.0

    regime = regime_multiplier(regime_label, config.regime_weights) if config.use_regime else 1.0

    gross = float(max(0.0, min(config.max_leverage, vol_scale * kelly * drawdown * regime)))
    return SizingDecision(
        gross=gross, vol_scale=vol_scale, kelly=kelly, drawdown=drawdown, regime=regime
    )
```

Note on warm-up: when `use_kelly` is on but there are <2 trailing points, Kelly is neutral 1.0 (can't estimate variance) — this keeps early days from collapsing to 0.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/sizing/test_policy.py -q`
Expected: PASS.

- [ ] **Step 5: Lint + types**

Run: `uv run ruff check quant/sizing tests/sizing && uv run ruff format quant/sizing tests/sizing && uv run mypy quant/sizing`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add quant/sizing/policy.py tests/sizing/test_policy.py
git commit -m "feat(sizing): compose components into a gross-exposure scalar"
```

---

### Task 4: Returns-overlay backtest + comparison

**Files:**
- Create: `quant/sizing/backtest.py`
- Test: `tests/sizing/test_backtest.py`

- [ ] **Step 1: Write failing tests**

Create `tests/sizing/test_backtest.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd

from quant.sizing.backtest import SizingComparison, apply_sizing, compare_sizing
from quant.sizing.models import SizingConfig


def _returns(n: int = 400, seed: int = 1) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    return pd.Series(rng.normal(0.0005, 0.01, size=n), index=idx, name="returns")


def test_apply_sizing_shape_and_index() -> None:
    r = _returns()
    sized, gross = apply_sizing(r, SizingConfig())
    assert len(sized) == len(r)
    assert len(gross) == len(r)
    assert (sized.index == r.index).all()
    assert (gross.index == r.index).all()


def test_pit_truncation_invariance() -> None:
    # THE critical test: gross[:k] computed on full series must equal gross
    # computed on the truncated series. No look-ahead.
    r = _returns(n=500, seed=4)
    cfg = SizingConfig()
    _, gross_full = apply_sizing(r, cfg)
    k = 300
    _, gross_trunc = apply_sizing(r.iloc[:k], cfg)
    np.testing.assert_allclose(
        gross_full.iloc[:k].to_numpy(), gross_trunc.to_numpy(), rtol=0, atol=0
    )


def test_gross_uses_only_prior_returns() -> None:
    # Day 0 gross must be the all-neutral default regardless of r[0]'s value,
    # because history before day 0 is empty.
    r = _returns(n=50)
    cfg = SizingConfig()
    _, gross = apply_sizing(r, cfg)
    # empty history -> vol/kelly/drawdown neutral; regime None -> 1.0
    assert gross.iloc[0] == 1.0


def test_regime_label_is_as_of_yesterday() -> None:
    r = _returns(n=10)
    # crisis from day 5 onward; gross on day 5 should still use day 4's label
    labels = pd.Series(["calm-bull"] * 10, index=r.index, name="label")
    labels.iloc[5:] = "crisis"
    cfg = SizingConfig(
        use_vol_target=False, use_kelly=False, use_drawdown=False, use_regime=True
    )
    _, gross = apply_sizing(r, cfg, regime_labels=labels)
    # day 5 uses day 4 label (calm-bull -> 1.0); day 6 uses day 5 label (crisis -> 0.0)
    assert gross.iloc[5] == 1.0
    assert gross.iloc[6] == 0.0


def test_sized_returns_equal_gross_times_returns() -> None:
    r = _returns(n=100, seed=2)
    cfg = SizingConfig()
    sized, gross = apply_sizing(r, cfg)
    np.testing.assert_allclose(sized.to_numpy(), (gross * r).to_numpy())


def test_compare_sizing_returns_complete_finite_metrics() -> None:
    r = _returns()
    comp = compare_sizing(r, SizingConfig())
    assert isinstance(comp, SizingComparison)
    keys = {"total_return", "cagr", "sharpe", "sortino", "max_drawdown", "ann_vol", "win_rate"}
    assert set(comp.baseline) == keys
    assert set(comp.sized) == keys
    assert all(np.isfinite(v) for v in comp.baseline.values())
    assert all(np.isfinite(v) for v in comp.sized.values())
    assert np.isfinite(comp.gross_mean)
    assert comp.gross_min <= comp.gross_mean <= comp.gross_max


def test_compare_sizing_empty_returns_is_safe() -> None:
    comp = compare_sizing(pd.Series(dtype=float), SizingConfig())
    assert all(np.isfinite(v) for v in comp.sized.values())
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/sizing/test_backtest.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement**

Create `quant/sizing/backtest.py`:

```python
"""Returns-overlay application of a sizing policy + baseline-vs-sized comparison."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant.backtest.metrics import (
    cagr,
    max_drawdown,
    sharpe,
    sortino,
    total_return,
    win_rate,
)
from quant.sizing.models import SizingConfig
from quant.sizing.policy import compute_gross
from quant.strategies._common import annualize_vol


def _as_of_label(labels: pd.Series | None, prior_ts: pd.Timestamp | None) -> str | None:
    """Most recent label at or before ``prior_ts`` (yesterday). None if unavailable."""
    if labels is None or prior_ts is None or labels.empty:
        return None
    eligible = labels.loc[:prior_ts]
    if eligible.empty:
        return None
    return str(eligible.iloc[-1])


def apply_sizing(
    returns: pd.Series,
    config: SizingConfig,
    regime_labels: pd.Series | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Apply the sizing overlay. Returns (sized_returns, gross_path), index-aligned.

    For day t, the gross scalar is computed from returns[:t] (strictly prior)
    and the regime label as of t-1 — never today's return or label.
    """
    arr = returns.to_numpy(dtype=float)
    index = returns.index
    n = len(returns)
    gross_vals = np.empty(n, dtype=float)
    for t in range(n):
        hist = arr[:t]
        prior_ts = index[t - 1] if t > 0 else None
        label = _as_of_label(regime_labels, prior_ts)
        gross_vals[t] = compute_gross(hist, label, config).gross
    gross = pd.Series(gross_vals, index=index, name="gross")
    sized = pd.Series(gross_vals * arr, index=index, name="sized_returns")
    return sized, gross


def _metrics(returns: pd.Series) -> dict[str, float]:
    return {
        "total_return": total_return(returns),
        "cagr": cagr(returns),
        "sharpe": sharpe(returns),
        "sortino": sortino(returns),
        "max_drawdown": max_drawdown(returns),
        "ann_vol": annualize_vol(returns),
        "win_rate": win_rate(returns),
    }


@dataclass(frozen=True)
class SizingComparison:
    """Baseline vs sized metrics plus gross-exposure summary."""

    baseline: dict[str, float]
    sized: dict[str, float]
    gross_mean: float
    gross_min: float
    gross_max: float
    config: SizingConfig


def compare_sizing(
    returns: pd.Series,
    config: SizingConfig,
    regime_labels: pd.Series | None = None,
) -> SizingComparison:
    """Compute baseline and sized metrics for ``returns`` under ``config``."""
    sized, gross = apply_sizing(returns, config, regime_labels)
    if len(gross) == 0:
        gmean = gmin = gmax = 0.0
    else:
        gmean = float(gross.mean())
        gmin = float(gross.min())
        gmax = float(gross.max())
    return SizingComparison(
        baseline=_metrics(returns),
        sized=_metrics(sized),
        gross_mean=gmean,
        gross_min=gmin,
        gross_max=gmax,
        config=config,
    )
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/sizing/test_backtest.py -q`
Expected: PASS.

- [ ] **Step 5: Lint + types**

Run: `uv run ruff check quant/sizing tests/sizing && uv run ruff format quant/sizing tests/sizing && uv run mypy quant/sizing`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add quant/sizing/backtest.py tests/sizing/test_backtest.py
git commit -m "feat(sizing): PIT returns-overlay backtest + comparison harness"
```

---

### Task 5: Public exports + property test

**Files:**
- Modify: `quant/sizing/__init__.py`
- Test: `tests/sizing/test_properties.py`

- [ ] **Step 1: Write failing tests**

Create `tests/sizing/test_properties.py`:

```python
from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from quant.sizing import (
    SizingComparison,
    SizingConfig,
    SizingDecision,
    apply_sizing,
    compare_sizing,
    compute_gross,
    drawdown_throttle,
    fractional_kelly,
    regime_multiplier,
    vol_target_scale,
)

_finite = st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6)


def test_public_api_is_importable() -> None:
    assert SizingConfig and SizingDecision and SizingComparison
    assert compute_gross and apply_sizing and compare_sizing
    assert vol_target_scale and fractional_kelly and drawdown_throttle and regime_multiplier


@given(rv=_finite, tv=st.floats(0.0, 1.0), ms=st.floats(0.0, 10.0))
def test_vol_target_scale_bounded(rv: float, tv: float, ms: float) -> None:
    out = vol_target_scale(rv, tv, ms)
    assert np.isfinite(out)
    assert 0.0 <= out <= max(1.0, ms)


@given(
    arr=st.lists(st.floats(-0.5, 0.5, allow_nan=False), min_size=0, max_size=300),
    floor=st.floats(0.0, 1.0),
)
def test_drawdown_throttle_bounded(arr: list[float], floor: float) -> None:
    out = drawdown_throttle(np.array(arr, dtype=float), floor)
    assert np.isfinite(out)
    assert 0.0 <= out <= 1.0


@settings(max_examples=50)
@given(
    arr=st.lists(st.floats(-0.2, 0.2, allow_nan=False), min_size=0, max_size=400),
    label=st.sampled_from([None, "calm-bull", "choppy", "crisis", "unknown"]),
)
def test_compute_gross_finite_and_capped(arr: list[float], label: str | None) -> None:
    cfg = SizingConfig()
    d = compute_gross(np.array(arr, dtype=float), label, cfg)
    assert np.isfinite(d.gross)
    assert 0.0 <= d.gross <= cfg.max_leverage
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/sizing/test_properties.py -q`
Expected: FAIL — imports not exported from `quant.sizing`.

- [ ] **Step 3: Implement exports**

Overwrite `quant/sizing/__init__.py`:

```python
"""Composable, point-in-time position sizing — an observed, comparison-only overlay."""

from quant.sizing.backtest import SizingComparison, apply_sizing, compare_sizing
from quant.sizing.components import (
    drawdown_throttle,
    fractional_kelly,
    regime_multiplier,
    vol_target_scale,
)
from quant.sizing.models import DEFAULT_REGIME_WEIGHTS, SizingConfig, SizingDecision
from quant.sizing.policy import compute_gross

__all__ = [
    "DEFAULT_REGIME_WEIGHTS",
    "SizingComparison",
    "SizingConfig",
    "SizingDecision",
    "apply_sizing",
    "compare_sizing",
    "compute_gross",
    "drawdown_throttle",
    "fractional_kelly",
    "regime_multiplier",
    "vol_target_scale",
]
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/sizing/test_properties.py -q`
Expected: PASS.

- [ ] **Step 5: Lint + types**

Run: `uv run ruff check quant/sizing tests/sizing && uv run ruff format quant/sizing tests/sizing && uv run mypy quant/sizing`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add quant/sizing/__init__.py tests/sizing/test_properties.py
git commit -m "feat(sizing): public API exports + hypothesis property tests"
```

---

### Task 6: CLI `quant sizing compare`

**Files:**
- Modify: `quant/cli.py` (add `sizing` group + `compare` command near the regime group, after the regime group definition block — append at end of file is fine)
- Test: `tests/sizing/test_cli.py`

- [ ] **Step 1: Write failing test**

Create `tests/sizing/test_cli.py`:

```python
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from quant.backtest.engine import BacktestResult
from quant.cli import cli


def _fake_result(n: int = 300) -> BacktestResult:
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    rng = np.random.default_rng(0)
    rets = pd.Series(rng.normal(0.0005, 0.01, size=n), index=idx, name="returns")
    equity = (1.0 + rets).cumprod() * 100_000.0
    from quant.backtest.engine import BacktestConfig

    return BacktestResult(
        equity_curve=equity,
        returns=rets,
        positions=pd.DataFrame(index=idx),
        trades=pd.DataFrame(),
        config=BacktestConfig(),
        starting_equity=100_000.0,
        ending_equity=float(equity.iloc[-1]),
    )


def test_sizing_compare_smoke(
    tmp_data_dir: Path, fake_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # tmp_data_dir points QUANT_DATA_DIR at a sandbox + makes the dir tree;
    # fake_env supplies dummy Alpaca/FRED creds so Settings() doesn't fail.
    # Stub bars + backtest so the test doesn't hit the network.
    import quant.cli as cli_mod

    monkeypatch.setattr(
        cli_mod, "get_bars", lambda *a, **k: pd.DataFrame({("SPY", "close"): [1.0, 2.0]})
    )
    monkeypatch.setattr(cli_mod, "_run_single_backtest", lambda *a, **k: _fake_result())

    runner = CliRunner()
    end = date.today()
    start = end - timedelta(days=900)
    res = runner.invoke(
        cli,
        ["sizing", "compare", "trend", "--start", str(start), "--end", str(end)],
    )
    assert res.exit_code == 0, res.output
    assert "Sharpe" in res.output
    assert "Gross exposure" in res.output
    # registry record appended
    reg = tmp_data_dir / "research" / "experiments.jsonl"
    assert reg.exists()
```

The `tmp_data_dir` and `fake_env` fixtures live in `tests/conftest.py`. Add
`import pytest` to the test module's imports. The data-dir env var is
`QUANT_DATA_DIR` (confirmed in `quant/util/config.py`).

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/sizing/test_cli.py -q`
Expected: FAIL — no `sizing` command / no `_run_single_backtest`.

- [ ] **Step 3: Implement CLI**

First read `quant/util/config.py` to confirm the data-dir env var and `Settings` shape. Then append to `quant/cli.py` (after the regime group block). Add a small helper `_run_single_backtest` so the test can monkeypatch it:

```python
@cli.group(help="Position sizing — an observed, comparison-only overlay (vol-target/Kelly/dd/regime).")
def sizing() -> None:
    pass


def _run_single_backtest(strategy_slug: str, start_date: date, end_date: date):  # type: ignore[no-untyped-def]
    """Run one default-param backtest and return its BacktestResult."""
    from quant.backtest.engine import run_backtest

    strategy_cls = REGISTRY[strategy_slug]
    universe = list(strategy_cls.spec.universe)
    bars = get_bars(BarRequest(symbols=universe, start=start_date, end=end_date))
    if bars.empty:
        raise click.ClickException(f"No bars for {strategy_slug!r} over {start_date}..{end_date}.")
    strat = strategy_cls.build(bars=bars)
    return run_backtest(strat, bars, BacktestConfig(), start_date, end_date)


def _load_regime_labels() -> pd.Series | None:
    path = _regime_series_path()
    if not path.exists():
        return None
    frame = pd.read_parquet(path)
    if "label" not in frame.columns:
        return None
    return frame["label"]


@sizing.command("compare", help="Compare a strategy's returns with vs without the sizing overlay.")
@click.argument("strategy")
@click.option("--start", default="2018-01-01", show_default=True)
@click.option("--end", default=None, help="History end (YYYY-MM-DD). Default: today.")
@click.option("--target-vol", default=0.15, show_default=True, type=float)
@click.option("--max-leverage", default=2.0, show_default=True, type=float)
@click.option("--kelly-fraction", default=0.5, show_default=True, type=float)
@click.option("--dd-floor", default=0.20, show_default=True, type=float)
@click.option("--no-vol-target", is_flag=True, default=False)
@click.option("--no-kelly", is_flag=True, default=False)
@click.option("--no-drawdown", is_flag=True, default=False)
@click.option("--no-regime", is_flag=True, default=False)
def sizing_compare(
    strategy: str,
    start: str,
    end: str | None,
    target_vol: float,
    max_leverage: float,
    kelly_fraction: float,
    dd_floor: float,
    no_vol_target: bool,
    no_kelly: bool,
    no_drawdown: bool,
    no_regime: bool,
) -> None:
    from quant.research.registry import ExperimentRecord, append_experiment
    from quant.sizing import SizingConfig, compare_sizing

    _require_strategy(strategy)
    settings = Settings()  # type: ignore[call-arg]
    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end) if end else date.today()

    console.print(f"[bold]Backtesting {strategy} {start_date}..{end_date}...[/bold]")
    result = _run_single_backtest(strategy, start_date, end_date)
    returns = result.returns
    if returns.empty:
        raise click.ClickException(f"Backtest for {strategy!r} produced no returns.")

    labels = _load_regime_labels()
    if labels is None:
        console.print("[yellow]No regime series found; regime component will be neutral.[/yellow]")

    config = SizingConfig(
        target_vol=target_vol,
        max_leverage=max_leverage,
        kelly_fraction=kelly_fraction,
        dd_floor=dd_floor,
        use_vol_target=not no_vol_target,
        use_kelly=not no_kelly,
        use_drawdown=not no_drawdown,
        use_regime=not no_regime,
    )
    comp = compare_sizing(returns, config, regime_labels=labels)

    table = Table(title=f"Sizing comparison — {strategy}")
    table.add_column("Metric")
    table.add_column("Baseline", justify="right")
    table.add_column("Sized", justify="right")
    rows = [
        ("Sharpe", "sharpe"),
        ("Sortino", "sortino"),
        ("Max drawdown", "max_drawdown"),
        ("Ann vol", "ann_vol"),
        ("CAGR", "cagr"),
        ("Total return", "total_return"),
        ("Win rate", "win_rate"),
    ]
    for label_text, key in rows:
        table.add_row(label_text, f"{comp.baseline[key]:.4f}", f"{comp.sized[key]:.4f}")
    console.print(table)
    console.print(
        f"Gross exposure — mean {comp.gross_mean:.2f}, "
        f"min {comp.gross_min:.2f}, max {comp.gross_max:.2f}"
    )

    gates = {
        "gate_sharpe_improved": comp.sized["sharpe"] >= comp.baseline["sharpe"],
        "gate_maxdd_improved": comp.sized["max_drawdown"] >= comp.baseline["max_drawdown"],
    }
    append_experiment(
        settings.data_dir / "research" / "experiments.jsonl",
        ExperimentRecord(
            run_id=f"sizing-{uuid.uuid4().hex[:12]}",
            created_at=datetime.now(UTC).replace(microsecond=0),
            strategy=strategy,
            kind="research",
            git_sha=_git_sha(),
            command=f"quant sizing compare {strategy} --start {start_date} --end {end_date}",
            params={
                "target_vol": target_vol,
                "max_leverage": max_leverage,
                "kelly_fraction": kelly_fraction,
                "dd_floor": dd_floor,
                "use_vol_target": not no_vol_target,
                "use_kelly": not no_kelly,
                "use_drawdown": not no_drawdown,
                "use_regime": not no_regime,
            },
            metrics={f"sized_{k}": v for k, v in comp.sized.items()}
            | {"gross_mean": comp.gross_mean},
            gates=gates,
            artifacts={},
            data_snapshot_id=None,
            wall_time_seconds=0.0,
        ),
    )
```

If `Settings` reads a different env var than `QUANT_DATA_DIR`, fix the test in Step 1 to match what `quant/util/config.py` actually uses.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/sizing/test_cli.py -q`
Expected: PASS.

- [ ] **Step 5: Lint + types**

Run: `uv run ruff check quant tests/sizing && uv run ruff format quant tests/sizing && uv run mypy quant`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add quant/cli.py tests/sizing/test_cli.py
git commit -m "feat(sizing): quant sizing compare CLI with registry logging"
```

---

### Task 7: Full-suite green + README note

**Files:**
- Modify: `README.md` (add a short "Position sizing (Pillar 4)" subsection near the regime section, if one exists)

- [ ] **Step 1: Full suite**

Run: `uv run pytest -q`
Expected: PASS (all prior tests + new sizing tests).

- [ ] **Step 2: Lint + format + types on whole repo**

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy quant`
Expected: all clean. (Note `ruff format --check` — the regime work previously went red because only `ruff check` was run.)

- [ ] **Step 3: README note**

Read `README.md`, find where the regime engine is documented, and add a concise subsection:

```markdown
### Position sizing (observed overlay)

`quant sizing compare <strategy>` reports how a composable gross-exposure
overlay — volatility targeting, fractional Kelly, a drawdown throttle, and the
regime multiplier — would have reshaped a strategy's realized return path. It is
point-in-time (the day-`t` scalar uses only data through `t-1`) and observation-
only: it does not change live allocation. Components are individually toggleable
(`--no-vol-target`, `--no-kelly`, `--no-drawdown`, `--no-regime`).
```

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs(sizing): document the position-sizing overlay in the README"
```

---

## Self-Review Notes

- **Spec coverage:** components (§4) → Task 1; models (§5) → Task 2; composition (§6) → Task 3; backtest/comparison (§7) → Task 4; PIT invariant (§9) → Task 4 truncation test; CLI (§8) → Task 6; testing strategy (§10) → Tasks 1–6 incl. property test (Task 5). All covered.
- **Type consistency:** `apply_sizing` returns `tuple[pd.Series, pd.Series]` everywhere it's referenced; `compare_sizing` consumes it; `SizingComparison` fields match between def and CLI usage; `compute_gross(returns_history, regime_label, config)` signature consistent across policy + backtest + tests.
- **No placeholders:** every code step has complete code. The only conditional is the `Settings` env-var name in Task 6, with an explicit instruction to confirm against `quant/util/config.py` — not a placeholder but a verify-then-use step.
