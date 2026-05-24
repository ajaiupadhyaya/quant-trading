# Validation Methodology Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the full validation battery from §4 of the design spec — Combinatorial Purged CV, Deflated & Probabilistic Sharpe, Monte Carlo bootstrap CIs, and regime stress tests — gated behind a `quant validate <strategy>` CLI command that produces a pass/fail report.

**Architecture:** Five small, pure-Python modules under `quant/backtest/` (`dsr.py`, `bootstrap.py`, `regimes.py`, `cpcv.py`, `validation.py`), each importable independently. The orchestrator in `validation.py` consumes a `WalkforwardResult` plus the bars frame, runs every diagnostic, and emits a `ValidationReport` dataclass with the four boolean pass-criteria. The HTML tear-sheet template gains sections for the new diagnostics. CLI `quant validate <strategy>` wires it end-to-end.

**Tech Stack:** Python 3.12, numpy, pandas, scipy.stats (already a transitive dep via matplotlib; we add it explicitly), pytest, Click. No new third-party packages beyond `scipy` (already widely-installed).

---

## Spec Reference

From `docs/specs/2026-05-23-quant-trading-design.md` §4:

| # | Diagnostic | Pass criterion |
|---|---|---|
| 1 | Walk-forward (done in Plan 2) | — |
| 2 | Combinatorial Purged CV | informs DSR |
| 3 | Deflated Sharpe Ratio | DSR ≥ 0.3 |
| 4 | Probabilistic Sharpe Ratio | PSR(0) ≥ 0.7 |
| 5 | MC bootstrap on returns | lower-5% total return > 0 |
| 6 | Regime stress test | positive return in ≥3 of 5 regimes |
| 7 | OOS-after-param-selection | walk-forward already covers this |
| 8 | Cost-sensitivity (0/5/15/30 bps) | report-only, no gate |

Pass-live gate is the AND of criteria 3, 4, 5, 6.

---

## File Structure

**New files:**
- `quant/backtest/dsr.py` — Probabilistic Sharpe Ratio and Deflated Sharpe Ratio (pure functions, no I/O)
- `quant/backtest/bootstrap.py` — IID and stationary-block resampling on a returns series; CI helpers
- `quant/backtest/regimes.py` — hard-coded historical regime windows + per-regime metric breakdown
- `quant/backtest/cpcv.py` — combinatorial purged cross-validation: group splits, purge/embargo, path aggregator
- `quant/backtest/validation.py` — orchestrator: `ValidationReport`, `run_validation`, pass-criteria gates
- `tests/backtest/test_dsr.py`
- `tests/backtest/test_bootstrap.py`
- `tests/backtest/test_regimes.py`
- `tests/backtest/test_cpcv.py`
- `tests/backtest/test_validation.py`

**Modified files:**
- `quant/backtest/templates/tearsheet.html.j2` — add CPCV / DSR / PSR / Bootstrap / Regime sections
- `quant/backtest/tearsheet.py` — pass new payload into the template
- `quant/cli.py:141-148` — replace stub `validate` with real implementation
- `tests/backtest/test_tearsheet.py` — assert new sections render
- `tests/test_cli.py` — assert `quant validate <slug>` exit codes and output
- `pyproject.toml` — add `scipy>=1.13` to runtime deps
- `README.md` — short "Validation" section under Backtesting

**Untouched (intentional):** strategies, data layer, execution, CI. Plan 3 is pure analytics.

---

## Conventions

- All numeric functions take `pd.Series` of daily simple returns (same convention as `quant/backtest/metrics.py`) and return floats, **not NaNs** — undefined results return `0.0` so tear-sheets never break.
- Per-period (un-annualized) Sharpe is used inside DSR/PSR formulas; annualized Sharpe is used in user-facing reports. Functions document which they take.
- Seeded RNGs everywhere — `np.random.default_rng(seed=...)`. Default `seed=0`.
- Type hints throughout; `from __future__ import annotations` at the top of every new file.
- No new comments unless explaining *why*; identifiers carry the *what*.

---

## Task 1: Probabilistic & Deflated Sharpe Ratio

**Files:**
- Create: `quant/backtest/dsr.py`
- Create: `tests/backtest/test_dsr.py`

PSR is the probability that the true (per-period) Sharpe exceeds a benchmark, given observed Sharpe, skew, kurtosis, and sample length. DSR is PSR evaluated at the multiple-testing-corrected benchmark, given the trial Sharpe variance and trial count.

- [ ] **Step 1: Write the failing tests**

```python
# tests/backtest/test_dsr.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.backtest.dsr import deflated_sharpe, probabilistic_sharpe, _sr_period_from_returns


def _normal_returns(n: int, mean: float, std: float, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", periods=n)
    return pd.Series(rng.normal(mean, std, n), index=idx)


def test_per_period_sharpe_helper_matches_mean_over_std() -> None:
    r = _normal_returns(2000, mean=0.001, std=0.01)
    sr = _sr_period_from_returns(r)
    assert sr == pytest.approx(r.mean() / r.std(ddof=1), rel=1e-9)


def test_psr_returns_high_prob_for_strong_track_record() -> None:
    r = _normal_returns(2000, mean=0.002, std=0.01)  # SR_period ~ 0.2
    psr = probabilistic_sharpe(r, sr_benchmark=0.0)
    assert psr > 0.99


def test_psr_near_half_when_sharpe_equals_benchmark() -> None:
    r = _normal_returns(2000, mean=0.0, std=0.01)
    psr = probabilistic_sharpe(r, sr_benchmark=0.0)
    assert 0.30 < psr < 0.70


def test_psr_returns_zero_on_empty_series() -> None:
    assert probabilistic_sharpe(pd.Series(dtype=float), sr_benchmark=0.0) == 0.0


def test_psr_returns_zero_on_zero_volatility() -> None:
    r = pd.Series([0.0] * 100, index=pd.bdate_range("2010-01-01", periods=100))
    assert probabilistic_sharpe(r, sr_benchmark=0.0) == 0.0


def test_dsr_deflates_strong_in_sample_sharpe_after_many_trials() -> None:
    r = _normal_returns(2000, mean=0.002, std=0.01)
    trial_sharpes = np.linspace(-0.1, 0.2, 50)  # 50 grid trials, max ~0.2 (annualized: high)
    dsr = deflated_sharpe(r, trial_sharpes=trial_sharpes)
    psr = probabilistic_sharpe(r, sr_benchmark=0.0)
    assert dsr < psr  # multiple-testing correction must reduce the probability


def test_dsr_with_one_trial_equals_psr_at_zero() -> None:
    r = _normal_returns(2000, mean=0.001, std=0.01)
    dsr = deflated_sharpe(r, trial_sharpes=np.array([_sr_period_from_returns(r)]))
    psr = probabilistic_sharpe(r, sr_benchmark=0.0)
    # n_trials=1 + zero variance => DSR == PSR(0) exactly
    assert dsr == pytest.approx(psr, abs=1e-9)


def test_dsr_returns_zero_on_empty_inputs() -> None:
    assert deflated_sharpe(pd.Series(dtype=float), trial_sharpes=np.array([])) == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/backtest/test_dsr.py -v`
Expected: ImportError / ModuleNotFoundError on `quant.backtest.dsr`.

- [ ] **Step 3: Implement the module**

```python
# quant/backtest/dsr.py
"""Probabilistic and Deflated Sharpe Ratio (Bailey & Lopez de Prado).

PSR is the probability that the true (per-period) Sharpe exceeds a benchmark.
DSR is PSR with the benchmark adjusted for the multiple-testing bias of
selecting the best among N trial strategies.

All Sharpe values in this module are per-period (un-annualized). Conversion
from a returns series uses the same ddof=1 convention as quant.backtest.metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

_STD_EPS = 1e-12
_EULER_MASCHERONI = 0.5772156649015329


def _sr_period_from_returns(returns: pd.Series) -> float:
    if len(returns) < 2:
        return 0.0
    std = float(returns.std(ddof=1))
    if std <= _STD_EPS:
        return 0.0
    return float(returns.mean()) / std


def _skew_kurt(returns: pd.Series) -> tuple[float, float]:
    """Return (skew, kurtosis_non_excess). Falls back to (0, 3) if undefined."""
    n = len(returns)
    if n < 4:
        return 0.0, 3.0
    arr = returns.to_numpy(dtype=float)
    mu = arr.mean()
    sigma = arr.std(ddof=1)
    if sigma <= _STD_EPS:
        return 0.0, 3.0
    z = (arr - mu) / sigma
    skew = float(np.mean(z**3))
    kurt = float(np.mean(z**4))  # non-excess
    return skew, kurt


def probabilistic_sharpe(returns: pd.Series, sr_benchmark: float) -> float:
    """Probabilistic Sharpe Ratio against a per-period benchmark.

    Returns Pr(true SR > sr_benchmark) given the observed sample. 0.0 on empty
    input or zero volatility.
    """
    n = len(returns)
    if n < 2:
        return 0.0
    sr = _sr_period_from_returns(returns)
    if sr == 0.0 and float(returns.std(ddof=1)) <= _STD_EPS:
        return 0.0
    skew, kurt = _skew_kurt(returns)
    denom_sq = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr * sr
    if denom_sq <= 0.0:
        return 0.0
    z = (sr - sr_benchmark) * np.sqrt(n - 1) / np.sqrt(denom_sq)
    return float(norm.cdf(z))


def deflated_sharpe(returns: pd.Series, trial_sharpes: np.ndarray) -> float:
    """Deflated Sharpe Ratio.

    ``trial_sharpes`` is an array of per-period Sharpe ratios from every
    backtest trial run during model selection (e.g., the cartesian-product
    grid in walk-forward). Returns 0.0 on empty inputs.
    """
    n = len(returns)
    if n < 2 or len(trial_sharpes) == 0:
        return 0.0

    n_trials = int(len(trial_sharpes))
    sr_trial_var = float(np.var(trial_sharpes, ddof=1)) if n_trials > 1 else 0.0
    sr_trial_std = float(np.sqrt(max(sr_trial_var, 0.0)))

    # Expected maximum of N i.i.d. standard normals (Lopez de Prado 2018):
    # E[max] ≈ (1 - γ) Φ⁻¹(1 - 1/N) + γ Φ⁻¹(1 - 1/(N e))
    if n_trials <= 1:
        expected_max_z = 0.0
    else:
        e = np.e
        expected_max_z = (1.0 - _EULER_MASCHERONI) * float(norm.ppf(1.0 - 1.0 / n_trials)) + (
            _EULER_MASCHERONI * float(norm.ppf(1.0 - 1.0 / (n_trials * e)))
        )

    sr_zero = sr_trial_std * expected_max_z
    return probabilistic_sharpe(returns, sr_benchmark=sr_zero)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/backtest/test_dsr.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add quant/backtest/dsr.py tests/backtest/test_dsr.py
git commit -m "feat(backtest): probabilistic + deflated Sharpe ratio"
```

---

## Task 2: Monte Carlo Bootstrap

**Files:**
- Create: `quant/backtest/bootstrap.py`
- Create: `tests/backtest/test_bootstrap.py`

IID resampling of daily returns destroys serial correlation; stationary-block resampling (Politis & Romano 1994) preserves it via geometrically-distributed block lengths. We expose both, plus a high-level `bootstrap_ci` that returns 5th/95th percentile CIs for total return, Sharpe, and max drawdown.

- [ ] **Step 1: Write the failing tests**

```python
# tests/backtest/test_bootstrap.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.backtest.bootstrap import (
    BootstrapCI,
    bootstrap_ci,
    iid_resample,
    stationary_block_resample,
)


def _normal_returns(n: int, mean: float, std: float, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", periods=n)
    return pd.Series(rng.normal(mean, std, n), index=idx)


def test_iid_resample_preserves_length_and_is_a_subset_of_input() -> None:
    r = _normal_returns(500, mean=0.0, std=0.01)
    sample = iid_resample(r, seed=42)
    assert len(sample) == len(r)
    assert set(np.unique(sample.to_numpy())).issubset(set(np.unique(r.to_numpy())))


def test_iid_resample_is_deterministic_under_same_seed() -> None:
    r = _normal_returns(200, 0.0, 0.01)
    s1 = iid_resample(r, seed=7)
    s2 = iid_resample(r, seed=7)
    pd.testing.assert_series_equal(s1, s2)


def test_iid_resample_differs_across_seeds() -> None:
    r = _normal_returns(200, 0.0, 0.01)
    s1 = iid_resample(r, seed=1)
    s2 = iid_resample(r, seed=2)
    assert not s1.equals(s2)


def test_stationary_block_resample_preserves_length() -> None:
    r = _normal_returns(500, 0.0, 0.01)
    s = stationary_block_resample(r, mean_block_len=5, seed=0)
    assert len(s) == len(r)


def test_stationary_block_resample_with_block_len_one_is_iid_like() -> None:
    # mean_block_len=1 collapses to IID; means should be in the same ballpark as IID.
    r = _normal_returns(2000, 0.001, 0.01)
    block = stationary_block_resample(r, mean_block_len=1, seed=0)
    assert block.mean() == pytest.approx(r.mean(), abs=0.003)


def test_bootstrap_ci_returns_bracketing_intervals_for_total_return() -> None:
    r = _normal_returns(1000, mean=0.001, std=0.01)
    ci = bootstrap_ci(r, n_resamples=200, mean_block_len=5, seed=0)
    assert isinstance(ci, BootstrapCI)
    assert ci.total_return_p05 < ci.total_return_median < ci.total_return_p95
    assert ci.sharpe_p05 < ci.sharpe_median < ci.sharpe_p95
    assert ci.max_drawdown_p05 < ci.max_drawdown_p95  # both negative; p05 is "worse"


def test_bootstrap_ci_empty_series_returns_zeros() -> None:
    ci = bootstrap_ci(pd.Series(dtype=float), n_resamples=50, mean_block_len=5, seed=0)
    assert ci.total_return_median == 0.0
    assert ci.sharpe_median == 0.0
    assert ci.max_drawdown_median == 0.0


def test_bootstrap_ci_deterministic_under_seed() -> None:
    r = _normal_returns(400, 0.0005, 0.01)
    a = bootstrap_ci(r, n_resamples=100, mean_block_len=5, seed=11)
    b = bootstrap_ci(r, n_resamples=100, mean_block_len=5, seed=11)
    assert a == b
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/backtest/test_bootstrap.py -v`
Expected: ImportError on `quant.backtest.bootstrap`.

- [ ] **Step 3: Implement the module**

```python
# quant/backtest/bootstrap.py
"""Monte Carlo bootstrap on a daily-returns series.

Two resamplers:
- ``iid_resample``: independent draws with replacement (destroys autocorr).
- ``stationary_block_resample``: Politis & Romano (1994) — geometrically-
  distributed block lengths, wraps around the end (circular). Preserves
  short-range serial correlation, which matters for Sharpe/drawdown CIs.

``bootstrap_ci`` returns 5/50/95 percentiles for total return, Sharpe, and
max drawdown across ``n_resamples`` stationary-block resamples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant.backtest.metrics import max_drawdown, sharpe, total_return


@dataclass(frozen=True)
class BootstrapCI:
    total_return_p05: float
    total_return_median: float
    total_return_p95: float
    sharpe_p05: float
    sharpe_median: float
    sharpe_p95: float
    max_drawdown_p05: float
    max_drawdown_median: float
    max_drawdown_p95: float
    n_resamples: int


def iid_resample(returns: pd.Series, seed: int = 0) -> pd.Series:
    """Independent resample with replacement, preserving length and index."""
    n = len(returns)
    if n == 0:
        return returns.copy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=n)
    return pd.Series(returns.to_numpy()[idx], index=returns.index)


def stationary_block_resample(
    returns: pd.Series, mean_block_len: int = 5, seed: int = 0
) -> pd.Series:
    """Politis-Romano stationary block bootstrap.

    Block lengths are drawn from Geometric(1/mean_block_len); blocks start at
    a uniformly-random offset and wrap around the series end (circular).
    """
    n = len(returns)
    if n == 0:
        return returns.copy()
    if mean_block_len < 1:
        raise ValueError(f"mean_block_len must be >= 1, got {mean_block_len}")

    rng = np.random.default_rng(seed)
    p = 1.0 / mean_block_len
    arr = returns.to_numpy()

    out = np.empty(n, dtype=arr.dtype)
    i = 0
    while i < n:
        start = int(rng.integers(0, n))
        # Geometric(p) here counts the number of Bernoulli trials until the
        # first success; numpy's geometric returns values >= 1, which is the
        # block length we want.
        block_len = int(rng.geometric(p))
        block_len = max(1, block_len)
        block_len = min(block_len, n - i)
        for k in range(block_len):
            out[i + k] = arr[(start + k) % n]
        i += block_len

    return pd.Series(out, index=returns.index)


def bootstrap_ci(
    returns: pd.Series,
    n_resamples: int = 1000,
    mean_block_len: int = 5,
    seed: int = 0,
) -> BootstrapCI:
    """Stationary-block bootstrap CIs for total return, Sharpe, and max DD."""
    if len(returns) == 0 or n_resamples <= 0:
        return BootstrapCI(
            total_return_p05=0.0,
            total_return_median=0.0,
            total_return_p95=0.0,
            sharpe_p05=0.0,
            sharpe_median=0.0,
            sharpe_p95=0.0,
            max_drawdown_p05=0.0,
            max_drawdown_median=0.0,
            max_drawdown_p95=0.0,
            n_resamples=0,
        )

    tr = np.empty(n_resamples, dtype=float)
    sr = np.empty(n_resamples, dtype=float)
    dd = np.empty(n_resamples, dtype=float)

    for k in range(n_resamples):
        sample = stationary_block_resample(returns, mean_block_len, seed=seed + k)
        tr[k] = total_return(sample)
        sr[k] = sharpe(sample)
        dd[k] = max_drawdown(sample)

    return BootstrapCI(
        total_return_p05=float(np.percentile(tr, 5)),
        total_return_median=float(np.percentile(tr, 50)),
        total_return_p95=float(np.percentile(tr, 95)),
        sharpe_p05=float(np.percentile(sr, 5)),
        sharpe_median=float(np.percentile(sr, 50)),
        sharpe_p95=float(np.percentile(sr, 95)),
        max_drawdown_p05=float(np.percentile(dd, 5)),
        max_drawdown_median=float(np.percentile(dd, 50)),
        max_drawdown_p95=float(np.percentile(dd, 95)),
        n_resamples=n_resamples,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/backtest/test_bootstrap.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add quant/backtest/bootstrap.py tests/backtest/test_bootstrap.py
git commit -m "feat(backtest): stationary-block bootstrap + CI helpers"
```

---

## Task 3: Regime Stress Tests

**Files:**
- Create: `quant/backtest/regimes.py`
- Create: `tests/backtest/test_regimes.py`

Hardcoded historical regime windows from §4 of the spec, plus a function that slices a returns series and produces per-regime `total_return`, `sharpe`, `max_drawdown`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/backtest/test_regimes.py
from __future__ import annotations

import numpy as np
import pandas as pd

from quant.backtest.regimes import (
    REGIMES,
    RegimeBreakdown,
    Regime,
    compute_regime_breakdown,
    count_positive_regimes,
)


def test_regimes_constant_has_all_five_windows() -> None:
    slugs = {r.slug for r in REGIMES}
    assert slugs == {"gfc-2008", "china-2015", "covid-2020", "bear-2022", "bull-2024"}


def test_each_regime_has_start_before_end() -> None:
    for r in REGIMES:
        assert r.start < r.end, r.slug


def test_compute_regime_breakdown_returns_one_entry_per_regime() -> None:
    idx = pd.bdate_range("2005-01-01", "2025-01-01")
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.0005, 0.01, len(idx)), index=idx)
    breakdown = compute_regime_breakdown(returns)
    assert len(breakdown) == len(REGIMES)
    slugs = [b.slug for b in breakdown]
    assert slugs == [r.slug for r in REGIMES]


def test_breakdown_handles_regime_with_no_overlap() -> None:
    # Only 2024 dates → 2008 GFC should be empty.
    idx = pd.bdate_range("2024-01-01", "2024-12-31")
    returns = pd.Series(0.001, index=idx)
    breakdown = compute_regime_breakdown(returns)
    gfc = next(b for b in breakdown if b.slug == "gfc-2008")
    assert gfc.n_days == 0
    assert gfc.total_return == 0.0
    assert gfc.sharpe == 0.0


def test_count_positive_regimes_matches_total_return_signs() -> None:
    breakdown = [
        RegimeBreakdown(
            slug=f"r{i}",
            name=f"R{i}",
            start=pd.Timestamp("2020-01-01").date(),
            end=pd.Timestamp("2020-06-01").date(),
            n_days=100,
            total_return=tr,
            sharpe=0.0,
            max_drawdown=0.0,
        )
        for i, tr in enumerate([0.1, -0.05, 0.02, -0.01, 0.0])
    ]
    # Strictly positive only: 0.1 and 0.02 → 2
    assert count_positive_regimes(breakdown) == 2


def test_breakdown_with_constant_positive_drift_yields_positive_total_return() -> None:
    idx = pd.bdate_range("2008-01-01", "2009-06-01")
    returns = pd.Series(0.001, index=idx)
    breakdown = compute_regime_breakdown(returns)
    gfc = next(b for b in breakdown if b.slug == "gfc-2008")
    assert gfc.n_days > 0
    assert gfc.total_return > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/backtest/test_regimes.py -v`
Expected: ImportError on `quant.backtest.regimes`.

- [ ] **Step 3: Implement the module**

```python
# quant/backtest/regimes.py
"""Hard-coded historical regime windows + per-regime metric breakdown."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from quant.backtest.metrics import max_drawdown, sharpe, total_return


@dataclass(frozen=True)
class Regime:
    slug: str
    name: str
    start: date
    end: date


REGIMES: tuple[Regime, ...] = (
    Regime("gfc-2008", "2008 Global Financial Crisis", date(2007, 10, 9), date(2009, 3, 9)),
    Regime("china-2015", "2015-16 China Selloff", date(2015, 8, 1), date(2016, 2, 11)),
    Regime("covid-2020", "2020 COVID Crash", date(2020, 2, 19), date(2020, 4, 7)),
    Regime("bear-2022", "2022 Bear Market", date(2022, 1, 3), date(2022, 10, 12)),
    Regime("bull-2024", "2023-24 Recovery Bull", date(2023, 10, 27), date(2024, 12, 31)),
)


@dataclass(frozen=True)
class RegimeBreakdown:
    slug: str
    name: str
    start: date
    end: date
    n_days: int
    total_return: float
    sharpe: float
    max_drawdown: float


def compute_regime_breakdown(returns: pd.Series) -> list[RegimeBreakdown]:
    """Slice ``returns`` into each regime window and compute key metrics.

    Returns one entry per regime in REGIMES order. Regimes with no overlap
    yield zero metrics (n_days=0).
    """
    out: list[RegimeBreakdown] = []
    for r in REGIMES:
        mask = (returns.index >= pd.Timestamp(r.start)) & (returns.index <= pd.Timestamp(r.end))
        slice_ = returns[mask]
        out.append(
            RegimeBreakdown(
                slug=r.slug,
                name=r.name,
                start=r.start,
                end=r.end,
                n_days=int(len(slice_)),
                total_return=total_return(slice_),
                sharpe=sharpe(slice_),
                max_drawdown=max_drawdown(slice_),
            )
        )
    return out


def count_positive_regimes(breakdown: list[RegimeBreakdown]) -> int:
    """Number of regimes with strictly-positive total return (n_days>0 required)."""
    return sum(1 for b in breakdown if b.n_days > 0 and b.total_return > 0.0)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/backtest/test_regimes.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add quant/backtest/regimes.py tests/backtest/test_regimes.py
git commit -m "feat(backtest): hardcoded regime windows + breakdown"
```

---

## Task 4: Combinatorial Purged Cross-Validation

**Files:**
- Create: `quant/backtest/cpcv.py`
- Create: `tests/backtest/test_cpcv.py`

Lopez de Prado's CPCV. Splits the timeline into N contiguous groups, picks K test groups out of N → C(N,K) combinations. Each combination yields a strategy run on the test groups (with purge/embargo around boundaries). Aggregated, we get the distribution of OOS Sharpes across paths — this distribution feeds DSR's `trial_sharpes` argument as an alternative to the walk-forward grid trials.

The "training" period here is a no-op for backtest purposes — params are fixed (chosen by walk-forward). CPCV measures robustness to test-set placement.

- [ ] **Step 1: Write the failing tests**

```python
# tests/backtest/test_cpcv.py
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from quant.backtest.cpcv import (
    CPCVConfig,
    CPCVResult,
    iter_combinations,
    make_groups,
    run_cpcv,
)
from quant.backtest.engine import BacktestConfig
from tests.conftest import synthetic_bars

from quant.strategies.base import Strategy, StrategySpec


class _EqualWeightOneShot(Strategy):
    spec = StrategySpec(
        slug="cpcv-test-eqw",
        name="EqualWeightOneShot",
        description="Test fixture",
        universe=["AAA", "BBB"],
        rebalance_frequency="monthly",
    )

    def generate_signals(self, asof: date) -> pd.Series:
        return pd.Series({"AAA": 1.0, "BBB": 1.0})

    def target_positions(self, asof: date, equity: float) -> dict[str, int]:
        return {"AAA": 10, "BBB": 10}


def test_make_groups_returns_n_contiguous_index_ranges() -> None:
    idx = pd.DatetimeIndex(pd.bdate_range("2010-01-01", periods=100))
    groups = make_groups(idx, n_groups=5)
    assert len(groups) == 5
    # Contiguous, non-overlapping, covers all
    flat = [t for g in groups for t in g]
    assert flat == list(idx)
    assert all(len(g) >= 20 - 1 for g in groups)


def test_iter_combinations_yields_n_choose_k() -> None:
    from math import comb

    combos = list(iter_combinations(n_groups=6, k_test=2))
    assert len(combos) == comb(6, 2)
    # Each combination is k_test distinct group indices
    for c in combos:
        assert len(set(c)) == 2
        assert all(0 <= i < 6 for i in c)


def test_iter_combinations_invalid_k_raises() -> None:
    with pytest.raises(ValueError):
        list(iter_combinations(n_groups=4, k_test=0))
    with pytest.raises(ValueError):
        list(iter_combinations(n_groups=4, k_test=5))


def test_run_cpcv_returns_one_path_sharpe_per_combination() -> None:
    bars = synthetic_bars(["AAA", "BBB"], date(2018, 1, 1), date(2022, 12, 31))
    cfg = CPCVConfig(n_groups=4, k_test=2, embargo_days=0)
    bt_cfg = BacktestConfig(starting_equity=100_000.0)

    def factory(params: dict, bars: pd.DataFrame) -> Strategy:
        return _EqualWeightOneShot(params=params)

    result = run_cpcv(
        strategy_factory=factory,
        params={},
        bars=bars,
        start=date(2018, 1, 1),
        end=date(2022, 12, 31),
        backtest_config=bt_cfg,
        cpcv_config=cfg,
    )
    from math import comb

    assert isinstance(result, CPCVResult)
    assert len(result.path_sharpes) == comb(4, 2)
    assert result.n_groups == 4
    assert result.k_test == 2


def test_run_cpcv_path_sharpes_finite() -> None:
    bars = synthetic_bars(["AAA", "BBB"], date(2018, 1, 1), date(2022, 12, 31))
    cfg = CPCVConfig(n_groups=4, k_test=2, embargo_days=0)
    bt_cfg = BacktestConfig(starting_equity=100_000.0)

    def factory(params: dict, bars: pd.DataFrame) -> Strategy:
        return _EqualWeightOneShot(params=params)

    result = run_cpcv(
        strategy_factory=factory,
        params={},
        bars=bars,
        start=date(2018, 1, 1),
        end=date(2022, 12, 31),
        backtest_config=bt_cfg,
        cpcv_config=cfg,
    )
    assert np.all(np.isfinite(result.path_sharpes))


def test_run_cpcv_with_empty_window_returns_empty_paths() -> None:
    bars = synthetic_bars(["AAA", "BBB"], date(2018, 1, 1), date(2018, 1, 5))
    cfg = CPCVConfig(n_groups=4, k_test=2, embargo_days=0)
    bt_cfg = BacktestConfig(starting_equity=100_000.0)

    def factory(params: dict, bars: pd.DataFrame) -> Strategy:
        return _EqualWeightOneShot(params=params)

    result = run_cpcv(
        strategy_factory=factory,
        params={},
        bars=bars,
        start=date(2018, 1, 1),
        end=date(2018, 1, 5),
        backtest_config=bt_cfg,
        cpcv_config=cfg,
    )
    # Fewer bars than groups → empty paths
    assert len(result.path_sharpes) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/backtest/test_cpcv.py -v`
Expected: ImportError on `quant.backtest.cpcv`.

- [ ] **Step 3: Implement the module**

```python
# quant/backtest/cpcv.py
"""Combinatorial Purged Cross-Validation (Lopez de Prado 2018).

Splits the timeline into N contiguous groups and runs the strategy on every
C(N, K) combination of K test groups, with an embargo around each test
boundary to mitigate serial-correlation leakage. The aggregated per-path
Sharpe distribution informs the Deflated Sharpe Ratio's trial-count term.

This implementation evaluates a strategy with *fixed* params on each test
segment — params should have been pre-selected by walk-forward. CPCV here
measures robustness of OOS Sharpe to test-set placement, not a search.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from itertools import combinations
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from quant.backtest.engine import BacktestConfig, run_backtest
from quant.backtest.metrics import sharpe

if TYPE_CHECKING:
    from quant.strategies.base import Strategy


@dataclass(frozen=True)
class CPCVConfig:
    n_groups: int = 6
    k_test: int = 2
    embargo_days: int = 5


@dataclass(frozen=True)
class CPCVResult:
    path_sharpes: np.ndarray  # one Sharpe per combination, shape (C(N,K),)
    n_groups: int
    k_test: int


StrategyFactory = Callable[[dict[str, Any], pd.DataFrame], "Strategy"]


def make_groups(index: pd.DatetimeIndex, n_groups: int) -> list[list[pd.Timestamp]]:
    """Partition ``index`` into ``n_groups`` contiguous, non-overlapping groups."""
    if n_groups <= 0:
        raise ValueError(f"n_groups must be > 0, got {n_groups}")
    if len(index) < n_groups:
        return []
    # np.array_split divides as evenly as possible; remainder distributed to leading groups.
    splits = np.array_split(np.asarray(index), n_groups)
    return [list(pd.DatetimeIndex(s)) for s in splits]


def iter_combinations(n_groups: int, k_test: int):
    """Yield each combination of k_test group indices out of range(n_groups)."""
    if k_test <= 0 or k_test >= n_groups:
        raise ValueError(f"k_test must satisfy 0 < k_test < n_groups; got {k_test}, {n_groups}")
    yield from combinations(range(n_groups), k_test)


def _test_window_from_groups(
    groups: list[list[pd.Timestamp]], test_indices: tuple[int, ...]
) -> tuple[date, date]:
    """Concatenate the chosen groups into a single [start, end] date window.

    Returns the contiguous span between the earliest and latest timestamp in the
    chosen groups; for non-contiguous group selections, the inner gap is left
    inside the window (the engine restricts via bars anyway).
    """
    timestamps: list[pd.Timestamp] = []
    for i in test_indices:
        timestamps.extend(groups[i])
    if not timestamps:
        raise ValueError("test_indices selected empty groups")
    return min(timestamps).date(), max(timestamps).date()


def run_cpcv(
    strategy_factory: StrategyFactory,
    params: dict[str, Any],
    bars: pd.DataFrame,
    start: date,
    end: date,
    backtest_config: BacktestConfig,
    cpcv_config: CPCVConfig,
) -> CPCVResult:
    """Run the strategy on each combinatorial test split; return path Sharpes."""
    mask = (bars.index >= pd.Timestamp(start)) & (bars.index <= pd.Timestamp(end))
    window_index = pd.DatetimeIndex(bars.index[mask])

    groups = make_groups(window_index, cpcv_config.n_groups)
    if not groups:
        return CPCVResult(
            path_sharpes=np.array([], dtype=float),
            n_groups=cpcv_config.n_groups,
            k_test=cpcv_config.k_test,
        )

    embargo = timedelta(days=cpcv_config.embargo_days)
    path_sharpes: list[float] = []

    for combo in iter_combinations(cpcv_config.n_groups, cpcv_config.k_test):
        test_start, test_end = _test_window_from_groups(groups, combo)
        # Embargo shrinks the test window away from training boundaries.
        test_start = test_start + embargo
        test_end = test_end - embargo
        if test_end <= test_start:
            continue

        strat = strategy_factory(params, bars)
        bt = run_backtest(strat, bars, backtest_config, test_start, test_end)
        path_sharpes.append(sharpe(bt.returns))

    return CPCVResult(
        path_sharpes=np.asarray(path_sharpes, dtype=float),
        n_groups=cpcv_config.n_groups,
        k_test=cpcv_config.k_test,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/backtest/test_cpcv.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add quant/backtest/cpcv.py tests/backtest/test_cpcv.py
git commit -m "feat(backtest): combinatorial purged cross-validation"
```

---

## Task 5: Validation Orchestrator + Pass Criteria

**Files:**
- Create: `quant/backtest/validation.py`
- Create: `tests/backtest/test_validation.py`

Orchestrator that consumes a `WalkforwardResult`, the original bars, and the strategy factory, runs all four diagnostics (DSR, PSR, Bootstrap, Regime), and emits a `ValidationReport` with the four boolean pass criteria and an overall `passed` field.

- [ ] **Step 1: Write the failing tests**

```python
# tests/backtest/test_validation.py
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from quant.backtest.cpcv import CPCVConfig
from quant.backtest.engine import BacktestConfig
from quant.backtest.validation import (
    ValidationReport,
    THRESHOLDS,
    run_validation,
)
from quant.backtest.walkforward import run_walkforward
from quant.strategies.base import Strategy, StrategySpec
from tests.conftest import synthetic_bars


class _EqualWeightFixture(Strategy):
    spec = StrategySpec(
        slug="validation-test-eqw",
        name="EqualWeightFixture",
        description="Test fixture",
        universe=["AAA", "BBB", "CCC"],
        rebalance_frequency="monthly",
    )
    default_params: dict = {"slot": 0}

    def generate_signals(self, asof: date) -> pd.Series:
        return pd.Series({s: 1.0 for s in self.spec.universe})

    def target_positions(self, asof: date, equity: float) -> dict[str, int]:
        return {"AAA": 10, "BBB": 10, "CCC": 10}


@pytest.fixture(scope="module")
def wf_result_and_bars():
    bars = synthetic_bars(
        ["AAA", "BBB", "CCC"], date(2010, 1, 1), date(2020, 12, 31), drift=0.0005
    )

    def factory(params: dict, bars: pd.DataFrame) -> Strategy:
        return _EqualWeightFixture(params=params)

    bt_cfg = BacktestConfig(starting_equity=100_000.0)
    wf = run_walkforward(
        strategy_factory=factory,
        param_grid={"slot": [0, 1, 2]},
        bars=bars,
        start=date(2010, 1, 1),
        end=date(2020, 12, 31),
        config=bt_cfg,
        train_years=5,
        test_years=1,
        step_months=12,
    )
    return wf, bars, factory


def test_run_validation_returns_report_with_all_fields(wf_result_and_bars) -> None:
    wf, bars, factory = wf_result_and_bars
    report = run_validation(
        wf_result=wf,
        bars=bars,
        strategy_factory=factory,
        chosen_params={"slot": 0},
        backtest_config=BacktestConfig(starting_equity=100_000.0),
        cpcv_config=CPCVConfig(n_groups=4, k_test=2, embargo_days=0),
        bootstrap_resamples=100,
        seed=0,
    )
    assert isinstance(report, ValidationReport)
    assert report.deflated_sharpe >= 0.0
    assert report.probabilistic_sharpe >= 0.0
    assert report.bootstrap_ci is not None
    assert len(report.regime_breakdown) == 5  # five regimes
    assert isinstance(report.passed, bool)


def test_thresholds_match_spec() -> None:
    assert THRESHOLDS.deflated_sharpe == 0.3
    assert THRESHOLDS.probabilistic_sharpe == 0.7
    assert THRESHOLDS.min_positive_regimes == 3


def test_report_passed_is_and_of_four_gates() -> None:
    report = ValidationReport(
        deflated_sharpe=0.5,
        probabilistic_sharpe=0.8,
        bootstrap_ci=None,  # treated as fail when missing
        regime_breakdown=[],
        cpcv_path_sharpes=np.array([]),
        n_positive_regimes=4,
        trial_sharpes=np.array([0.1]),
        gate_deflated_sharpe=True,
        gate_probabilistic_sharpe=True,
        gate_bootstrap_lower=False,
        gate_regime=True,
    )
    assert report.passed is False


def test_report_passed_true_when_all_gates_true() -> None:
    report = ValidationReport(
        deflated_sharpe=0.5,
        probabilistic_sharpe=0.8,
        bootstrap_ci=None,
        regime_breakdown=[],
        cpcv_path_sharpes=np.array([]),
        n_positive_regimes=4,
        trial_sharpes=np.array([0.1]),
        gate_deflated_sharpe=True,
        gate_probabilistic_sharpe=True,
        gate_bootstrap_lower=True,
        gate_regime=True,
    )
    assert report.passed is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/backtest/test_validation.py -v`
Expected: ImportError on `quant.backtest.validation`.

- [ ] **Step 3: Implement the module**

```python
# quant/backtest/validation.py
"""Validation orchestrator: runs the full §4 battery and emits a pass/fail report.

Consumes a WalkforwardResult plus the original bars+factory and produces
booleans for the four pass-live criteria.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from quant.backtest.bootstrap import BootstrapCI, bootstrap_ci
from quant.backtest.cpcv import CPCVConfig, run_cpcv
from quant.backtest.dsr import deflated_sharpe, probabilistic_sharpe
from quant.backtest.engine import BacktestConfig
from quant.backtest.regimes import (
    RegimeBreakdown,
    compute_regime_breakdown,
    count_positive_regimes,
)
from quant.backtest.walkforward import WalkforwardResult

if TYPE_CHECKING:
    from quant.strategies.base import Strategy


StrategyFactory = Callable[[dict[str, Any], pd.DataFrame], "Strategy"]


@dataclass(frozen=True)
class _Thresholds:
    deflated_sharpe: float = 0.3
    probabilistic_sharpe: float = 0.7
    min_positive_regimes: int = 3


THRESHOLDS = _Thresholds()


@dataclass(frozen=True)
class ValidationReport:
    deflated_sharpe: float
    probabilistic_sharpe: float
    bootstrap_ci: BootstrapCI | None
    regime_breakdown: list[RegimeBreakdown]
    cpcv_path_sharpes: np.ndarray
    n_positive_regimes: int
    trial_sharpes: np.ndarray
    gate_deflated_sharpe: bool
    gate_probabilistic_sharpe: bool
    gate_bootstrap_lower: bool
    gate_regime: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return (
            self.gate_deflated_sharpe
            and self.gate_probabilistic_sharpe
            and self.gate_bootstrap_lower
            and self.gate_regime
        )


def _trial_sharpes_from_cpcv(cpcv_paths: np.ndarray) -> np.ndarray:
    """Convert CPCV per-path Sharpes from annualized to per-period scale.

    walk-forward Sharpes are annualized (×√252). DSR's formula expects
    per-period Sharpes, so we divide back out.
    """
    if len(cpcv_paths) == 0:
        return cpcv_paths
    return cpcv_paths / np.sqrt(252.0)


def run_validation(
    wf_result: WalkforwardResult,
    bars: pd.DataFrame,
    strategy_factory: StrategyFactory,
    chosen_params: dict[str, Any],
    backtest_config: BacktestConfig,
    cpcv_config: CPCVConfig = CPCVConfig(),
    bootstrap_resamples: int = 1000,
    bootstrap_block_len: int = 5,
    seed: int = 0,
) -> ValidationReport:
    """Run DSR, PSR, bootstrap, regimes, and CPCV; build the pass-fail report."""
    oos_returns = wf_result.oos_returns

    # CPCV path Sharpes feed DSR's trial count + variance.
    if len(oos_returns) > 0:
        cpcv = run_cpcv(
            strategy_factory=strategy_factory,
            params=chosen_params,
            bars=bars,
            start=oos_returns.index.min().date(),
            end=oos_returns.index.max().date(),
            backtest_config=backtest_config,
            cpcv_config=cpcv_config,
        )
        cpcv_paths = cpcv.path_sharpes
    else:
        cpcv_paths = np.array([], dtype=float)

    trial_sharpes = _trial_sharpes_from_cpcv(cpcv_paths)

    psr = probabilistic_sharpe(oos_returns, sr_benchmark=0.0)
    dsr = deflated_sharpe(oos_returns, trial_sharpes=trial_sharpes)

    if len(oos_returns) > 0:
        ci: BootstrapCI | None = bootstrap_ci(
            oos_returns,
            n_resamples=bootstrap_resamples,
            mean_block_len=bootstrap_block_len,
            seed=seed,
        )
    else:
        ci = None

    breakdown = compute_regime_breakdown(oos_returns)
    n_positive = count_positive_regimes(breakdown)

    gate_dsr = dsr >= THRESHOLDS.deflated_sharpe
    gate_psr = psr >= THRESHOLDS.probabilistic_sharpe
    gate_boot = ci is not None and ci.total_return_p05 > 0.0
    gate_regime = n_positive >= THRESHOLDS.min_positive_regimes

    return ValidationReport(
        deflated_sharpe=dsr,
        probabilistic_sharpe=psr,
        bootstrap_ci=ci,
        regime_breakdown=breakdown,
        cpcv_path_sharpes=cpcv_paths,
        n_positive_regimes=n_positive,
        trial_sharpes=trial_sharpes,
        gate_deflated_sharpe=gate_dsr,
        gate_probabilistic_sharpe=gate_psr,
        gate_bootstrap_lower=gate_boot,
        gate_regime=gate_regime,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/backtest/test_validation.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add quant/backtest/validation.py tests/backtest/test_validation.py
git commit -m "feat(backtest): validation orchestrator + pass-fail report"
```

---

## Task 6: Add scipy to runtime dependencies

**Files:**
- Modify: `pyproject.toml`

scipy is imported by `dsr.py` (`scipy.stats.norm`); add it explicitly.

- [ ] **Step 1: Inspect current deps**

Run: `grep -A30 '^dependencies = \[' pyproject.toml`
Expected: list ending with `]` and no `scipy` line.

- [ ] **Step 2: Add scipy to dependencies**

Add the line `    "scipy>=1.13",` to the `[project] dependencies` array in `pyproject.toml`, keeping alphabetical order. Example final state of that section:

```toml
dependencies = [
    "click>=8.1",
    ...
    "scipy>=1.13",
    ...
]
```

- [ ] **Step 3: Sync the lockfile**

Run: `uv sync`
Expected: scipy installed without errors.

- [ ] **Step 4: Run the full validation test suite to confirm scipy resolves**

Run: `uv run pytest tests/backtest/test_dsr.py tests/backtest/test_validation.py -v`
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add scipy>=1.13 runtime dep for DSR/PSR"
```

---

## Task 7: Tear-sheet renders the new sections

**Files:**
- Modify: `quant/backtest/tearsheet.py`
- Modify: `quant/backtest/templates/tearsheet.html.j2`
- Modify: `tests/backtest/test_tearsheet.py`

Extend the tear-sheet to render DSR/PSR badges, bootstrap CI table, regime breakdown table, and CPCV-path Sharpe distribution histogram. The tear-sheet is the human-readable view of `ValidationReport`.

- [ ] **Step 1: Read the existing template to identify the insertion point**

Run: `wc -l quant/backtest/templates/tearsheet.html.j2`
Expected: a line count. Open the file and identify the closing `</body>` tag.

- [ ] **Step 2: Write the failing test**

Append to `tests/backtest/test_tearsheet.py`:

```python
def test_tearsheet_renders_validation_sections_when_report_provided(tmp_path):
    """When write_tearsheet receives a ValidationReport, render new sections."""
    from datetime import date
    from quant.backtest.cpcv import CPCVConfig
    from quant.backtest.engine import BacktestConfig
    from quant.backtest.tearsheet import write_tearsheet
    from quant.backtest.validation import run_validation
    from quant.backtest.walkforward import run_walkforward
    from quant.strategies.base import Strategy, StrategySpec
    from tests.conftest import synthetic_bars

    class _Eqw(Strategy):
        spec = StrategySpec(
            slug="ts-validation",
            name="TS Validation",
            description="",
            universe=["AAA", "BBB"],
            rebalance_frequency="monthly",
        )

        def generate_signals(self, asof):
            return pd.Series({"AAA": 1.0, "BBB": 1.0})

        def target_positions(self, asof, equity):
            return {"AAA": 10, "BBB": 10}

    bars = synthetic_bars(["AAA", "BBB"], date(2010, 1, 1), date(2020, 12, 31))

    def factory(params, bars):
        return _Eqw(params=params)

    bt_cfg = BacktestConfig(starting_equity=100_000.0)
    wf = run_walkforward(
        strategy_factory=factory,
        param_grid={},
        bars=bars,
        start=date(2010, 1, 1),
        end=date(2020, 12, 31),
        config=bt_cfg,
        train_years=5,
        test_years=1,
        step_months=12,
    )
    report = run_validation(
        wf_result=wf,
        bars=bars,
        strategy_factory=factory,
        chosen_params={},
        backtest_config=bt_cfg,
        cpcv_config=CPCVConfig(n_groups=4, k_test=2, embargo_days=0),
        bootstrap_resamples=50,
    )
    path = write_tearsheet(
        wf,
        slug="ts-validation",
        strategy_name="TS Validation",
        out_dir=tmp_path,
        validation=report,
    )
    html = path.read_text(encoding="utf-8")
    assert "Deflated Sharpe" in html
    assert "Probabilistic Sharpe" in html
    assert "Regime Stress" in html
    assert "Bootstrap" in html


def test_tearsheet_without_validation_still_renders(tmp_path):
    """Tear-sheet remains backwards-compatible when validation is None."""
    from datetime import date
    from quant.backtest.engine import BacktestConfig
    from quant.backtest.tearsheet import write_tearsheet
    from quant.backtest.walkforward import run_walkforward
    from quant.strategies.base import Strategy, StrategySpec
    from tests.conftest import synthetic_bars

    class _Eqw(Strategy):
        spec = StrategySpec(
            slug="ts-novalid",
            name="No Validation",
            description="",
            universe=["AAA"],
            rebalance_frequency="monthly",
        )

        def generate_signals(self, asof):
            return pd.Series({"AAA": 1.0})

        def target_positions(self, asof, equity):
            return {"AAA": 10}

    bars = synthetic_bars(["AAA"], date(2014, 1, 1), date(2020, 12, 31))

    def factory(params, bars):
        return _Eqw(params=params)

    wf = run_walkforward(
        strategy_factory=factory,
        param_grid={},
        bars=bars,
        start=date(2014, 1, 1),
        end=date(2020, 12, 31),
        config=BacktestConfig(),
        train_years=5,
        test_years=1,
        step_months=12,
    )
    path = write_tearsheet(wf, slug="ts-novalid", strategy_name="No Validation", out_dir=tmp_path)
    html = path.read_text(encoding="utf-8")
    # New sections must NOT appear if report is None
    assert "Deflated Sharpe" not in html
    assert "Regime Stress" not in html
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/backtest/test_tearsheet.py -v -k "validation"`
Expected: TypeError (unexpected keyword `validation`) on the first new test.

- [ ] **Step 4: Add `validation` parameter and CPCV histogram helper to `tearsheet.py`**

Add this helper near the other `_*_chart` helpers in `quant/backtest/tearsheet.py`:

```python
def _cpcv_distribution_chart(path_sharpes: np.ndarray) -> str:
    fig, ax = plt.subplots(figsize=(9, 2.5))
    if len(path_sharpes) > 0:
        ax.hist(path_sharpes, bins=min(30, max(5, len(path_sharpes) // 2)),
                color="#2c7fb8", alpha=0.75)
    ax.set_xlabel("CPCV path Sharpe (annualized)")
    ax.set_ylabel("Frequency")
    ax.set_title("CPCV Path Sharpe Distribution")
    ax.grid(True, alpha=0.3)
    return _fig_to_base64(fig)
```

Modify the `write_tearsheet` signature and body:

```python
def write_tearsheet(
    result: WalkforwardResult,
    slug: str,
    strategy_name: str,
    out_dir: Path,
    validation: "ValidationReport | None" = None,
) -> Path:
```

Inside `write_tearsheet`, after computing `charts`, add:

```python
    if validation is not None and len(validation.cpcv_path_sharpes) > 0:
        charts["cpcv"] = _cpcv_distribution_chart(validation.cpcv_path_sharpes)
```

And add `validation=validation,` to the `template.render(...)` call.

Add the import (guarded for cycles):

```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from quant.backtest.validation import ValidationReport
```

- [ ] **Step 5: Extend the Jinja template**

Open `quant/backtest/templates/tearsheet.html.j2`. Immediately before `</body>`, append:

```html
{% if validation %}
<section class="validation">
  <h2>Validation Battery</h2>
  <div class="gate-grid">
    <div class="gate {% if validation.gate_deflated_sharpe %}pass{% else %}fail{% endif %}">
      <div class="gate-label">Deflated Sharpe</div>
      <div class="gate-value">{{ "%.3f"|format(validation.deflated_sharpe) }}</div>
      <div class="gate-thresh">≥ 0.30</div>
    </div>
    <div class="gate {% if validation.gate_probabilistic_sharpe %}pass{% else %}fail{% endif %}">
      <div class="gate-label">Probabilistic Sharpe</div>
      <div class="gate-value">{{ "%.3f"|format(validation.probabilistic_sharpe) }}</div>
      <div class="gate-thresh">≥ 0.70</div>
    </div>
    <div class="gate {% if validation.gate_bootstrap_lower %}pass{% else %}fail{% endif %}">
      <div class="gate-label">Bootstrap lower-5% total return</div>
      <div class="gate-value">
        {% if validation.bootstrap_ci %}{{ "%.2f%%"|format(validation.bootstrap_ci.total_return_p05 * 100) }}{% else %}—{% endif %}
      </div>
      <div class="gate-thresh">&gt; 0</div>
    </div>
    <div class="gate {% if validation.gate_regime %}pass{% else %}fail{% endif %}">
      <div class="gate-label">Regime Stress</div>
      <div class="gate-value">{{ validation.n_positive_regimes }}/5</div>
      <div class="gate-thresh">≥ 3 positive</div>
    </div>
  </div>

  <h3>Regime Breakdown</h3>
  <table class="regime-table">
    <thead><tr><th>Regime</th><th>Window</th><th>Days</th><th>Total Return</th><th>Sharpe</th><th>Max DD</th></tr></thead>
    <tbody>
      {% for r in validation.regime_breakdown %}
      <tr>
        <td>{{ r.name }}</td>
        <td>{{ r.start }} → {{ r.end }}</td>
        <td>{{ r.n_days }}</td>
        <td>{{ "%+.2f%%"|format(r.total_return * 100) }}</td>
        <td>{{ "%.2f"|format(r.sharpe) }}</td>
        <td>{{ "%.2f%%"|format(r.max_drawdown * 100) }}</td>
      </tr>
      {% endfor %}
    </tbody>
  </table>

  {% if validation.bootstrap_ci %}
  <h3>Bootstrap CIs (n={{ validation.bootstrap_ci.n_resamples }}, 5/50/95 pct)</h3>
  <table class="bootstrap-table">
    <thead><tr><th>Metric</th><th>5%</th><th>median</th><th>95%</th></tr></thead>
    <tbody>
      <tr><td>Total return</td>
        <td>{{ "%+.2f%%"|format(validation.bootstrap_ci.total_return_p05 * 100) }}</td>
        <td>{{ "%+.2f%%"|format(validation.bootstrap_ci.total_return_median * 100) }}</td>
        <td>{{ "%+.2f%%"|format(validation.bootstrap_ci.total_return_p95 * 100) }}</td>
      </tr>
      <tr><td>Sharpe</td>
        <td>{{ "%.2f"|format(validation.bootstrap_ci.sharpe_p05) }}</td>
        <td>{{ "%.2f"|format(validation.bootstrap_ci.sharpe_median) }}</td>
        <td>{{ "%.2f"|format(validation.bootstrap_ci.sharpe_p95) }}</td>
      </tr>
      <tr><td>Max drawdown</td>
        <td>{{ "%.2f%%"|format(validation.bootstrap_ci.max_drawdown_p05 * 100) }}</td>
        <td>{{ "%.2f%%"|format(validation.bootstrap_ci.max_drawdown_median * 100) }}</td>
        <td>{{ "%.2f%%"|format(validation.bootstrap_ci.max_drawdown_p95 * 100) }}</td>
      </tr>
    </tbody>
  </table>
  {% endif %}

  {% if charts.cpcv %}
  <h3>CPCV Path Sharpe Distribution</h3>
  <img src="data:image/png;base64,{{ charts.cpcv }}" alt="CPCV histogram" />
  {% endif %}
</section>

<style>
  .gate-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin: 12px 0; }
  .gate { padding: 8px; border-radius: 4px; text-align: center; }
  .gate.pass { background: #d4f4dd; }
  .gate.fail { background: #f4d4d4; }
  .gate-label { font-size: 0.9em; color: #444; }
  .gate-value { font-size: 1.6em; font-weight: 600; margin: 4px 0; }
  .gate-thresh { font-size: 0.8em; color: #666; }
  .regime-table, .bootstrap-table { border-collapse: collapse; margin: 8px 0; width: 100%; }
  .regime-table th, .regime-table td, .bootstrap-table th, .bootstrap-table td {
    padding: 4px 8px; border-bottom: 1px solid #eee; text-align: left;
  }
</style>
{% endif %}
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/backtest/test_tearsheet.py -v`
Expected: all passed (including pre-existing tear-sheet tests and the two new ones).

- [ ] **Step 7: Commit**

```bash
git add quant/backtest/tearsheet.py quant/backtest/templates/tearsheet.html.j2 tests/backtest/test_tearsheet.py
git commit -m "feat(backtest): render validation report in HTML tear-sheet"
```

---

## Task 8: Wire `quant validate <strategy>` CLI

**Files:**
- Modify: `quant/cli.py:141-148`
- Modify: `tests/test_cli.py`

Replace the stub with a real implementation that loads cached bars, runs walk-forward to choose params, runs `run_validation`, prints a Rich-formatted pass/fail summary, writes the tear-sheet with the validation section, and exits non-zero if the strategy fails the gate.

- [ ] **Step 1: Read existing CLI structure to mirror `backtest` command's bar-loading**

Run: `sed -n '86,140p' quant/cli.py`
Expected: see how `backtest` loads bars and constructs the factory. Mirror that.

- [ ] **Step 2: Write the failing CLI test**

The registry is a plain `dict[str, type[Strategy]]` in `quant/strategies/__init__.py` (verified). Use `REGISTRY[slug] = cls` to add and `del REGISTRY[slug]` to remove. Append to `tests/test_cli.py`:

```python
def test_validate_command_exit_code_when_strategy_unknown():
    from click.testing import CliRunner
    from quant.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["validate", "no-such-strategy"])
    assert result.exit_code != 0


def test_validate_command_runs_to_completion_on_known_strategy(monkeypatch, tmp_path, fake_env, tmp_data_dir):
    """Smoke: validate completes (pass or fail) and writes a tear-sheet."""
    from datetime import date

    import pandas as pd
    from click.testing import CliRunner

    from quant.cli import cli
    from quant.strategies import REGISTRY
    from quant.strategies.base import Strategy, StrategySpec
    from tests.conftest import synthetic_bars

    class _Smoke(Strategy):
        spec = StrategySpec(
            slug="cli-smoke",
            name="CLI Smoke",
            description="",
            universe=["AAA"],
            rebalance_frequency="monthly",
        )

        def generate_signals(self, asof):
            return pd.Series({"AAA": 1.0})

        def target_positions(self, asof, equity):
            return {"AAA": 10}

    REGISTRY["cli-smoke"] = _Smoke
    try:
        # Monkeypatch get_bars to skip network and return our synthetic frame.
        bars = synthetic_bars(["AAA"], date(2010, 1, 1), date(2020, 12, 31))
        monkeypatch.setattr("quant.cli.get_bars", lambda _req: bars)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["validate", "cli-smoke", "--start", "2010-01-01", "--end", "2020-12-31",
             "--bootstrap-resamples", "50"],
        )
        # exit_code may be 0 (pass) or 2 (fail-gate); both indicate the command ran.
        assert result.exit_code in (0, 2), result.output
        assert "Deflated Sharpe" in result.output
    finally:
        del REGISTRY["cli-smoke"]
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -v -k validate`
Expected: failure — current stub raises ClickException with text about "not implemented".

- [ ] **Step 4: Replace the validate command in `quant/cli.py`**

The existing `backtest` command (lines 86–138) is the template — same bar-loading via `get_bars(BarRequest(...))`, same `Settings().data_dir / "backtests" / slug` output path, same `console` (already module-global). Mirror that exactly.

Add this import at the top of `quant/cli.py` alongside the existing imports:

```python
from rich.table import Table
```

Then replace lines 141–148 (the existing stub) with:

```python
@cli.command(help="Run the full validation battery (walk-forward + CPCV + DSR + ...).")
@click.argument("strategy")
@click.option("--start", default="2010-01-01", show_default=True,
              help="History start date (YYYY-MM-DD).")
@click.option("--end", default=None, help="History end date (YYYY-MM-DD). Default: today.")
@click.option("--bootstrap-resamples", default=1000, show_default=True, type=int)
@click.option("--cpcv-groups", default=6, show_default=True, type=int)
@click.option("--cpcv-k-test", default=2, show_default=True, type=int)
def validate(
    strategy: str,
    start: str,
    end: str | None,
    bootstrap_resamples: int,
    cpcv_groups: int,
    cpcv_k_test: int,
) -> None:
    from quant.backtest.cpcv import CPCVConfig
    from quant.backtest.validation import run_validation

    _require_strategy(strategy)

    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end) if end else date.today()

    settings = Settings()  # type: ignore[call-arg]
    strategy_cls = REGISTRY[strategy]
    universe = list(strategy_cls.spec.universe)

    console.print(f"[bold]Fetching bars for {len(universe)} symbols...[/bold]")
    bars = get_bars(BarRequest(symbols=universe, start=start_date, end=end_date))
    if bars.empty:
        raise click.ClickException(
            f"No bars returned for {strategy!r} over {start_date}..{end_date}."
        )

    def factory(params: dict[str, object], bars_for_strategy):  # type: ignore[no-untyped-def]
        return strategy_cls.build(bars=bars_for_strategy, params=params)

    console.print("[bold]Running walk-forward...[/bold]")
    wf = run_walkforward(
        strategy_factory=factory,
        param_grid={},
        bars=bars,
        start=start_date,
        end=end_date,
        config=BacktestConfig(),
    )
    chosen = wf.per_window_params[-1][1] if wf.per_window_params else {}

    console.print("[bold]Running validation battery (CPCV + DSR + bootstrap + regimes)...[/bold]")
    report = run_validation(
        wf_result=wf,
        bars=bars,
        strategy_factory=factory,
        chosen_params=chosen,
        backtest_config=BacktestConfig(),
        cpcv_config=CPCVConfig(n_groups=cpcv_groups, k_test=cpcv_k_test),
        bootstrap_resamples=bootstrap_resamples,
    )

    out_dir = settings.data_dir / "backtests" / strategy
    html_path = write_tearsheet(
        result=wf,
        slug=strategy,
        strategy_name=strategy_cls.spec.name,
        out_dir=out_dir,
        validation=report,
    )

    table = Table(title=f"Validation report — {strategy}")
    table.add_column("Gate")
    table.add_column("Value")
    table.add_column("Threshold")
    table.add_column("Pass?")
    table.add_row("Deflated Sharpe", f"{report.deflated_sharpe:.3f}", "≥ 0.30",
                  "✓" if report.gate_deflated_sharpe else "✗")
    table.add_row("Probabilistic Sharpe", f"{report.probabilistic_sharpe:.3f}", "≥ 0.70",
                  "✓" if report.gate_probabilistic_sharpe else "✗")
    boot_lower = (f"{report.bootstrap_ci.total_return_p05 * 100:+.2f}%"
                  if report.bootstrap_ci else "—")
    table.add_row("Bootstrap lower-5%", boot_lower, "> 0",
                  "✓" if report.gate_bootstrap_lower else "✗")
    table.add_row("Regime stress (positive)", f"{report.n_positive_regimes}/5", "≥ 3",
                  "✓" if report.gate_regime else "✗")
    console.print(table)
    console.print(f"\n[bold]Overall: {'PASS' if report.passed else 'FAIL'}[/]")
    console.print(f"Tear-sheet: {html_path}")

    if not report.passed:
        raise SystemExit(2)
```

Note: `Settings()` requires the `fake_env` fixture in tests (already imported by `tests/conftest.py`); add `fake_env` to the test's argument list — already done in Step 2.

- [ ] **Step 5: Run the CLI tests**

Run: `uv run pytest tests/test_cli.py -v -k validate`
Expected: 2 passed.

- [ ] **Step 6: Smoke run from the shell**

Run: `uv run quant validate --help`
Expected: usage prints with the six options listed above.

- [ ] **Step 7: Commit**

```bash
git add quant/cli.py tests/test_cli.py
git commit -m "feat(cli): wire quant validate <strategy> end-to-end"
```

---

## Task 9: README + design-doc stamp

**Files:**
- Modify: `README.md`
- Modify: `docs/specs/2026-05-23-quant-trading-design.md` (status footer)

- [ ] **Step 1: Add Validation section to README**

In `README.md`, under the existing Backtesting section, append:

````markdown
### Validation

After a backtest, run the full §4 validation battery:

```bash
uv run quant validate momentum
```

The report includes:
- **Deflated Sharpe Ratio** — multiple-testing-corrected Sharpe (Bailey & Lopez de Prado).
- **Probabilistic Sharpe Ratio** — Pr(true Sharpe > 0).
- **Stationary-block bootstrap** — 5/50/95 percentile CIs for total return, Sharpe, max DD.
- **Regime stress tests** — per-regime metrics across GFC, China '15, COVID, '22 bear, '24 bull.
- **Combinatorial Purged CV** — path-Sharpe distribution.

Exit code `0` = passes the live gate (DSR ≥ 0.30, PSR ≥ 0.70, bootstrap lower-5% > 0, ≥3 positive regimes). Exit code `2` = fails one or more gates.
````

- [ ] **Step 2: Stamp the design doc**

In `docs/specs/2026-05-23-quant-trading-design.md`, locate the existing "Plan 2 of 6 complete" reference (if any) and add at the top of the doc:

```markdown
> **Implementation status:** Plan 1 ✅ · Plan 2 ✅ · Plan 3 ✅ (this doc) · Plans 4–6 pending.
```

- [ ] **Step 3: Run full test suite to confirm no regressions**

Run: `uv run pytest -q`
Expected: all tests pass, no warnings about deprecated imports.

- [ ] **Step 4: Run lint + types**

Run: `uv run ruff check . && uv run mypy quant`
Expected: 0 errors.

- [ ] **Step 5: Commit**

```bash
git add README.md docs/specs/2026-05-23-quant-trading-design.md
git commit -m "docs: Plan 3 — validation battery available via `quant validate`"
```

---

## Final integration check

- [ ] **Step 1: End-to-end smoke run**

Run: `uv run quant validate --help` and verify the help text.

Run (against any registered strategy with cached bars, or skip if data layer not yet exercised):
```
uv run quant validate <slug> --start 2014-01-01 --end 2020-12-31 --bootstrap-resamples 100
```
Expected: Rich table prints; tear-sheet written to `data/backtests/<slug>/tearsheet.html`; exit code 0 or 2.

- [ ] **Step 2: Confirm the merge branch**

Run: `git log --oneline main..HEAD`
Expected: ~9 commits, one per task, all under the feature branch.

- [ ] **Step 3: Merge to main**

```bash
git checkout main
git merge --no-ff <feature-branch> -m "Merge: Plan 3 of 6 — Validation battery"
git push origin main
```

---

## Out of scope (deferred to later plans)

- **Cost-sensitivity sweep** (0/5/15/30 bps): the spec mentions it as report-only. Easy follow-up; just loop `run_walkforward` with varied `BacktestConfig.slippage_bps` and tabulate. Not gated, so a small Plan 3.1 if needed.
- **quantstats integration**: the spec mentions quantstats as a baseline; we implement our own metrics for cleaner gating, then integrate quantstats display-only in Plan 6 (TUI) if useful.
- **Walk-forward param-grid feeding DSR directly**: currently CPCV provides the trial distribution. An alternative is to pass `select_best_params` grid Sharpes through. Keep CPCV-based for v1; revisit if DSR is unstable.
- **Strategy-specific universe loading in the CLI**: the `validate` command assumes `load_bars(universe, start, end)` exists. If it doesn't, that's a 5-line addition to `quant/data/bars.py` — but verify before implementing.
