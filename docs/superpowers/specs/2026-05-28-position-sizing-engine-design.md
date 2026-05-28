# Position-Sizing Engine Design (Pillar 4)

**Date:** 2026-05-28
**Status:** Approved (autonomous build — user waived approval gates for pillars 2–4)
**Pillar:** 4 of 4 in the autonomous-system vision (regime detector → monitoring daemon → options/Greeks → **position sizing**)

## 1. Goal

A composable, point-in-time **position-sizing layer** that converts a strategy's
raw daily return stream into a risk-managed return stream by applying a daily
**gross-exposure scalar** built from four independent components:

1. **Volatility targeting** — lever toward a target annualized volatility.
2. **Fractional Kelly** — attenuate by trailing edge-to-variance ratio.
3. **Drawdown throttle** — cut exposure as the strategy's own equity draws down.
4. **Regime multiplier** — scale by the Pillar-1 regime label (calm/choppy/crisis).

Built **observed-first**, exactly like the regime engine (Pillar 1): this ships a
**backtest-comparison harness + CLI** that *quantifies* what the sizing overlay
would have done. It does **not** auto-wire into live allocation. Wiring a chosen
sizing policy into `quant rebalance` is a deliberate follow-on spec, gated on the
same evidence discipline the rest of the repo uses.

## 2. Why a returns overlay (not an engine rewrite)

The strategies already emit target shares; the backtest engine fills them. Two
ways to apply sizing:

- **(A) Thread exposure through every strategy's `target_shares`.** Invasive,
  couples sizing to each strategy, hard to compare apples-to-apples.
- **(B) Returns overlay (chosen).** Given a strategy's daily returns `r_t`, apply
  a gross scalar `g_t` known at the *start* of day `t` (computed only from info
  through `t-1`): sized return is `g_t · r_t`. Strategy-agnostic, PIT-clean,
  cheap, trivially composable, and ideal for a comparison harness.

**Documented limitation:** a returns overlay ignores the path-dependence of share
rounding and the extra turnover cost of re-sizing. It answers "how would this
risk transform have reshaped the realized return path?" — a research/observation
question — not "what fills would we have gotten?" That is consistent with the
observed-first stance; cost-aware sizing inside the engine is a follow-on.

## 3. Package layout

New package `quant/sizing/`:

| File | Responsibility |
|------|----------------|
| `components.py` | Four pure component functions. No I/O, no pandas state. |
| `models.py` | `SizingConfig` (frozen) + `SizingDecision` (frozen). |
| `policy.py` | `compute_gross(...)` composes components → `SizingDecision`. |
| `backtest.py` | `apply_sizing(...)`, `compare_sizing(...)`, `SizingComparison`. |
| `__init__.py` | Curated public exports. |

CLI: a new `quant sizing` group with one command, `compare`, in `quant/cli.py`.

Tests under `tests/sizing/`.

No new dependencies. Reuse `quant.strategies._common.annualize_vol`,
`quant.backtest.metrics.*`, `quant.regime` series output, and the existing
backtest + registry plumbing.

## 4. Components (`quant/sizing/components.py`)

All four are pure functions of plain floats / 1-D numpy arrays. Each returns a
finite float and degrades to a neutral value on bad input (never raises, never
returns NaN — downstream registry serialization requires finite metrics).

### 4.1 `vol_target_scale`

```python
def vol_target_scale(realized_vol: float, target_vol: float, max_scale: float) -> float:
    """Leverage scalar to push realized vol toward target.

    scale = target_vol / realized_vol, clamped to [0, max_scale].
    Returns 1.0 (neutral) when realized_vol <= 0 or inputs are non-finite.
    """
```

This is the **base** exposure: lever up in calm periods, down in turbulent ones.
`realized_vol` is the trailing annualized vol of the strategy's own returns.

### 4.2 `fractional_kelly`

```python
def fractional_kelly(
    mean_return: float, variance: float, fraction: float, cap: float
) -> float:
    """Fractional Kelly fraction for a continuous return stream.

    Full Kelly for a return with per-period mean mu and variance s2 is f* = mu / s2.
    Returns clamp(fraction * mu / s2, 0.0, cap). Long-only by construction
    (negative edge -> 0.0). Returns 0.0 when variance <= 0 or inputs non-finite.
    """
```

Acts as an **edge-confidence attenuator** in `[0, cap]` (default cap `1.0`).
Estimated from trailing strategy returns (sample mean + sample variance).

**Double-counting note.** Vol-targeting sizes by *risk*; full Kelly sizes by
*edge / risk*, so naively multiplying both would divide by variance twice. We
resolve this explicitly: vol-targeting is the base exposure and Kelly is a
**unit-capped confidence multiplier** (cap `1.0`). The composition is documented
(§6) and every component is independently toggleable via config, so a user who
wants pure Kelly sizing can disable vol-targeting.

### 4.3 `drawdown_throttle`

```python
def drawdown_throttle(returns_window: np.ndarray, dd_floor: float) -> float:
    """Daniel-Moskowitz exposure attenuator on the strategy's own equity.

    Builds the trailing equity curve from returns_window, computes current
    drawdown vs trailing peak, returns the linear ramp 1 + dd/dd_floor clamped
    to [0, 1]. At zero drawdown -> 1.0; at -dd_floor or worse -> 0.0.
    Returns 1.0 on empty window or dd_floor <= 0.
    """
```

This generalizes the existing `_common.drawdown_leverage_factor` (which runs on a
wide proxy-basket frame) to a **1-D strategy-equity** series — the most direct
instrument for the strategy's own regime risk. The ramp formula is identical, so
behavior is consistent with the strategy-level control already in the repo.

### 4.4 `regime_multiplier`

```python
def regime_multiplier(
    label: str | None, weights: Mapping[str, float], default: float = 1.0
) -> float:
    """Map a regime label to an exposure multiplier.

    weights default to {"calm-bull": 1.0, "choppy": 0.5, "crisis": 0.0}
    (mirrors the de-risk weights used in regime validation). Unknown or None
    label -> default (1.0, i.e. neutral when no regime signal is available).
    """
```

Consumes the Pillar-1 regime series labels. When no regime series is present the
caller passes `label=None` and this returns the neutral `default`.

## 5. Config + decision (`quant/sizing/models.py`)

```python
@dataclass(frozen=True)
class SizingConfig:
    # volatility targeting
    target_vol: float = 0.15          # annualized
    vol_lookback_days: int = 63
    max_leverage: float = 2.0         # also the final gross cap
    use_vol_target: bool = True
    # fractional Kelly
    kelly_fraction: float = 0.5
    kelly_cap: float = 1.0
    kelly_lookback_days: int = 252
    use_kelly: bool = True
    # drawdown throttle
    dd_floor: float = 0.20
    dd_lookback_days: int = 252
    use_drawdown: bool = True
    # regime multiplier
    regime_weights: Mapping[str, float] = <calm-bull:1.0, choppy:0.5, crisis:0.0>
    use_regime: bool = True
```

(`regime_weights` is stored as an immutable mapping; the default is built in
`__post_init__`/`field(default_factory=...)` since dataclass defaults can't be a
bare dict.)

```python
@dataclass(frozen=True)
class SizingDecision:
    gross: float       # final composite scalar, in [0, max_leverage]
    vol_scale: float   # component values (post-toggle; disabled component = 1.0)
    kelly: float
    drawdown: float
    regime: float
```

## 6. Composition (`quant/sizing/policy.py`)

```python
def compute_gross(
    returns_history: np.ndarray,   # trailing strategy returns up to and incl. t-1
    regime_label: str | None,
    config: SizingConfig,
) -> SizingDecision:
```

Steps (each component neutral = `1.0` when its toggle is off):

1. `vol_scale = vol_target_scale(annualize_vol(tail(vol_lookback)), target_vol, max_leverage)`
2. `kelly = fractional_kelly(mean(tail(kelly_lookback)), var(tail(kelly_lookback)), fraction, cap)`
3. `drawdown = drawdown_throttle(tail(dd_lookback), dd_floor)`
4. `regime = regime_multiplier(regime_label, regime_weights)`
5. `gross = clamp(vol_scale * kelly * drawdown * regime, 0.0, max_leverage)`

`annualize_vol` here is reused from `_common` (operates on a pandas Series; the
policy wraps the numpy tail in a Series). Mean/variance for Kelly use sample
statistics (`ddof=1`) on the trailing window, annualized consistently with the
vol estimate (mean × 252, variance × 252) so `mu/s2` is unit-consistent and the
Kelly fraction is scale-correct.

Warm-up: when `returns_history` is shorter than a component's lookback, that
component uses whatever history exists (its function already no-ops to neutral on
too-little data), so early days size near-neutral rather than exploding.

## 7. Backtest application (`quant/sizing/backtest.py`)

```python
def apply_sizing(
    returns: pd.Series,
    config: SizingConfig,
    regime_labels: pd.Series | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Return (sized_returns, gross_path), both indexed like `returns`."""
```

For each integer position `t` in `returns`:

- `hist = returns.to_numpy()[:t]` — rows `0 .. t-1`, strictly **before** `t`
  (PIT: `g_t` knows nothing about `r_t`).
- `label = as_of_label(regime_labels, returns.index[t])` — the most recent label
  **at or before** `returns.index[t-1]` (yesterday's regime), or `None` if no
  series / no prior label. Never peeks at today's label.
- `g_t = compute_gross(hist, label, config).gross`. With empty `hist` (`t == 0`)
  all components no-op to neutral, so `g_0` is well-defined.
- `sized_t = g_t * returns.iloc[t]`.

Returns the sized-returns Series and the gross-exposure path (for diagnostics).

```python
@dataclass(frozen=True)
class SizingComparison:
    baseline: dict[str, float]   # metrics on raw returns
    sized: dict[str, float]      # metrics on sized returns
    gross_mean: float
    gross_min: float
    gross_max: float
    config: SizingConfig

def compare_sizing(
    returns: pd.Series,
    config: SizingConfig,
    regime_labels: pd.Series | None = None,
) -> SizingComparison:
```

Metrics dict (all from existing helpers, all finite): `total_return`, `cagr`,
`sharpe`, `sortino`, `max_drawdown`, `ann_vol` (via `annualize_vol`), `win_rate`.

## 8. CLI: `quant sizing compare`

```
quant sizing compare <strategy> [--start ...] [--end ...]
    [--target-vol 0.15] [--max-leverage 2.0]
    [--kelly-fraction 0.5] [--dd-floor 0.20]
    [--no-vol-target] [--no-kelly] [--no-drawdown] [--no-regime]
```

Behavior:

1. Resolve the strategy from `REGISTRY`, fetch bars over `[start, end]`, run a
   single `run_backtest(strategy_cls.build(bars=bars), bars, BacktestConfig(), start, end)`
   to get `result.returns`.
2. Load the regime series from `data/regime/regime_series.parquet` if present;
   take its `label` column as `regime_labels`. Otherwise `None` (regime component
   neutral) with a printed note.
3. Build `SizingConfig` from the options.
4. `compare_sizing(...)` → print a two-column comparison table (baseline vs sized:
   Sharpe, Sortino, MaxDD, AnnVol, CAGR, TotalReturn, WinRate) plus a gross
   exposure summary (mean / min / max).
5. Append an `ExperimentRecord` (`kind="research"`) to the registry with the sized
   metrics, the config as params, and two boolean gates:
   `gate_sharpe_improved` (sized Sharpe ≥ baseline) and
   `gate_maxdd_improved` (sized max drawdown ≥ baseline, i.e. shallower). Metrics
   serialized must be finite — they are, by construction.

## 9. Point-in-time discipline (the load-bearing invariant)

The whole engine is worthless if it peeks. The guarantees:

- `g_t` is computed from `returns[:t]` (rows strictly before `t`) and yesterday's
  regime label. It multiplies `r_t`. No element of the gross path on day `t`
  depends on any return or label on day `t` or later.
- **Truncation-invariance test (the proof):** computing `apply_sizing` on
  `returns` and on `returns.iloc[:k]` must yield byte-identical gross values for
  the first `k` days. This is the same audit the regime engine uses and is the
  single most important test in the suite.

## 10. Testing strategy

- **`components.py`:** unit tests per function incl. degenerate inputs (zero vol,
  negative edge, empty window, NaN, `dd_floor=0`), monotonicity (higher realized
  vol → lower `vol_target_scale`; deeper drawdown → lower throttle), clamping.
- **`policy.py`:** toggles (each off → that component is exactly `1.0`); gross is
  the product; final clamp at `max_leverage`; warm-up neutrality.
- **`backtest.py`:** PIT truncation-invariance (critical); `apply_sizing` shape +
  index alignment; regime-label as-of lookup never uses today's label; a known
  hand-computed small example; `compare_sizing` metrics finite and dict-complete.
- **CLI:** smoke test via Click runner on a tiny synthetic / cached universe;
  asserts table prints and a registry record is appended.
- **Property test (hypothesis):** for arbitrary finite return arrays, every
  component output is finite and within its documented bound.

## 11. Out of scope (explicit)

- Wiring a chosen sizing policy into live `quant rebalance` (follow-on spec).
- Cost-aware / turnover-aware sizing inside the engine (returns overlay only).
- Per-symbol or cross-sectional sizing (this is portfolio-gross sizing).
- Optimizing / fitting sizing params (this reports a *given* config; no search).

## 12. Relationship to existing code

- Reuses `annualize_vol` and the Daniel-Moskowitz ramp shape from
  `quant/strategies/_common.py` (drawdown_throttle is the 1-D analog).
- Reuses every metric in `quant/backtest/metrics.py`.
- Consumes the Pillar-1 `data/regime/regime_series.parquet` `label` column.
- Logs to the same research registry (`quant/research/registry.py`) as
  walk-forward and validation, with `kind="research"`.
- The regime multiplier's weights mirror the de-risk weights in
  `quant/regime/validation.py`, keeping one canonical notion of "how much to
  trust each regime."
