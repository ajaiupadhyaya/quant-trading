"""Deterministic capital allocation for live-governed strategies."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from quant.governance.models import GovernanceState, StrategyState, ValidationEvidence
from quant.sizing.components import fractional_kelly

AllocationMode = Literal[
    "equal-live", "dsr-weighted", "capped-evidence-score", "risk-parity", "fractional-kelly", "hrp"
]

_RISK_MODES: frozenset[str] = frozenset({"risk-parity", "fractional-kelly"})


@dataclass(frozen=True)
class AllocationConfig:
    mode: AllocationMode = "equal-live"
    max_weight: float = 0.40
    min_weight: float = 0.05
    # Risk-based modes only (ignored by equal-live / evidence modes):
    kelly_fraction: float = 0.5
    kelly_cap: float = 1.0
    min_observations: int = 60


def allocate_capital(
    states_by_slug: dict[str, StrategyState],
    *,
    evidence_by_slug: dict[str, ValidationEvidence],
    config: AllocationConfig | None = None,
    returns_by_slug: dict[str, np.ndarray] | None = None,
) -> dict[str, float]:
    """Return normalized allocation weights for governance-live strategies only.

    ``returns_by_slug`` (per-strategy daily OOS returns) is required by the
    risk-based modes; when absent or insufficient those modes fail open to
    ``equal-live``.
    """
    config = config or AllocationConfig()
    live = {
        slug: state
        for slug, state in sorted(states_by_slug.items())
        if state.state is GovernanceState.LIVE
    }
    if not live:
        return {}
    if len(live) == 1:
        return {next(iter(live)): 1.0}

    if config.mode == "equal-live":
        raw = {slug: 1.0 for slug in live}
    elif config.mode == "dsr-weighted":
        raw = {}
        for slug in live:
            evidence = evidence_by_slug.get(slug)
            raw[slug] = max(float(evidence.deflated_sharpe), 0.0) if evidence else 0.0
    elif config.mode in _RISK_MODES:
        risk_raw = risk_based_raw_weights(
            returns_by_slug or {}, list(live), config.mode, config
        )
        # Fail open: any unmeasurable strategy / all-zero edge -> equal-live.
        raw = risk_raw if risk_raw is not None else {slug: 1.0 for slug in live}
    elif config.mode == "hrp":
        hrp_raw = hrp_raw_weights(returns_by_slug or {}, list(live), config)
        # Fail open (all-or-nothing): unmeasurable covariance -> equal-live.
        raw = hrp_raw if hrp_raw is not None else {slug: 1.0 for slug in live}
    else:
        raw = {slug: _evidence_score(evidence_by_slug.get(slug)) for slug in live}
    if sum(raw.values()) <= 0:
        raw = {slug: 1.0 for slug in live}
    return _normalize_with_cap_and_floor(
        raw,
        max_weight=config.max_weight,
        min_weight=config.min_weight,
    )


def strategy_risk(returns: np.ndarray, min_observations: int) -> tuple[float, float]:
    """Daily ``(mean, std)`` (sample std, ddof=1) of a return series.

    Returns ``(nan, nan)`` when the strategy is unmeasurable: fewer than
    ``min_observations`` finite returns, or a non-finite / non-positive std
    (a flat curve carries no risk signal we can weight on).
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < min_observations:
        return float("nan"), float("nan")
    std = float(r.std(ddof=1))
    if not math.isfinite(std) or std <= 0.0:
        return float("nan"), float("nan")
    return float(r.mean()), std


def risk_based_raw_weights(
    returns_by_slug: dict[str, np.ndarray],
    live_slugs: list[str],
    mode: str,
    config: AllocationConfig,
) -> dict[str, float] | None:
    """Pre-normalization risk weights for the live strategies, or ``None``.

    ``risk-parity`` => ``1/std`` (inverse vol). ``fractional-kelly`` =>
    ``clamp(fraction*mean/var, 0, cap)`` (reusing :func:`fractional_kelly`).

    All-or-nothing & fail-open: returns ``None`` (⇒ caller uses equal-live) if
    ANY live strategy is unmeasurable (missing/short/degenerate curve) or if every
    raw weight is ≤ 0. Never silently weights a subset.
    """
    raw: dict[str, float] = {}
    for slug in live_slugs:
        returns = returns_by_slug.get(slug)
        if returns is None:
            return None
        mean, std = strategy_risk(returns, config.min_observations)
        if not math.isfinite(std):
            return None
        if mode == "risk-parity":
            raw[slug] = 1.0 / std
        else:  # fractional-kelly
            raw[slug] = fractional_kelly(
                mean, std * std, config.kelly_fraction, config.kelly_cap
            )
    if sum(raw.values()) <= 0.0:
        return None
    return raw


def hrp_raw_weights(
    returns_by_slug: dict[str, np.ndarray],
    live_slugs: list[str],
    config: AllocationConfig,
) -> dict[str, float] | None:
    """Covariance-aware Hierarchical Risk Parity (Lopez de Prado) raw weights, or ``None``.

    Builds a strategy-level covariance from the trailing-aligned OOS return curves and runs
    HRP (reusing the proven strategy-level :func:`quant.strategies.risk_parity.hrp_weights`),
    so *correlated* strategies are diversified down — unlike the diagonal inverse-vol mode,
    which ignores cross-strategy correlation.

    All-or-nothing & fail-open (mirrors :func:`risk_based_raw_weights`): returns ``None``
    (⇒ caller uses equal-live) if any live strategy is missing/short, the trailing-aligned
    window is below ``config.min_observations``, any strategy is degenerate (non-finite or
    zero vol), or the resulting weights are non-finite / non-positive. Never weights a subset.
    """
    import pandas as pd

    series: list[np.ndarray] = []
    for slug in live_slugs:
        returns = returns_by_slug.get(slug)
        if returns is None:
            return None
        arr = np.asarray(returns, dtype=float)
        series.append(arr[np.isfinite(arr)])
    if len(series) < 2:
        return None
    common = min(arr.size for arr in series)
    if common < config.min_observations:
        return None
    # Trailing alignment: the OOS curves carry no dates here, so align on the shared tail.
    matrix = pd.DataFrame({slug: series[i][-common:] for i, slug in enumerate(live_slugs)})

    cov = matrix.cov()
    stds = np.sqrt(np.diag(cov.values))
    if not np.all(np.isfinite(stds)) or np.any(stds <= 0.0):
        return None  # a degenerate / flat curve carries no risk signal we can weight on
    # Correlation derived from the same covariance keeps the HRP distance matrix consistent.
    safe = np.where(stds > 0.0, stds, 1.0)
    corr_values = np.nan_to_num(cov.values / np.outer(safe, safe), nan=0.0)
    np.fill_diagonal(corr_values, 1.0)
    corr = pd.DataFrame(corr_values, index=cov.index, columns=cov.columns)

    from quant.strategies.risk_parity import hrp_weights

    weights = hrp_weights(cov, corr)
    if weights.empty or not np.all(np.isfinite(weights.to_numpy())) or float(weights.sum()) <= 0.0:
        return None
    return {slug: float(weights.get(slug, 0.0)) for slug in live_slugs}


def load_strategy_returns(
    evidence_by_slug: dict[str, ValidationEvidence], *, root: Path
) -> dict[str, np.ndarray]:
    """Load each strategy's walk-forward OOS daily returns from its parquet curve.

    Impure edge for the risk-based modes: reads ``evidence.walkforward_path``
    (``equity`` column → ``pct_change``). Best-effort — a slug whose parquet is
    missing or unreadable is simply omitted (which triggers the caller's
    equal-live fallback). No exception escapes.
    """
    import pandas as pd

    out: dict[str, np.ndarray] = {}
    for slug, evidence in evidence_by_slug.items():
        wf = evidence.walkforward_path
        candidates = [root / wf, root.parent / wf, Path(wf)]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            continue
        try:
            df = pd.read_parquet(path)
            equity = df["equity"].astype(float)
            returns = equity.pct_change().dropna().to_numpy(dtype=float)
        except Exception:
            continue
        if returns.size > 0:
            out[slug] = returns
    return out


def _evidence_score(evidence: ValidationEvidence | None) -> float:
    if evidence is None:
        return 0.0
    bootstrap = max(float(evidence.bootstrap_total_return_p05 or 0.0), 0.0)
    holdout = max(float(evidence.holdout_total_return or 0.0), 0.0)
    return max(evidence.deflated_sharpe, 0.0) + bootstrap + holdout


def _normalize_with_cap(raw: dict[str, float], *, max_weight: float) -> dict[str, float]:
    total = float(sum(max(v, 0.0) for v in raw.values()))
    if total <= 0:
        return {}
    weights = {slug: max(score, 0.0) / total for slug, score in raw.items()}
    if max_weight <= 0:
        return weights

    capped: dict[str, float] = {}
    remaining = dict(weights)
    remaining_weight = 1.0
    while remaining:
        over = {slug: w for slug, w in remaining.items() if w * remaining_weight > max_weight}
        if not over:
            subtotal = sum(remaining.values())
            for slug, w in remaining.items():
                capped[slug] = remaining_weight * w / subtotal
            break
        for slug in over:
            capped[slug] = max_weight
            remaining_weight -= max_weight
            remaining.pop(slug)
        subtotal = sum(remaining.values())
        remaining = {slug: w / subtotal for slug, w in remaining.items()} if subtotal > 0 else {}
    return {slug: capped[slug] for slug in sorted(capped)}


def _normalize_with_cap_and_floor(
    raw: dict[str, float],
    *,
    max_weight: float,
    min_weight: float,
) -> dict[str, float]:
    if min_weight <= 0 or len(raw) * min_weight > 1.0:
        return _normalize_with_cap(raw, max_weight=max_weight)

    weights = _normalize_with_cap(raw, max_weight=max_weight)
    if not weights:
        return weights

    below = {slug: min_weight for slug, weight in weights.items() if weight < min_weight}
    if not below:
        return weights

    locked_total = sum(below.values())
    free = {slug: weight for slug, weight in weights.items() if slug not in below}
    if not free:
        return {slug: 1.0 / len(weights) for slug in sorted(weights)}

    free_total = sum(free.values())
    if free_total <= 0:
        scaled = {slug: (1.0 - locked_total) / len(free) for slug in free}
    else:
        scaled = {slug: weight / free_total * (1.0 - locked_total) for slug, weight in free.items()}
    combined = {**below, **scaled}
    return _normalize_with_cap(combined, max_weight=max_weight)
