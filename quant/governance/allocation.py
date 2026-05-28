"""Deterministic capital allocation for live-governed strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from quant.governance.models import GovernanceState, StrategyState, ValidationEvidence

AllocationMode = Literal["equal-live", "dsr-weighted", "capped-evidence-score"]


@dataclass(frozen=True)
class AllocationConfig:
    mode: AllocationMode = "equal-live"
    max_weight: float = 0.40
    min_weight: float = 0.05


def allocate_capital(
    states_by_slug: dict[str, StrategyState],
    *,
    evidence_by_slug: dict[str, ValidationEvidence],
    config: AllocationConfig | None = None,
) -> dict[str, float]:
    """Return normalized allocation weights for governance-live strategies only."""
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
    else:
        raw = {slug: _evidence_score(evidence_by_slug.get(slug)) for slug in live}
    if sum(raw.values()) <= 0:
        raw = {slug: 1.0 for slug in live}
    return _normalize_with_cap_and_floor(
        raw,
        max_weight=config.max_weight,
        min_weight=config.min_weight,
    )


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
