"""Richer read-only context for the Claude decision-maker (Phase A).

Gathers the slowly-changing "situational" state a human quant would review before
the day — the current market regime, per-strategy validation evidence + governance
state, capital allocation, recent execution quality, and a macro snapshot — into
one structured, serializable dataclass.

Every reader is best-effort and FAIL-OPEN: a missing or unreadable artifact
degrades to ``None``/empty and never raises. This module performs NO trading and
changes NO state; it only reads. It is the INPUT to ``quant.analyst.advisor``.
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from quant.util.logging import logger


@dataclass(frozen=True)
class RegimeSnapshot:
    """Latest point-in-time regime posterior + canonical label."""

    asof: str | None
    label: str | None
    p_calm: float | None
    p_choppy: float | None
    p_crisis: float | None


@dataclass(frozen=True)
class StrategyEvidence:
    """Per-strategy governance state + the validation evidence behind it."""

    slug: str
    state: str | None
    deflated_sharpe: float | None
    probabilistic_sharpe: float | None
    bootstrap_lower: float | None
    gates_passed: int | None
    gates_total: int | None
    validation_age_days: int | None
    reason: str | None


@dataclass(frozen=True)
class AnalystContext:
    """One day's read-only situational context for the advisor."""

    asof: date
    regime: RegimeSnapshot | None = None
    allocation: dict[str, float] = field(default_factory=dict)
    evidence: list[StrategyEvidence] = field(default_factory=list)
    recon: dict[str, Any] | None = None
    macro: dict[str, float] = field(default_factory=dict)
    portfolio_risk: Any | None = None  # quant.risk.PortfolioRisk | None (lazy to avoid cycle)
    signals: Any | None = None  # quant.research.signals.MarketSignals | None (latest logged)


# --- best-effort readers (each fail-open) ----------------------------------


def _read_regime(data_dir: Path) -> RegimeSnapshot | None:
    path = data_dir / "regime" / "regime_series.parquet"
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        if df.empty:
            return None
        last = df.iloc[-1]

        def _g(col: str) -> float | None:
            return float(last[col]) if col in df.columns else None

        asof_val = df.index[-1]
        return RegimeSnapshot(
            asof=str(getattr(asof_val, "date", lambda: asof_val)()),
            label=str(last["label"]) if "label" in df.columns else None,
            p_calm=_g("p_calm"),
            p_choppy=_g("p_choppy"),
            p_crisis=_g("p_crisis"),
        )
    except Exception as exc:  # fail-open
        logger.info("analyst.context: regime read skipped ({!r})", exc)
        return None


def _read_evidence(data_dir: Path) -> list[StrategyEvidence]:
    try:
        from quant.governance.store import (
            load_strategy_states,
            load_validation_manifest,
            strategy_states_path,
            validation_manifest_path,
        )

        evidence = load_validation_manifest(validation_manifest_path(data_dir))
        states = load_strategy_states(strategy_states_path(data_dir))
    except Exception as exc:  # fail-open
        logger.info("analyst.context: evidence read skipped ({!r})", exc)
        return []

    out: list[StrategyEvidence] = []
    slugs = sorted(set(evidence) | set(states))
    for slug in slugs:
        ev = evidence.get(slug)
        st = states.get(slug)
        gates_passed = gates_total = None
        ds = ps = boot = None
        if ev is not None:
            gate_flags = [
                getattr(ev, name, None)
                for name in (
                    "gate_deflated_sharpe",
                    "gate_probabilistic_sharpe",
                    "gate_bootstrap_lower",
                    "gate_regime",
                    "gate_holdout",
                )
            ]
            present = [bool(g) for g in gate_flags if g is not None]
            if present:
                gates_passed = sum(present)
                gates_total = len(present)
            ds = _opt_float(getattr(ev, "deflated_sharpe", None))
            ps = _opt_float(getattr(ev, "probabilistic_sharpe", None))
            boot = _opt_float(getattr(ev, "bootstrap_total_return_p05", None))
        state_str = None
        age = reason = None
        if st is not None:
            raw_state = getattr(st, "state", None)
            state_str = getattr(raw_state, "value", None) or (
                str(raw_state) if raw_state is not None else None
            )
            age = getattr(st, "validation_age_days", None)
            reason = getattr(st, "reason", None) or None
        out.append(
            StrategyEvidence(
                slug=slug,
                state=state_str,
                deflated_sharpe=ds,
                probabilistic_sharpe=ps,
                bootstrap_lower=boot,
                gates_passed=gates_passed,
                gates_total=gates_total,
                validation_age_days=age,
                reason=reason,
            )
        )
    return out


def _read_allocation(data_dir: Path) -> dict[str, float]:
    try:
        from quant.governance.store import allocation_path, load_allocation

        return {k: float(v) for k, v in load_allocation(allocation_path(data_dir)).items()}
    except Exception as exc:  # fail-open
        logger.info("analyst.context: allocation read skipped ({!r})", exc)
        return {}


def _read_recon(data_dir: Path) -> dict[str, Any] | None:
    """Latest persisted reconciliation summary (slippage/fill metrics), if any."""
    recon_dir = data_dir.parent / "docs" / "live-recon"
    candidates = sorted(recon_dir.glob("*.json")) if recon_dir.exists() else []
    if not candidates:
        return None
    try:
        return json.loads(candidates[-1].read_text(encoding="utf-8"))  # type: ignore[no-any-return]
    except Exception as exc:  # fail-open
        logger.info("analyst.context: recon read skipped ({!r})", exc)
        return None


def _read_macro() -> dict[str, float]:
    """VIX + 10y yield from the FRED cache. Best-effort; never hits the network hard."""
    out: dict[str, float] = {}
    try:
        from quant.data import macro as macro_mod

        for key, fn_name in (("vix", "vix"), ("ust10y", "tenyear_yield")):
            try:
                series = getattr(macro_mod, fn_name)()
                if series is not None and len(series) > 0:
                    out[key] = float(series.iloc[-1])
            except Exception:  # one series failing must not sink the rest
                continue
    except Exception as exc:  # fail-open
        logger.info("analyst.context: macro read skipped ({!r})", exc)
    return out


def _opt_float(v: Any) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _portfolio_risk(
    positions: dict[str, int] | None, equity: float | None, asof: date
) -> Any | None:
    if not positions or not equity or equity <= 0:
        return None
    try:
        from quant.risk.portfolio import live_portfolio_risk

        return live_portfolio_risk(positions, float(equity), asof=asof)
    except Exception as exc:  # fail-open
        logger.info("analyst.context: portfolio risk skipped ({!r})", exc)
        return None


def _read_signals(data_dir: Path, asof: date) -> Any | None:
    """Latest logged quant signal battery (no recompute/network). Fail-open."""
    try:
        from quant.research.signals import read_latest_signals, signals_path

        return read_latest_signals(signals_path(data_dir))
    except Exception as exc:  # fail-open
        logger.info("analyst.context: signals read skipped ({!r})", exc)
        return None


def gather_analyst_context(
    data_dir: Path,
    asof: date,
    *,
    include_macro: bool = True,
    positions: dict[str, int] | None = None,
    equity: float | None = None,
) -> AnalystContext:
    """Assemble the day's read-only context. Each piece is best-effort/fail-open.

    ``positions``/``equity`` (the live Alpaca snapshot, when the caller has it)
    enable the portfolio-risk view (VaR/CVaR/beta of the current book)."""
    return AnalystContext(
        asof=asof,
        regime=_read_regime(data_dir),
        allocation=_read_allocation(data_dir),
        evidence=_read_evidence(data_dir),
        recon=_read_recon(data_dir),
        macro=_read_macro() if include_macro else {},
        portfolio_risk=_portfolio_risk(positions, equity, asof),
        signals=_read_signals(data_dir, asof),
    )


def render_context(ctx: AnalystContext) -> str:
    """Deterministic plain-text rendering — the LLM input and a human-readable record."""
    lines: list[str] = [f"As-of: {ctx.asof.isoformat()}"]

    if ctx.regime is not None and ctx.regime.label is not None:
        post = []
        for name, val in (
            ("calm", ctx.regime.p_calm),
            ("choppy", ctx.regime.p_choppy),
            ("crisis", ctx.regime.p_crisis),
        ):
            if val is not None:
                post.append(f"{name} {val:.0%}")
        post_str = f" (posterior: {', '.join(post)})" if post else ""
        lines.append(f"Regime: {ctx.regime.label}{post_str}")
    else:
        lines.append("Regime: unavailable")

    if ctx.macro:
        macro_str = ", ".join(f"{k}={v:,.2f}" for k, v in sorted(ctx.macro.items()))
        lines.append(f"Macro: {macro_str}")

    if ctx.allocation:
        alloc = ", ".join(f"{k} {v:.0%}" for k, v in sorted(ctx.allocation.items()))
        lines.append(f"Capital allocation: {alloc}")

    if ctx.evidence:
        lines.append("Strategy evidence:")
        for e in ctx.evidence:
            bits: list[str] = [f"  - {e.slug}: {e.state or 'state?'}"]
            if e.gates_passed is not None and e.gates_total is not None:
                bits.append(f"gates {e.gates_passed}/{e.gates_total}")
            if e.deflated_sharpe is not None:
                bits.append(f"DSR {e.deflated_sharpe:.2f}")
            if e.probabilistic_sharpe is not None:
                bits.append(f"PSR {e.probabilistic_sharpe:.2f}")
            if e.bootstrap_lower is not None:
                bits.append(f"boot_p05 {e.bootstrap_lower:+.2%}")
            if e.validation_age_days is not None:
                bits.append(f"age {e.validation_age_days}d")
            line = bits[0] + (" | " + ", ".join(bits[1:]) if len(bits) > 1 else "")
            if e.reason:
                line += f" — {e.reason}"
            lines.append(line)

    if ctx.portfolio_risk is not None:
        with contextlib.suppress(Exception):  # render is best-effort
            lines.append("Portfolio risk: " + ctx.portfolio_risk.render())

    if ctx.signals is not None:
        with contextlib.suppress(Exception):  # render is best-effort
            from quant.research.signals import render_signals

            lines.append(render_signals(ctx.signals))

    if ctx.recon:
        msb = ctx.recon.get("mean_slippage_bps")
        n = ctx.recon.get("n_fills") or ctx.recon.get("n")
        recon_bits = []
        if msb is not None:
            recon_bits.append(f"mean slippage {float(msb):+.1f}bps")
        if n is not None:
            recon_bits.append(f"{n} fills")
        if recon_bits:
            lines.append("Execution (last recon): " + ", ".join(recon_bits))

    return "\n".join(lines)
