"""PIT returns-overlay application of a hedge + baseline-vs-hedged comparison.

Mirrors quant/sizing/backtest.py. Observed-only: never touches live state.
"""

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
from quant.options.models import HedgeConfig, HedgeDecision
from quant.options.policy import build_hedge
from quant.strategies._common import annualize_vol

_TRADING_DAYS = 252
_FALLBACK_VOL = 0.15


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


def _vol_now(spy_ret_hist: list[float], config: HedgeConfig) -> float:
    arr = np.asarray(spy_ret_hist[-config.vol_lookback_days :], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return _FALLBACK_VOL
    daily = float(np.std(arr, ddof=1))
    ann = daily * float(np.sqrt(_TRADING_DAYS))
    return ann if ann > 1e-6 else _FALLBACK_VOL


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
    equity. PIT -- proven by a truncation-invariance test.
    """
    index = returns.index
    n = len(returns)
    r = returns.to_numpy(dtype=float)
    spy = spy_close.reindex(index).to_numpy(dtype=float)

    baseline_equity = np.cumprod(1.0 + np.nan_to_num(r))
    hedge_pnl_daily = np.zeros(n, dtype=float)

    decisions: list[HedgeDecision] = []
    held: HedgeDecision | None = None
    prev_value = 0.0  # per-unit structure value yesterday
    tenor_years = config.tenor_days / 365.0
    bars_to_expiry = max(1, round(config.tenor_days * (_TRADING_DAYS / 365.0)))

    book_ret_hist: list[float] = []
    spy_ret_hist: list[float] = []

    for t in range(n):
        spot = spy[t]
        # Append day t-1's realized returns so both histories reference the SAME
        # period (book return r[t-1] paired with the SPY return over t-2 -> t-1).
        # A one-day offset here silently collapses the book/SPY correlation and
        # drives estimated beta -- and thus the hedge size -- to zero.
        if t > 0:
            book_ret_hist.append(r[t - 1])
            prev_spy = spy[t - 2] if t >= 2 else 0.0
            cur_spy = spy[t - 1]
            spy_ret_hist.append((cur_spy / prev_spy - 1.0) if prev_spy > 0 else 0.0)

        is_roll = (t % config.roll_days == 0) or held is None

        if held is not None:
            bars_left = max(0, held.structure.expiry_index - t)
            t_left = bars_left / _TRADING_DAYS
        else:
            t_left = tenor_years

        if not (np.isfinite(spot) and spot > 0):
            continue

        if is_roll:
            label = _as_of_label(regime_labels, index[t - 1] if t > 0 else None)
            vol_now = _vol_now(spy_ret_hist, config)
            # realize close of the prior structure at today's spot before opening new
            if held is not None:
                close_val = held.structure.value(
                    spot, t_left, vol_now, config.risk_free, config.div_yield
                )
                hedge_pnl_daily[t] += held.contracts * (close_val - prev_value)
            book_value = float(baseline_equity[t])
            new_dec = build_hedge(
                spot,
                np.asarray(book_ret_hist, dtype=float),
                np.asarray(spy_ret_hist, dtype=float),
                label,
                config,
                book_value,
                expiry_index=t + bars_to_expiry,
            )
            hedge_pnl_daily[t] -= new_dec.premium  # pay premium for the new structure
            held = new_dec
            prev_value = new_dec.structure.value(
                spot, tenor_years, vol_now, config.risk_free, config.div_yield
            )
            decisions.append(new_dec)
        elif held is not None:
            vol_now = _vol_now(spy_ret_hist, config)
            cur_val = held.structure.value(
                spot, t_left, vol_now, config.risk_free, config.div_yield
            )
            hedge_pnl_daily[t] += held.contracts * (cur_val - prev_value)
            prev_value = cur_val

    hedge_pnl = pd.Series(hedge_pnl_daily, index=index, name="hedge_pnl")
    hedged_equity = baseline_equity + np.cumsum(hedge_pnl_daily)
    hedged_equity = np.maximum(hedged_equity, 1e-9)  # guard div-by-zero
    hedged_ret_vals = np.empty(n, dtype=float)
    if n > 0:
        hedged_ret_vals[0] = hedged_equity[0] - 1.0
        hedged_ret_vals[1:] = hedged_equity[1:] / hedged_equity[:-1] - 1.0
    hedged = pd.Series(hedged_ret_vals, index=index, name="hedged_returns")
    return hedged, HedgeLedger(decisions=decisions, hedge_pnl=hedge_pnl)


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
