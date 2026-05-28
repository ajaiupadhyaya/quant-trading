"""Combined-book backtest: run all strategies into a single joint equity curve.

Spec §7.2: strategies run independently, attributed via client_order_id, but
share a single Alpaca account. For backtest accounting we treat them as
parallel sub-portfolios — each strategy gets ``allocation[slug] * total_equity``
and produces its own equity curve, and the combined equity is just the sum.

This sidesteps cross-strategy netting (which doesn't affect economics in a
zero-commission backtest) and keeps the per-strategy attribution clean. The
result is what spec §8 calls "the integration test" — the live paper-trade
equity should track this curve within a few percent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from quant.backtest.engine import BacktestConfig, BacktestResult, run_backtest
from quant.strategies.base import Strategy
from quant.util.logging import logger


@dataclass(frozen=True)
class CombinedResult:
    """Output of ``run_combined_book``."""

    equity_curve: pd.Series  # joint, daily
    returns: pd.Series  # joint daily simple returns
    trades: pd.DataFrame  # all per-strategy trades, with strategy_slug column
    per_strategy: dict[str, BacktestResult] = field(default_factory=dict)
    allocation: dict[str, float] = field(default_factory=dict)
    starting_equity: float = 0.0
    ending_equity: float = 0.0


def _equal_allocation(slugs: list[str]) -> dict[str, float]:
    n = max(len(slugs), 1)
    return {slug: 1.0 / n for slug in slugs}


def run_combined_book(
    strategies: dict[str, Strategy],
    bars_per_strategy: dict[str, pd.DataFrame],
    config: BacktestConfig,
    start: date,
    end: date,
    allocation: dict[str, float] | None = None,
) -> CombinedResult:
    """Run each strategy in its own sub-portfolio and sum the equity curves.

    Args:
        strategies: slug -> built Strategy instance (already loaded with bars).
        bars_per_strategy: slug -> wide-format bars for that strategy's universe.
        config: shared BacktestConfig. starting_equity is split via allocation.
        start, end: window edges (inclusive).
        allocation: slug -> fraction of total equity. Must sum to ~1.0.
            Defaults to equal across the supplied strategies.

    Returns:
        A CombinedResult whose equity_curve is the joint daily equity.
    """
    if set(strategies) != set(bars_per_strategy):
        raise ValueError(
            "strategies and bars_per_strategy must have identical keys; "
            f"got {sorted(strategies)} vs {sorted(bars_per_strategy)}"
        )

    slugs = sorted(strategies)
    alloc = dict(allocation) if allocation is not None else _equal_allocation(slugs)
    if not slugs:
        empty = pd.Series(dtype=float)
        return CombinedResult(
            equity_curve=empty,
            returns=empty,
            trades=pd.DataFrame(),
            per_strategy={},
            allocation={},
            starting_equity=config.starting_equity,
            ending_equity=config.starting_equity,
        )

    missing = set(slugs) - set(alloc)
    if missing:
        raise ValueError(f"allocation missing keys: {sorted(missing)}")
    total_alloc = sum(alloc.values())
    if not (0.99 <= total_alloc <= 1.01):
        raise ValueError(f"allocation must sum to ~1.0; got {total_alloc:.4f}")

    per_strategy: dict[str, BacktestResult] = {}
    trade_frames: list[pd.DataFrame] = []
    equity_frames: list[pd.Series] = []

    for slug in slugs:
        strat = strategies[slug]
        bars = bars_per_strategy[slug]
        slice_equity = config.starting_equity * alloc[slug]
        sub_config = BacktestConfig(
            starting_equity=slice_equity,
            slippage_bps=config.slippage_bps,
            commission_bps=config.commission_bps,
            annual_borrow_bps=config.annual_borrow_bps,
            annual_financing_bps=config.annual_financing_bps,
            execution=config.execution,
        )
        logger.info(
            "combined-book: running {} on {:.0f} equity ({:.1%} allocation)",
            slug,
            slice_equity,
            alloc[slug],
        )
        result = run_backtest(strategy=strat, bars=bars, config=sub_config, start=start, end=end)
        per_strategy[slug] = result
        if not result.equity_curve.empty:
            equity_frames.append(result.equity_curve.rename(slug))
        if not result.trades.empty:
            trade_frames.append(result.trades.assign(strategy_slug=slug))

    if not equity_frames:
        empty = pd.Series(dtype=float)
        return CombinedResult(
            equity_curve=empty,
            returns=empty,
            trades=pd.DataFrame(),
            per_strategy=per_strategy,
            allocation=alloc,
            starting_equity=config.starting_equity,
            ending_equity=config.starting_equity,
        )

    equity_panel = pd.concat(equity_frames, axis=1).sort_index()
    # Forward-fill each strategy's equity onto every shared date so dates with
    # only some strategies trading still get a stable joint mark-to-market.
    equity_panel = equity_panel.ffill().fillna(0.0)
    combined_equity = equity_panel.sum(axis=1)
    combined_equity.name = "equity"
    combined_returns = combined_equity.pct_change().fillna(0.0)
    combined_returns.name = "returns"

    combined_trades = (
        pd.concat(trade_frames, ignore_index=True).sort_values("date").reset_index(drop=True)
        if trade_frames
        else pd.DataFrame()
    )

    return CombinedResult(
        equity_curve=combined_equity,
        returns=combined_returns,
        trades=combined_trades,
        per_strategy=per_strategy,
        allocation=alloc,
        starting_equity=config.starting_equity,
        ending_equity=float(combined_equity.iloc[-1]),
    )
