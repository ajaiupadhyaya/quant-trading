"""Backtest engine.

Daily-frequency, deterministic, single-pass. At each rebalance day the strategy
proposes target positions; the engine reconciles vs current and executes the
diff on the next bar (or the same bar's close, depending on config). Slippage
and commission are charged per trade. Equity is marked to market on every bar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

Side = Literal["buy", "sell"]


@dataclass(frozen=True)
class BacktestConfig:
    """Engine configuration. All defaults are intentional — change with care."""

    starting_equity: float = 100_000.0
    slippage_bps: float = 5.0
    commission_bps: float = 0.0
    execution: Literal["next_open", "close"] = "next_open"


@dataclass(frozen=True)
class FillReport:
    """The result of applying costs to a single order."""

    fill_price: float
    slippage_cost: float
    commission_cost: float


@dataclass(frozen=True)
class BacktestResult:
    """Output of run_backtest."""

    equity_curve: pd.Series  # daily, indexed by date
    returns: pd.Series  # daily simple returns, indexed by date
    positions: pd.DataFrame  # rows=date, cols=symbol, values=shares
    trades: (
        pd.DataFrame
    )  # columns: date, symbol, side, qty, fill_price, slippage_cost, commission_cost, strategy_slug
    config: BacktestConfig
    starting_equity: float
    ending_equity: float
    metadata: dict[str, object] = field(default_factory=dict)


def apply_costs(qty: int, mid_price: float, side: Side, config: BacktestConfig) -> FillReport:
    """Move the mid-price by slippage and compute commission as bps of notional.

    Buy: fill_price = mid * (1 + slippage_bps / 1e4)
    Sell: fill_price = mid * (1 - slippage_bps / 1e4)
    Commission: |qty| * fill_price * commission_bps / 1e4
    """
    if side not in ("buy", "sell"):
        raise ValueError(f"Unknown side {side!r}; expected 'buy' or 'sell'")
    if qty == 0:
        return FillReport(fill_price=mid_price, slippage_cost=0.0, commission_cost=0.0)

    slip = config.slippage_bps / 1e4
    sign = +1.0 if side == "buy" else -1.0
    fill_price = mid_price * (1.0 + sign * slip)
    slippage_cost = abs(qty) * abs(fill_price - mid_price)
    commission_cost = abs(qty) * fill_price * config.commission_bps / 1e4
    return FillReport(
        fill_price=fill_price,
        slippage_cost=slippage_cost,
        commission_cost=commission_cost,
    )


# Forward declaration so the package __init__ can import the name even before
# the run loop lands in Task 6. The Task-6 commit replaces this stub.
def run_backtest(*args: object, **kwargs: object) -> BacktestResult:  # pragma: no cover
    raise NotImplementedError("run_backtest implemented in Task 6")
