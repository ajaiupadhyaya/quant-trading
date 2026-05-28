"""Backtest engine.

Daily-frequency, deterministic, single-pass. At each rebalance day the strategy
proposes target positions; the engine reconciles vs current and executes the
diff on the next bar (or the same bar's close, depending on config). Slippage
and commission are charged per trade. Equity is marked to market on every bar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Literal

import pandas as pd

from quant.backtest.financing import financing_charge

if TYPE_CHECKING:
    from quant.strategies.base import Strategy

Side = Literal["buy", "sell"]


@dataclass(frozen=True)
class BacktestConfig:
    """Engine configuration. All defaults are intentional — change with care."""

    starting_equity: float = 100_000.0
    slippage_bps: float = 5.0
    commission_bps: float = 0.0
    # Financing (gap #2, slice 2a): short borrow fee + margin-debit financing,
    # accrued daily (actual/365). On by default. annual_financing_bps is a flat
    # approximation of the broker call rate and only bites under >1x gross.
    annual_borrow_bps: float = 50.0
    annual_financing_bps: float = 200.0
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


def run_backtest(
    strategy: Strategy,
    bars: pd.DataFrame,
    config: BacktestConfig,
    start: date,
    end: date,
) -> BacktestResult:
    """Simulate ``strategy`` over ``bars`` restricted to ``[start, end]``.

    ``bars`` must be a wide DataFrame with MultiIndex columns ``(symbol, field)``
    and a DatetimeIndex. ``field`` must include at least ``open`` and ``close``.

    Algorithm per bar:
      1. If ``execution == "next_open"``, fill any orders queued from yesterday's
         rebalance at today's open (slipped).
      2. Mark to market on today's close → record equity + position snapshot.
      3. If today is a rebalance day, diff target vs current positions into orders.
         For ``execution == "close"`` fill immediately at today's close; otherwise
         queue for tomorrow's open.
    """
    from quant.backtest.calendar import is_rebalance_day

    trades_columns: list[str] = [
        "date",
        "symbol",
        "side",
        "qty",
        "fill_price",
        "slippage_cost",
        "commission_cost",
        "strategy_slug",
    ]

    # Slice the history to the requested window.
    mask = (bars.index >= pd.Timestamp(start)) & (bars.index <= pd.Timestamp(end))
    history = pd.DatetimeIndex(bars.index[mask])

    if len(history) == 0:
        return BacktestResult(
            equity_curve=pd.Series(dtype=float, name="equity"),
            returns=pd.Series(dtype=float, name="returns"),
            positions=pd.DataFrame(),
            trades=pd.DataFrame(columns=trades_columns),
            config=config,
            starting_equity=config.starting_equity,
            ending_equity=config.starting_equity,
            metadata={
                "borrow_cost": 0.0,
                "margin_financing_cost": 0.0,
                "financing_cost_total": 0.0,
            },
        )

    cash: float = config.starting_equity
    positions: dict[str, int] = {}
    equity_records: list[float] = []
    position_records: list[dict[str, int]] = []
    trade_records: list[dict[str, object]] = []

    # Pending orders queued by a prior bar's rebalance (only used when execution == "next_open").
    pending: list[tuple[str, int, Side]] = []  # (symbol, qty, side)

    prev_ts: pd.Timestamp | None = None
    borrow_total: float = 0.0
    margin_financing_total: float = 0.0

    def _execute_fill(ts: pd.Timestamp, sym: str, qty: int, side: Side, mid: float) -> None:
        nonlocal cash
        fill = apply_costs(qty=qty, mid_price=mid, side=side, config=config)
        notional = qty * fill.fill_price
        if side == "buy":
            cash -= notional + fill.commission_cost
            positions[sym] = positions.get(sym, 0) + qty
        else:
            cash += notional - fill.commission_cost
            positions[sym] = positions.get(sym, 0) - qty
        trade_records.append(
            {
                "date": ts,
                "symbol": sym,
                "side": side,
                "qty": qty,
                "fill_price": fill.fill_price,
                "slippage_cost": fill.slippage_cost,
                "commission_cost": fill.commission_cost,
                "strategy_slug": strategy.spec.slug,
            }
        )

    for ts in history:
        asof: date = ts.date()

        # 0. Accrue overnight financing on positions/cash carried from the prior
        #    bar, priced at the PRIOR bar's close (PIT, no lookahead).
        if prev_ts is not None:
            prior_close = {
                sym: float(bars[(sym, "close")].loc[prev_ts])
                for sym in positions
                if (sym, "close") in bars.columns
            }
            charge = financing_charge(
                positions=positions,
                prior_close=prior_close,
                cash=cash,
                days_elapsed=(ts - prev_ts).days,
                annual_borrow_bps=config.annual_borrow_bps,
                annual_financing_bps=config.annual_financing_bps,
            )
            cash -= charge.total
            borrow_total += charge.borrow_cost
            margin_financing_total += charge.margin_financing_cost
        prev_ts = ts

        # 1. Execute pending fills on today's open (if any from prior bar's rebalance).
        if pending:
            for sym, qty, side in pending:
                if (sym, "open") not in bars.columns:
                    continue
                mid = float(bars[(sym, "open")].loc[ts])
                _execute_fill(ts, sym, qty, side, mid)
            pending = []

        # 2. Mark-to-market on today's close.
        equity = cash
        for sym, qty in positions.items():
            if qty != 0 and (sym, "close") in bars.columns:
                equity += qty * float(bars[(sym, "close")].loc[ts])
        equity_records.append(equity)
        position_records.append(dict(positions))

        # 3. Rebalance decision.
        if is_rebalance_day(asof, strategy.spec.rebalance_frequency, history):
            try:
                target = strategy.target_positions(asof, equity)
            except Exception:
                # Any strategy error halts trading for the day; equity is still marked-to-market.
                target = {}

            new_orders: list[tuple[str, int, Side]] = []
            symbols_to_consider = sorted(set(target) | set(positions))
            for sym in symbols_to_consider:
                tgt = int(target.get(sym, 0))
                cur = positions.get(sym, 0)
                delta = tgt - cur
                if delta == 0:
                    continue
                if (cur > 0 and tgt < 0) or (cur < 0 and tgt > 0):
                    # Zero-crossing: flatten current then reopen on the other side.
                    flatten_side: Side = "sell" if cur > 0 else "buy"
                    new_orders.append((sym, abs(cur), flatten_side))
                    open_side: Side = "buy" if tgt > 0 else "sell"
                    new_orders.append((sym, abs(tgt), open_side))
                else:
                    side_dir: Side = "buy" if delta > 0 else "sell"
                    new_orders.append((sym, abs(delta), side_dir))

            if config.execution == "close":
                for sym, qty, side in new_orders:
                    if (sym, "close") not in bars.columns:
                        continue
                    mid = float(bars[(sym, "close")].loc[ts])
                    _execute_fill(ts, sym, qty, side, mid)
            else:
                pending = new_orders

    equity_curve = pd.Series(equity_records, index=history, name="equity")
    returns = equity_curve.pct_change().fillna(0.0)
    returns.name = "returns"

    positions_df = pd.DataFrame(position_records, index=history).fillna(0).astype(int)

    trades_df = pd.DataFrame(trade_records)
    if trades_df.empty:
        trades_df = pd.DataFrame(columns=trades_columns)

    ending_equity = float(equity_curve.iloc[-1]) if len(equity_curve) else config.starting_equity

    return BacktestResult(
        equity_curve=equity_curve,
        returns=returns,
        positions=positions_df,
        trades=trades_df,
        config=config,
        starting_equity=config.starting_equity,
        ending_equity=ending_equity,
        metadata={
            "borrow_cost": borrow_total,
            "margin_financing_cost": margin_financing_total,
            "financing_cost_total": borrow_total + margin_financing_total,
        },
    )
