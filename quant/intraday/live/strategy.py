"""Intraday mean-reversion proof-of-life strategy. Implements the shared
IntradayStrategy protocol so it could also be driven by the existing simulator.
Economic rationale: very short-horizon mean reversion in liquid index ETFs from
microstructure noise / liquidity provision. Assumptions: no persistent intraday
drift over the lookback. How it fails: trends/news regimes (it fades the move) and
spread/slippage eating the small edge — which is why the loop also flattens by close
and the sleeve is tightly capped."""

from __future__ import annotations

import statistics
from collections import defaultdict, deque

from quant.intraday.data.events import Event, QuoteBar
from quant.intraday.live.config import SleeveConfig
from quant.intraday.strategy import Order, OrderType, Side, StrategyContext


class MeanReversionStrategy:
    """z-score fade on a rolling window of mids. Reusable under IntradayStrategy."""

    def __init__(self, config: SleeveConfig, *, unit_shares: int = 10) -> None:
        self._cfg = config
        self._unit = unit_shares
        self._mids: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=config.mean_reversion_lookback)
        )

    def on_event(self, event: Event, ctx: StrategyContext) -> list[Order]:
        if not isinstance(event, QuoteBar):
            return []  # this strategy trades off NBBO mids only
        sym = event.symbol
        window = self._mids[sym]
        # Compute z-score against the *existing* history before appending the new
        # event; this avoids look-ahead bias and floating-point boundary issues when
        # the window contains a constant run (sd=0 → pure direction signal).
        if len(window) >= self._cfg.mean_reversion_lookback:
            mu = statistics.fmean(window)
            sd = statistics.pstdev(window)
            mid = event.mid
            if sd == 0.0:
                z: float = 0.0 if mid == mu else (1.0 if mid > mu else -1.0) * 1e9
            else:
                z = (mid - mu) / sd
        else:
            z = 0.0  # not enough history yet
        window.append(event.mid)
        if len(window) < self._cfg.mean_reversion_lookback:
            return []
        pos = ctx.position(sym)
        # Exit first: if we hold and have reverted inside the exit band, flatten.
        if pos != 0 and abs(z) <= self._cfg.exit_z:
            side = Side.BUY if pos < 0 else Side.SELL
            return [Order(symbol=sym, side=side, qty=abs(pos), type=OrderType.MARKET)]
        # Entry: fade a large deviation (only if flat).
        if pos == 0 and abs(z) >= self._cfg.entry_z:
            side = Side.SELL if z > 0 else Side.BUY
            return [Order(symbol=sym, side=side, qty=self._unit, type=OrderType.MARKET)]
        return []
