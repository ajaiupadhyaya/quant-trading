"""On-demand live reconciliation runner.

Joins ``data/live/trades.parquet`` (intent) with Alpaca's order history
(outcome), computes per-fill slippage / timing / fidelity, and writes a
dated Markdown report into ``docs/live-recon/YYYY-MM-DD.md``.

The report file is written to disk only; commit it by hand after review.

Usage::

    uv run python scripts/reconcile_live.py
    uv run python scripts/reconcile_live.py --since 2026-05-22 --until 2026-05-26
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from datetime import UTC, date, timedelta
from pathlib import Path

import pandas as pd

from quant.backtest.engine import BacktestConfig
from quant.data.bars import BarRequest, get_bars
from quant.execution.alpaca import AlpacaClient, OrderRow
from quant.live.recon import ReconciliationReport, reconcile
from quant.live.recon_render import render_markdown
from quant.util.config import Settings
from quant.util.logging import logger

REPO_ROOT = Path(__file__).resolve().parents[1]
TRADES_PATH = REPO_ROOT / "data" / "live" / "trades.parquet"
REPORT_DIR = REPO_ROOT / "docs" / "live-recon"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconcile live Alpaca fills vs backtest model.")
    parser.add_argument(
        "--since",
        type=date.fromisoformat,
        default=None,
        help="Start date (inclusive). Default: 7 days before --until.",
    )
    parser.add_argument(
        "--until",
        type=date.fromisoformat,
        default=date.today(),
        help="End date (inclusive). Default: today.",
    )
    return parser.parse_args()


def _make_bar_fetcher() -> Callable[[str, date], float | None]:
    """Return a cached bar-fetcher that returns the prior-day close.

    ``get_bars`` returns a wide DataFrame with MultiIndex columns of the form
    ``(symbol, field)``.  We extract the ``close`` level for the requested
    symbol directly via the tuple key ``(symbol, "close")``.
    """
    cache: dict[tuple[str, date], float | None] = {}

    def fetch(symbol: str, asof: date) -> float | None:
        key = (symbol, asof)
        if key in cache:
            return cache[key]
        try:
            df = get_bars(BarRequest(symbols=[symbol], start=asof, end=asof))
            if df.empty or (symbol, "close") not in df.columns:
                cache[key] = None
                return None
            close_series = df[(symbol, "close")].dropna()
            if close_series.empty:
                cache[key] = None
                return None
            close = float(close_series.iloc[-1])
            cache[key] = close
            return close
        except Exception as exc:
            logger.warning("bar fetch failed for {} @ {}: {}", symbol, asof, exc)
            cache[key] = None
            return None

    return fetch


def _make_mid_fetcher(settings: Settings) -> Callable[[OrderRow], float | None]:
    """Return a best-effort fill-minute OHLC midpoint fetcher from Alpaca bars."""
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
    except Exception as exc:
        logger.warning("alpaca data imports unavailable for mid-price lookup: {}", exc)

        def unavailable(_: OrderRow) -> float | None:
            return None

        return unavailable

    client = StockHistoricalDataClient(
        api_key=settings.alpaca_api_key,
        secret_key=settings.alpaca_secret_key,
    )
    cache: dict[tuple[str, str], float | None] = {}

    def fetch(order: OrderRow) -> float | None:
        if order.filled_at is None:
            return None
        filled_at = order.filled_at.astimezone(UTC)
        minute = filled_at.replace(second=0, microsecond=0)
        key = (order.symbol, minute.isoformat())
        if key in cache:
            return cache[key]
        try:
            req = StockBarsRequest(
                symbol_or_symbols=[order.symbol],
                timeframe=TimeFrame.Minute,
                start=minute,
                end=minute + timedelta(minutes=1),
            )
            bars = client.get_stock_bars(req)
            raw: pd.DataFrame = bars.df  # type: ignore[union-attr]
            if raw.empty:
                cache[key] = None
                return None
            if isinstance(raw.index, pd.MultiIndex):
                raw = raw.xs(order.symbol, level=0)
            row = raw.iloc[0]
            high = float(row["high"])
            low = float(row["low"])
            cache[key] = (high + low) / 2.0
            return cache[key]
        except Exception as exc:
            logger.warning("mid-price fetch failed for {} @ {}: {}", order.symbol, minute, exc)
            cache[key] = None
            return None

    return fetch


def main() -> int:
    args = _parse_args()
    until: date = args.until
    since: date = args.since or (until - timedelta(days=7))

    if not TRADES_PATH.exists():
        print(f"No trades.parquet at {TRADES_PATH}; nothing to reconcile.", file=sys.stderr)
        return 0

    trades_all = pd.read_parquet(TRADES_PATH)
    trades_all["date"] = pd.to_datetime(trades_all["date"]).dt.date
    mask = (trades_all["date"] >= since) & (trades_all["date"] <= until)
    trades = trades_all.loc[mask & ~trades_all["dry_run"]].copy()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / f"{until.isoformat()}.md"

    if trades.empty:
        report = ReconciliationReport(
            since=since,
            until=until,
            modeled_slippage_bps=BacktestConfig().slippage_bps,
            rows=[],
        )
        report_path.write_text(render_markdown(report))
        print(f"Wrote empty report: {report_path}")
        return 0

    try:
        settings = Settings()  # type: ignore[call-arg]
        client = AlpacaClient(settings=settings)
        orders = client.list_orders(since=since, until=until)
    except Exception as exc:
        print(f"Alpaca API error: {exc}", file=sys.stderr)
        return 1

    report = reconcile(
        trades=trades,
        orders=orders,
        bar_fetcher=_make_bar_fetcher(),
        mid_fetcher=_make_mid_fetcher(settings),
        modeled_slippage_bps=BacktestConfig().slippage_bps,
        since=since,
        until=until,
    )

    report_path.write_text(render_markdown(report))
    print(f"Wrote {report_path} — {len(report.rows)} reconciled rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
