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
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from quant.backtest.engine import BacktestConfig
from quant.data.bars import BarRequest, get_bars
from quant.execution.alpaca import AlpacaClient
from quant.live.recon import ReconciliationReport, reconcile
from quant.live.recon_render import render_markdown
from quant.util.logging import logger

REPO_ROOT = Path(__file__).resolve().parents[1]
TRADES_PATH = REPO_ROOT / "data" / "live" / "trades.parquet"
REPORT_DIR = REPO_ROOT / "docs" / "live-recon"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconcile live Alpaca fills vs backtest model.")
    parser.add_argument("--since", type=date.fromisoformat, default=None,
                        help="Start date (inclusive). Default: 7 days before --until.")
    parser.add_argument("--until", type=date.fromisoformat, default=date.today(),
                        help="End date (inclusive). Default: today.")
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
    trades = trades_all.loc[mask].copy()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / f"{until.isoformat()}.md"

    if trades.empty:
        report = ReconciliationReport(
            since=since, until=until,
            modeled_slippage_bps=BacktestConfig().slippage_bps,
            rows=[],
        )
        report_path.write_text(render_markdown(report))
        print(f"Wrote empty report: {report_path}")
        return 0

    try:
        client = AlpacaClient()
        orders = client.list_orders(since=since, until=until)
    except Exception as exc:
        print(f"Alpaca API error: {exc}", file=sys.stderr)
        return 1

    report = reconcile(
        trades=trades,
        orders=orders,
        bar_fetcher=_make_bar_fetcher(),
        modeled_slippage_bps=BacktestConfig().slippage_bps,
        since=since,
        until=until,
    )

    report_path.write_text(render_markdown(report))
    print(f"Wrote {report_path} — {len(report.rows)} reconciled rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
