"""HTML tear-sheet writer.

Renders a self-contained HTML report (charts embedded as base64 PNGs) for a
walk-forward result. Also writes the OOS equity curve as parquet and the
per-window chosen params as JSON to the same directory.
"""

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use("Agg")  # headless rendering; no GUI required
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from quant.backtest.combined import CombinedResult
from quant.backtest.metrics import (
    cagr,
    max_drawdown,
    sharpe,
    sortino,
    total_return,
    win_rate,
)
from quant.backtest.walkforward import WalkforwardResult

if TYPE_CHECKING:
    from quant.backtest.validation import ValidationReport

_TEMPLATE_DIR = Path(__file__).parent / "templates"


@dataclass(frozen=True)
class _MetricsBundle:
    total_return: float
    cagr: float
    sharpe: float
    sortino: float
    max_drawdown: float
    win_rate: float
    n_trades: int
    starting_equity: float
    ending_equity: float


def _json_safe(v: object) -> object:
    """Best-effort coercion of param values into JSON-serializable shapes."""
    if isinstance(v, tuple):
        return [_json_safe(x) for x in v]
    if isinstance(v, list):
        return [_json_safe(x) for x in v]
    if isinstance(v, str | int | float | bool) or v is None:
        return v
    return str(v)


def _fig_to_base64(fig: Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _equity_chart(equity: pd.Series) -> str:
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(equity.index, np.asarray(equity.values), color="#1a3a8f", linewidth=1.2)
    ax.set_ylabel("Equity ($)")
    ax.set_title("OOS Equity Curve")
    ax.grid(True, alpha=0.3)
    return _fig_to_base64(fig)


def _drawdown_chart(equity: pd.Series) -> str:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    dd_values = np.asarray(dd.values)
    fig, ax = plt.subplots(figsize=(9, 2.5))
    ax.fill_between(dd.index, dd_values, 0, color="#c0392b", alpha=0.4)
    ax.plot(dd.index, dd_values, color="#c0392b", linewidth=0.8)
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_title("Drawdown")
    ax.grid(True, alpha=0.3)
    return _fig_to_base64(fig)


def _monthly_chart(returns: pd.Series) -> str:
    if len(returns) == 0:
        fig, ax = plt.subplots(figsize=(9, 2.0))
        ax.text(0.5, 0.5, "no monthly data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return _fig_to_base64(fig)

    monthly = (1.0 + returns).resample("ME").prod() - 1.0
    monthly_idx = pd.DatetimeIndex(monthly.index)
    pivot = pd.DataFrame(
        {"year": monthly_idx.year, "month": monthly_idx.month, "ret": np.asarray(monthly.values)}
    ).pivot(index="year", columns="month", values="ret")
    pivot = pivot.reindex(columns=range(1, 13))

    fig, ax = plt.subplots(figsize=(9, max(2.0, 0.35 * len(pivot))))
    cmap = plt.get_cmap("RdYlGn")
    pivot_arr = np.asarray(pivot.values, dtype=float)
    vmax = float(np.nanmax(np.abs(pivot_arr))) if pivot.size else 0.05
    if not np.isfinite(vmax) or vmax == 0.0:
        vmax = 0.05
    vmin = -vmax
    im = ax.imshow(pivot_arr, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(12), labels=["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax.set_yticks(range(len(pivot.index)), labels=[str(y) for y in pivot.index])
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot_arr[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.1%}", ha="center", va="center", fontsize=7, color="#222")
    fig.colorbar(im, ax=ax, fraction=0.03, format=FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_title("Monthly Returns")
    return _fig_to_base64(fig)


def _distribution_chart(returns: pd.Series) -> str:
    fig, ax = plt.subplots(figsize=(9, 2.5))
    if len(returns) > 0:
        ax.hist(np.asarray(returns.values), bins=60, color="#1a3a8f", alpha=0.7)
    ax.set_ylabel("Frequency")
    ax.set_xlabel("Daily return")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:+.1%}"))
    ax.set_title("Daily-returns Distribution")
    ax.grid(True, alpha=0.3)
    return _fig_to_base64(fig)


def _cpcv_distribution_chart(path_sharpes: np.ndarray) -> str:
    fig, ax = plt.subplots(figsize=(9, 2.5))
    if len(path_sharpes) > 0:
        ax.hist(
            path_sharpes, bins=min(30, max(5, len(path_sharpes) // 2)), color="#2c7fb8", alpha=0.75
        )
    ax.set_xlabel("CPCV path Sharpe (annualized)")
    ax.set_ylabel("Frequency")
    ax.set_title("CPCV Path Sharpe Distribution")
    ax.grid(True, alpha=0.3)
    return _fig_to_base64(fig)


def write_tearsheet(
    result: WalkforwardResult,
    slug: str,
    strategy_name: str,
    out_dir: Path,
    validation: ValidationReport | None = None,
) -> Path:
    """Render the HTML tear-sheet + sidecar parquet + JSON. Return the HTML path."""
    out_dir.mkdir(parents=True, exist_ok=True)

    n_windows = len(result.per_window_params)
    oos_start = str(result.oos_equity_curve.index.min().date()) if n_windows > 0 else "—"
    oos_end = str(result.oos_equity_curve.index.max().date()) if n_windows > 0 else "—"

    metrics = _MetricsBundle(
        total_return=total_return(result.oos_returns),
        cagr=cagr(result.oos_returns),
        sharpe=sharpe(result.oos_returns),
        sortino=sortino(result.oos_returns),
        max_drawdown=max_drawdown(result.oos_returns),
        win_rate=win_rate(result.oos_returns),
        n_trades=len(result.oos_trades),
        starting_equity=float(result.combined_result.starting_equity),
        ending_equity=float(result.combined_result.ending_equity),
    )

    charts: dict[str, str] = {}
    if n_windows > 0:
        charts = {
            "equity": _equity_chart(result.oos_equity_curve),
            "drawdown": _drawdown_chart(result.oos_equity_curve),
            "monthly": _monthly_chart(result.oos_returns),
            "distribution": _distribution_chart(result.oos_returns),
        }
    if validation is not None and len(validation.cpcv_path_sharpes) > 0:
        charts["cpcv"] = _cpcv_distribution_chart(validation.cpcv_path_sharpes)

    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("tearsheet.html.j2")

    windows_payload: list[dict[str, Any]] = [
        {
            "train_start": str(w.train_start),
            "train_end": str(w.train_end),
            "test_start": str(w.test_start),
            "test_end": str(w.test_end),
            "params": params,
        }
        for w, params in result.per_window_params
    ]

    html = template.render(
        strategy_name=strategy_name,
        slug=slug,
        n_windows=n_windows,
        oos_start=oos_start,
        oos_end=oos_end,
        metrics=metrics,
        charts=charts,
        windows=windows_payload,
        validation=validation,
    )

    html_path = out_dir / "tearsheet.html"
    html_path.write_text(html, encoding="utf-8")

    # Sidecar parquet
    equity_df = result.oos_equity_curve.to_frame(name="equity")
    equity_df.to_parquet(out_dir / "walkforward.parquet")

    # Sidecar JSON. ``latest`` is the most-recent window's chosen params — i.e.
    # what live trading should use. Live rebalance can read this directly
    # instead of re-running a full walk-forward each day.
    latest_params = (
        {k: _json_safe(v) for k, v in dict(result.per_window_params[-1][1]).items()}
        if result.per_window_params
        else {}
    )
    payload: dict[str, Any] = {
        "slug": slug,
        "strategy_name": strategy_name,
        "n_windows": n_windows,
        "latest": latest_params,
        "windows": windows_payload,
    }
    (out_dir / "chosen_params.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return html_path


def write_combined_tearsheet(
    result: CombinedResult,
    out_dir: Path,
) -> Path:
    """Render the combined-book HTML tear-sheet + sidecar parquet.

    Shows the joint equity / drawdown / monthly heatmap, plus a per-strategy
    breakdown table (allocation, end equity, Sharpe, CAGR, MaxDD) and a
    stacked equity-curve chart so the reader can see each strategy's
    contribution to the combined curve.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = _MetricsBundle(
        total_return=total_return(result.returns),
        cagr=cagr(result.returns),
        sharpe=sharpe(result.returns),
        sortino=sortino(result.returns),
        max_drawdown=max_drawdown(result.returns),
        win_rate=win_rate(result.returns),
        n_trades=len(result.trades),
        starting_equity=float(result.starting_equity),
        ending_equity=float(result.ending_equity),
    )

    charts: dict[str, str] = {}
    if not result.equity_curve.empty:
        charts["equity"] = _equity_chart(result.equity_curve)
        charts["drawdown"] = _drawdown_chart(result.equity_curve)
        charts["monthly"] = _monthly_chart(result.returns)
        charts["distribution"] = _distribution_chart(result.returns)
        charts["stacked"] = _stacked_equity_chart(result)

    per_strategy_rows: list[dict[str, Any]] = []
    for slug in sorted(result.per_strategy):
        sub = result.per_strategy[slug]
        per_strategy_rows.append(
            {
                "slug": slug,
                "allocation": result.allocation.get(slug, 0.0),
                "starting_equity": float(sub.starting_equity),
                "ending_equity": float(sub.ending_equity),
                "total_return": total_return(sub.returns),
                "sharpe": sharpe(sub.returns),
                "cagr": cagr(sub.returns),
                "max_drawdown": max_drawdown(sub.returns),
                "n_trades": len(sub.trades),
            }
        )

    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("combined_tearsheet.html.j2")
    html = template.render(
        metrics=metrics,
        charts=charts,
        per_strategy=per_strategy_rows,
        n_strategies=len(per_strategy_rows),
        oos_start=str(result.equity_curve.index.min().date())
        if not result.equity_curve.empty
        else "—",
        oos_end=str(result.equity_curve.index.max().date())
        if not result.equity_curve.empty
        else "—",
    )
    html_path = out_dir / "tearsheet.html"
    html_path.write_text(html, encoding="utf-8")

    # Sidecar parquets
    if not result.equity_curve.empty:
        result.equity_curve.to_frame(name="equity").to_parquet(out_dir / "equity.parquet")
    if not result.trades.empty:
        result.trades.to_parquet(out_dir / "trades.parquet")

    return html_path


def _stacked_equity_chart(result: CombinedResult) -> str:
    """Per-strategy equity curves stacked into one figure."""
    fig, ax = plt.subplots(figsize=(9, 4))
    for slug in sorted(result.per_strategy):
        sub = result.per_strategy[slug]
        if sub.equity_curve.empty:
            continue
        ax.plot(sub.equity_curve.index, np.asarray(sub.equity_curve.values), label=slug, alpha=0.8)
    if not result.equity_curve.empty:
        ax.plot(
            result.equity_curve.index,
            np.asarray(result.equity_curve.values),
            label="COMBINED",
            color="black",
            linewidth=2.0,
        )
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity ($)")
    ax.set_title("Per-strategy + combined equity")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    return _fig_to_base64(fig)
