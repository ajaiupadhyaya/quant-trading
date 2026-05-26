"""Markdown renderer for ReconciliationReport. Pure formatting, no I/O."""

from __future__ import annotations

from io import StringIO

from quant.live.recon import ReconciliationReport


def render_markdown(report: ReconciliationReport) -> str:
    buf = StringIO()
    _write_header(buf, report)

    if not report.rows:
        buf.write("\n_no trades to reconcile in this window._\n")
        return buf.getvalue()

    _write_summary(buf, report)
    _write_slippage_section(buf, report)
    _write_timing_section(buf, report)
    _write_fidelity_section(buf, report)
    _write_per_symbol_section(buf, report)
    return buf.getvalue()


def _write_header(buf: StringIO, report: ReconciliationReport) -> None:
    title_date = report.until.isoformat()
    buf.write(f"# Live Reconciliation {title_date}\n\n")
    buf.write(f"**Window:** {report.since.isoformat()} → {report.until.isoformat()}  \n")
    modeled = report.modeled_slippage_bps
    modeled_str = f"{modeled:.1f}" if modeled == int(modeled) else f"{modeled:g}"
    buf.write(f"**Modeled cost (engine, symmetric):** {modeled_str} bps  \n")
    buf.write(f"**Total orders in window:** {len(report.rows)}  \n")
    buf.write(
        "\n> _The signed mean below is `(fill − signal) / signal` (sign-adjusted "
        "for side). It captures **signal-to-fill price drift**, not pure execution "
        "cost: signal_price is the close that produced the signal, fill_price is "
        "the next-open execution. Do not use as cost-model calibration without an "
        "intraday-mid-based metric. See `docs/live-recon/cost-model-interpretation.md`._\n"
    )


def _write_summary(buf: StringIO, report: ReconciliationReport) -> None:
    n_filled = sum(1 for r in report.rows if r.status == "filled")
    n_partial = sum(1 for r in report.rows if r.status == "partial")
    n_rejected = sum(1 for r in report.rows if r.status == "rejected")
    n_missing = sum(1 for r in report.rows if r.status == "missing")
    n_no_signal = sum(1 for r in report.rows if r.status == "no_signal_price")
    n_no_fill_price = sum(1 for r in report.rows if r.status == "no_fill_price")

    buf.write("\n## Summary\n\n")
    buf.write("| status | count |\n|---|---:|\n")
    buf.write(f"| filled | {n_filled} |\n")
    buf.write(f"| partial | {n_partial} |\n")
    buf.write(f"| rejected | {n_rejected} |\n")
    buf.write(f"| missing | {n_missing} |\n")
    buf.write(f"| no_signal_price | {n_no_signal} |\n")
    buf.write(f"| no_fill_price | {n_no_fill_price} |\n")


def _write_slippage_section(buf: StringIO, report: ReconciliationReport) -> None:
    buf.write("\n## Signal-to-fill drift (filled orders)\n\n")
    by_strat = report.aggregate_by_strategy()
    buf.write(
        "| strategy | n | mean signed drift (bps) | diagnostic Δ vs modeled cost |\n"
        "|---|---:|---:|---:|\n"
    )
    for strat, stats in sorted(by_strat.items()):
        mean_slip = stats["mean_slippage_bps"]
        if mean_slip is None:
            buf.write(f"| {strat} | 0 | — | — |\n")
            continue
        delta = float(mean_slip) - report.modeled_slippage_bps
        buf.write(f"| {strat} | {stats['n_filled']} | {float(mean_slip):.2f} | {delta:+.2f} |\n")


def _write_timing_section(buf: StringIO, report: ReconciliationReport) -> None:
    buf.write("\n## Timing\n\n")
    by_strat = report.aggregate_by_strategy()
    buf.write("| strategy | median fill lag (s) |\n|---|---:|\n")
    for strat, stats in sorted(by_strat.items()):
        lag = stats["median_fill_lag_s"]
        buf.write(f"| {strat} | {'—' if lag is None else f'{float(lag):.1f}'} |\n")


def _write_fidelity_section(buf: StringIO, report: ReconciliationReport) -> None:
    buf.write("\n## Fidelity\n\n")
    flagged = [
        r for r in report.rows
        if r.status in {"partial", "rejected", "missing", "no_signal_price", "no_fill_price"}
    ]
    if not flagged:
        buf.write("_all orders filled cleanly._\n")
        return
    buf.write("| coid | symbol | side | submitted | filled | status |\n|---|---|---|---:|---:|---|\n")
    for r in flagged:
        buf.write(
            f"| `{r.client_order_id}` | {r.symbol} | {r.side} | "
            f"{r.submitted_qty} | {r.filled_qty} | {r.status} |\n"
        )


def _write_per_symbol_section(buf: StringIO, report: ReconciliationReport) -> None:
    buf.write("\n## Per-symbol breakdown\n\n")
    by_sym = report.aggregate_by_symbol()
    buf.write("| symbol | n filled | mean slippage (bps) | median lag (s) |\n|---|---:|---:|---:|\n")
    for sym, stats in sorted(by_sym.items()):
        slip = stats["mean_slippage_bps"]
        lag = stats["median_fill_lag_s"]
        slip_s = "—" if slip is None else f"{float(slip):.2f}"
        lag_s = "—" if lag is None else f"{float(lag):.1f}"
        buf.write(f"| {sym} | {stats['n_filled']} | {slip_s} | {lag_s} |\n")
