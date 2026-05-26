"""Snapshot tests for the Markdown renderer."""

from __future__ import annotations

from datetime import date

from quant.live.recon import ReconciliationReport, ReconRow
from quant.live.recon_render import render_markdown


def _fixture_report() -> ReconciliationReport:
    return ReconciliationReport(
        since=date(2026, 5, 26),
        until=date(2026, 5, 26),
        modeled_slippage_bps=5.0,
        rows=[
            ReconRow(
                client_order_id="trend-20260526-SPY-a",
                strategy="trend", symbol="SPY", side="buy",
                submission_date=date(2026, 5, 26),
                submitted_qty=100, filled_qty=100,
                signal_price=499.87, fill_price=500.12,
                slippage_bps=5.001, fill_lag_seconds=4.0,
                status="filled",
            ),
            ReconRow(
                client_order_id="trend-20260526-DBC-b",
                strategy="trend", symbol="DBC", side="buy",
                submission_date=date(2026, 5, 26),
                submitted_qty=50, filled_qty=0,
                signal_price=25.00, fill_price=None,
                slippage_bps=None, fill_lag_seconds=None,
                status="rejected",
            ),
        ],
    )


def test_render_markdown_contains_required_sections() -> None:
    md = render_markdown(_fixture_report())

    assert "# Live Reconciliation 2026-05-26" in md
    assert "## Summary" in md
    assert "## Signal-to-fill drift (filled orders)" in md
    assert "## Timing" in md
    assert "## Fidelity" in md
    assert "## Per-symbol breakdown" in md
    # modeled benchmark surfaces
    assert "5.0 bps" in md or "5.00 bps" in md
    # both rows present
    assert "SPY" in md and "DBC" in md
    # rejected row appears in fidelity section
    assert "rejected" in md


def test_render_markdown_empty_report_still_valid() -> None:
    empty = ReconciliationReport(
        since=date(2026, 5, 26),
        until=date(2026, 5, 26),
        modeled_slippage_bps=5.0,
        rows=[],
    )
    md = render_markdown(empty)
    assert "# Live Reconciliation 2026-05-26" in md
    assert "no trades to reconcile" in md.lower()
