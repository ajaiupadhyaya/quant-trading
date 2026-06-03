"""AnalystContext integration: the signals block is read fail-open and rendered."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from quant.analyst.context import gather_analyst_context, render_context
from quant.research.signals import (
    append_signals,
    build_market_signals,
    signals_path,
)
from tests.research.conftest import close_panel, macro_series

ASOF = date(2024, 6, 27)


def _seed_log(data_dir: Path):
    panel = close_panel(seed=3)
    idx = panel.index
    rec = build_market_signals(
        closes=panel,
        vix=macro_series(18.0, idx),
        dgs10=macro_series(4.0, idx),
        dgs2=macro_series(4.2, idx),
        asof=idx[-1].date(),
    )
    append_signals(signals_path(data_dir), rec)
    return rec


def test_read_signals_absent_is_failopen(tmp_path: Path) -> None:
    ctx = gather_analyst_context(tmp_path, ASOF, include_macro=False)
    assert ctx.signals is None  # no log on disk


def test_gather_context_reads_latest_signals(tmp_path: Path) -> None:
    rec = _seed_log(tmp_path)
    ctx = gather_analyst_context(tmp_path, ASOF, include_macro=False)
    assert ctx.signals is not None
    assert ctx.signals.asof == rec.asof


def test_render_context_includes_signals_line(tmp_path: Path) -> None:
    _seed_log(tmp_path)
    ctx = gather_analyst_context(tmp_path, ASOF, include_macro=False)
    text = render_context(ctx)
    assert "Research signals" in text


def test_render_context_suppressed_on_bad_signals(tmp_path: Path) -> None:
    # A signals object whose render raises must not sink the whole context render.
    ctx = gather_analyst_context(tmp_path, ASOF, include_macro=False)
    object.__setattr__(ctx, "signals", object())  # render_signals(...) will raise
    text = render_context(ctx)
    assert "As-of" in text  # render still produced output
    assert "Research signals" not in text  # the bad block was suppressed


def test_corrupt_log_is_failopen(tmp_path: Path) -> None:
    p = signals_path(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{garbage not json\n", encoding="utf-8")
    ctx = gather_analyst_context(tmp_path, ASOF, include_macro=False)
    assert ctx.signals is None  # unreadable line -> None, never raises
