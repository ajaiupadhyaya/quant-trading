"""Claude-spend cost metering: pricing, fail-open recording, aggregation, budget helper."""

from datetime import UTC, datetime
from types import SimpleNamespace

from quant.analyst.spend import (
    cost_usd,
    load_records,
    over_daily_budget,
    record_spend,
    summarize,
)

_NOW = datetime(2026, 6, 9, 14, 30, tzinfo=UTC)


def _usage(it: int, ot: int, cr: int = 0, cw: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        input_tokens=it,
        output_tokens=ot,
        cache_read_input_tokens=cr,
        cache_creation_input_tokens=cw,
    )


def test_cost_usd_per_model_rates() -> None:
    assert cost_usd("claude-opus-4-8", input_tokens=1_000_000, output_tokens=0) == 5.0
    assert cost_usd("claude-opus-4-8", input_tokens=0, output_tokens=1_000_000) == 25.0
    assert cost_usd("claude-sonnet-4-6", input_tokens=1_000_000, output_tokens=0) == 3.0
    assert cost_usd("claude-sonnet-4-6", input_tokens=0, output_tokens=1_000_000) == 15.0
    assert cost_usd("claude-haiku-4-5", input_tokens=1_000_000, output_tokens=0) == 1.0
    assert cost_usd("claude-haiku-4-5", input_tokens=0, output_tokens=1_000_000) == 5.0


def test_cost_usd_cache_multipliers_and_unknown_model() -> None:
    # cache reads at 0.1x input, writes at 1.25x input (Opus input $5/1M)
    assert (
        abs(
            cost_usd(
                "claude-opus-4-8", input_tokens=0, output_tokens=0, cache_read_tokens=1_000_000
            )
            - 0.5
        )
        < 1e-9
    )
    assert (
        abs(
            cost_usd(
                "claude-opus-4-8", input_tokens=0, output_tokens=0, cache_write_tokens=1_000_000
            )
            - 6.25
        )
        < 1e-9
    )
    # unknown model prices as Opus (highest tier); a suffixed id resolves to its base rate
    assert cost_usd("some-future-model", input_tokens=1_000_000, output_tokens=0) == 5.0
    assert cost_usd("claude-opus-4-8[1m]", input_tokens=1_000_000, output_tokens=0) == 5.0


def test_record_spend_writes_and_computes_cost(tmp_path) -> None:
    rec = record_spend(
        call_site="digest",
        model="claude-haiku-4-5",
        usage=_usage(2000, 500),
        data_dir=tmp_path,
        now=_NOW,
    )
    assert rec is not None
    assert rec.date == "2026-06-09" and rec.call_site == "digest"
    # 2000*$1/1M + 500*$5/1M = 0.002 + 0.0025 = 0.0045
    assert rec.cost_usd == 0.0045
    rows = load_records(tmp_path)
    assert len(rows) == 1 and rows[0]["cost_usd"] == 0.0045


def test_record_spend_failopen_on_missing_usage(tmp_path) -> None:
    assert (
        record_spend(call_site="watch", model="claude-opus-4-8", usage=None, data_dir=tmp_path)
        is None
    )
    assert load_records(tmp_path) == []  # nothing written when there is no usage


def test_load_records_skips_corrupt_lines(tmp_path) -> None:
    record_spend(
        call_site="watch",
        model="claude-opus-4-8",
        usage=_usage(10, 10),
        data_dir=tmp_path,
        now=_NOW,
    )
    with (tmp_path / "research" / "claude_spend.jsonl").open("a") as f:
        f.write("{ this is not json\n")
    assert len(load_records(tmp_path)) == 1  # the good record survives a corrupt line


def test_summarize_aggregates(tmp_path) -> None:
    record_spend(
        call_site="digest",
        model="claude-haiku-4-5",
        usage=_usage(1000, 0),
        data_dir=tmp_path,
        now=datetime(2026, 6, 8, 12, tzinfo=UTC),
    )
    record_spend(
        call_site="watch",
        model="claude-opus-4-8",
        usage=_usage(1000, 0),
        data_dir=tmp_path,
        now=_NOW,
    )
    s = summarize(load_records(tmp_path), asof_date="2026-06-09")
    assert s["calls"] == 2
    assert set(s["by_call_site"]) == {"digest", "watch"}
    assert set(s["by_model"]) == {"claude-haiku-4-5", "claude-opus-4-8"}
    assert s["today_usd"] == 0.005  # only the opus row falls on 2026-06-09 (1000*$5/1M)
    assert s["total_usd"] == round(0.001 + 0.005, 6)


def test_over_daily_budget(tmp_path) -> None:
    record_spend(
        call_site="watch",
        model="claude-opus-4-8",
        usage=_usage(1_000_000, 0),
        data_dir=tmp_path,
        now=_NOW,
    )  # $5.00 on 2026-06-09
    rows = load_records(tmp_path)
    assert over_daily_budget(rows, daily_budget_usd=4.0, date="2026-06-09") is True
    assert over_daily_budget(rows, daily_budget_usd=10.0, date="2026-06-09") is False
    assert over_daily_budget(rows, daily_budget_usd=0.0, date="2026-06-09") is False  # disabled
    assert over_daily_budget(rows, daily_budget_usd=1.0, date="2026-06-08") is False  # other day
