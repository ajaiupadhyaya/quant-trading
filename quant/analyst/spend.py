"""Claude-spend cost metering for the analyst's Anthropic calls.

A small, fail-open ledger: each metered call appends one record (tokens + estimated USD)
to ``data/research/claude_spend.jsonl`` so spend is observable per day / model / call-site.
This is PURE OBSERVABILITY — recording never blocks, delays, or skips a Claude call, and a
metering failure can never affect the analyst output. A soft *budget gate* that actually
skips calls is a separate, deliberate opt-in (``over_daily_budget`` is provided for it but
is not consulted by the call sites yet)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from quant.util.config import Settings
from quant.util.logging import logger

# USD per 1M tokens (input, output). Source: Anthropic pricing via the claude-api skill
# reference (cached 2026-05). Opus 4.x $5/$25, Sonnet 4.x $3/$15, Haiku 4.5 $1/$5.
_PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-opus-4-5": (5.0, 25.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}
# Unknown model → price as Opus (the most expensive tier) so estimates never understate.
_DEFAULT_PRICING: tuple[float, float] = (5.0, 25.0)
# Cache reads bill ~0.1x input; 5-minute cache writes ~1.25x input.
_CACHE_READ_MULT = 0.1
_CACHE_WRITE_MULT = 1.25


def _rate_for(model: str) -> tuple[float, float]:
    """(input, output) USD/1M for a model id; tolerant of id suffixes; Opus-priced if unknown."""
    if model in _PRICING:
        return _PRICING[model]
    for key, rate in _PRICING.items():
        if model.startswith(key):  # e.g. "claude-opus-4-8[1m]" or a dated/fast variant
            return rate
    return _DEFAULT_PRICING


def cost_usd(
    model: str,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    """Estimated USD for one call given its token usage (cache reads/writes priced off input)."""
    in_rate, out_rate = _rate_for(model)
    return (
        input_tokens * in_rate
        + cache_read_tokens * in_rate * _CACHE_READ_MULT
        + cache_write_tokens * in_rate * _CACHE_WRITE_MULT
        + output_tokens * out_rate
    ) / 1_000_000.0


@dataclass(frozen=True)
class SpendRecord:
    ts: str  # ISO-8601 UTC timestamp
    date: str  # YYYY-MM-DD (UTC) — the daily aggregation key
    call_site: str  # "digest" | "advisor.brief" | "advisor.propose" | "watch" | ...
    model: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    cost_usd: float


def ledger_path(data_dir: Path | None = None) -> Path:
    base = data_dir if data_dir is not None else Settings().data_dir  # type: ignore[call-arg]
    return Path(base) / "research" / "claude_spend.jsonl"


def record_spend(
    *,
    call_site: str,
    model: str | None,
    usage: Any,
    data_dir: Path | None = None,
    now: datetime | None = None,
) -> SpendRecord | None:
    """Append one usage record to the spend ledger. FAIL-OPEN: never raises and returns
    None on any problem (incl. ``usage is None``, e.g. a template fallback or test fake) —
    metering must not affect the Claude call."""
    if usage is None:
        return None
    try:
        mdl = model or "unknown"
        it = int(getattr(usage, "input_tokens", 0) or 0)
        ot = int(getattr(usage, "output_tokens", 0) or 0)
        cr = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
        cw = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
        stamp = now if now is not None else datetime.now(UTC)
        rec = SpendRecord(
            ts=stamp.isoformat(),
            date=stamp.date().isoformat(),
            call_site=call_site,
            model=mdl,
            input_tokens=it,
            output_tokens=ot,
            cache_read_tokens=cr,
            cache_write_tokens=cw,
            cost_usd=round(
                cost_usd(
                    mdl,
                    input_tokens=it,
                    output_tokens=ot,
                    cache_read_tokens=cr,
                    cache_write_tokens=cw,
                ),
                6,
            ),
        )
        path = ledger_path(data_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps(asdict(rec)) + "\n")
        return rec
    except Exception as exc:  # fail-open: a metering failure must not break the analyst call
        logger.info("analyst.spend: record skipped ({!r})", exc)
        return None


def load_records(data_dir: Path | None = None) -> list[dict[str, Any]]:
    """Read the ledger as a list of dicts; missing file → []; corrupt lines are skipped."""
    path = ledger_path(data_dir)
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue  # tolerate a single corrupt line without losing the rest
    return out


def summarize(records: list[dict[str, Any]], *, asof_date: str | None = None) -> dict[str, Any]:
    """Aggregate spend across the records: total, count, per-day/model/call-site, and the
    ``asof_date`` day's spend when given."""
    by_day: dict[str, float] = {}
    by_model: dict[str, float] = {}
    by_site: dict[str, float] = {}
    total = 0.0
    for r in records:
        c = float(r.get("cost_usd", 0.0) or 0.0)
        total += c
        by_day[r.get("date", "")] = by_day.get(r.get("date", ""), 0.0) + c
        by_model[r.get("model", "")] = by_model.get(r.get("model", ""), 0.0) + c
        by_site[r.get("call_site", "")] = by_site.get(r.get("call_site", ""), 0.0) + c
    return {
        "total_usd": round(total, 6),
        "calls": len(records),
        "today_usd": (round(by_day.get(asof_date, 0.0), 6) if asof_date is not None else None),
        "by_day": {k: round(v, 6) for k, v in sorted(by_day.items())},
        "by_model": {k: round(v, 6) for k, v in sorted(by_model.items())},
        "by_call_site": {k: round(v, 6) for k, v in sorted(by_site.items())},
    }


def over_daily_budget(records: list[dict[str, Any]], *, daily_budget_usd: float, date: str) -> bool:
    """True if ``date``'s spend already meets/exceeds the soft daily budget (<=0 disables).

    Provided for a future opt-in budget gate; the live analyst call sites do NOT consult
    this yet — recording is decoupled from any skip decision by design."""
    if daily_budget_usd <= 0:
        return False
    spent = sum(float(r.get("cost_usd", 0.0) or 0.0) for r in records if r.get("date") == date)
    return spent >= daily_budget_usd
