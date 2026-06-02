"""Analyst layer (E2): a daily, read-only digest of the trading day.

Gathers the day's facts from the shared ``data_dir`` (trades, equity, positions,
guardrail status, scheduler run-ledger, governance), narrates them with the
Claude API, and delivers the result to Slack. It NEVER submits orders and never
resumes a halt — it only reads.
"""

from quant.analyst.digest import (
    DigestData,
    DigestResult,
    gather_digest_data,
    narrate,
    render_facts,
    run_digest,
)

__all__ = [
    "DigestData",
    "DigestResult",
    "gather_digest_data",
    "narrate",
    "render_facts",
    "run_digest",
]
