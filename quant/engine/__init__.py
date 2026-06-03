"""Continuous market-state engine (Phase 6) — the always-on, READ-ONLY quant brain.

A long-running loop maintains a live ``MarketState`` (signals + regime + risk +
posture + the live book), persists it every cycle, and a deterministic event bus
emits structured events on material changes — escalating ONLY impactful ones to
Claude (rate-limited, cost-controlled). It places NO orders, sets no halt, and
changes no governance/allocation/config. Actuation is a separate, human-gated
phase; this engine only observes, records, and notifies.
"""

from quant.engine.events import EngineEvent, EventConfig, detect_events
from quant.engine.intraday import (
    IntradaySignals,
    compute_intraday_signals,
    live_intraday_signals,
)
from quant.engine.loop import EngineConfig, run_engine
from quant.engine.state import MarketState, build_market_state, render_state

__all__ = [
    "EngineConfig",
    "EngineEvent",
    "EventConfig",
    "IntradaySignals",
    "MarketState",
    "build_market_state",
    "compute_intraday_signals",
    "detect_events",
    "live_intraday_signals",
    "render_state",
    "run_engine",
]
