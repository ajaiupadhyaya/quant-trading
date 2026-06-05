"""Macro / policy / event-risk layer (Phase 7C) — the 'politics' discipline.

A deterministic calendar of high-impact scheduled events (FOMC / jobs / OpEx /
elections / quarter-end) combined with FRED policy-uncertainty (EPU), financial
conditions (NFCI), financial stress (STLFSI4), and the VIX term structure into a
single read of the political/macro risk environment. Read-only and advisory.
"""

from quant.macro.events import (
    EventRisk,
    ScheduledEvent,
    compute_event_risk,
    live_event_risk,
    next_high_impact_event,
    render_event_risk,
    upcoming_events,
)
from quant.macro.nowcast import (
    MacroNowcast,
    MacroNowcastConfig,
    compute_macro_nowcast,
    live_macro_nowcast,
    render_macro_nowcast,
)

__all__ = [
    "EventRisk",
    "MacroNowcast",
    "MacroNowcastConfig",
    "ScheduledEvent",
    "compute_event_risk",
    "compute_macro_nowcast",
    "live_event_risk",
    "live_macro_nowcast",
    "next_high_impact_event",
    "render_event_risk",
    "render_macro_nowcast",
    "upcoming_events",
]
