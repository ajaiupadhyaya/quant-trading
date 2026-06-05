"""Research experiment registry, comparison helpers, and the quant signals engine."""

from quant.research.registry import (
    ExperimentRecord,
    append_experiment,
    compare_experiments,
    leaderboard,
    list_experiments,
)
from quant.research.signals import (
    AssetSignal,
    MarketSignals,
    SignalConfig,
    append_signals,
    build_market_signals,
    load_market_signals,
    read_latest_signals,
    render_signals,
    signals_path,
)

__all__ = [
    "AssetSignal",
    "ExperimentRecord",
    "MarketSignals",
    "SignalConfig",
    "append_experiment",
    "append_signals",
    "build_market_signals",
    "compare_experiments",
    "leaderboard",
    "list_experiments",
    "load_market_signals",
    "read_latest_signals",
    "render_signals",
    "signals_path",
]
