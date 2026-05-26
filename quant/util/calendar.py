"""Calendar convenience shim.

Re-exports the project's canonical NYSE calendar helpers under the names used
by ``quant.live.recon`` and any other callers that prefer the shorter alias.
All real logic lives in ``quant.util.trading_calendar``; nothing new here.
"""

from __future__ import annotations

from datetime import date

from quant.util.trading_calendar import previous_trading_day as _previous_trading_day


def prior_trading_day(asof: date) -> date:
    """Return the most recent NYSE trading day strictly before ``asof``."""
    return _previous_trading_day(asof)
