from datetime import UTC, datetime

from quant.intraday.live.config import SleeveConfig
from quant.intraday.live.journal import TickRecord, append_tick
from quant.intraday.live.recon import position_mismatches, summarize_day


class _Pos:
    def __init__(self, symbol, qty):
        self.symbol, self.qty = symbol, qty


class _Broker:
    def __init__(self, positions):
        self._p = positions

    def positions(self):
        return self._p


def test_summarize_day_aggregates_journal(tmp_path):
    for i in range(3):
        append_tick(
            tmp_path,
            TickRecord(
                ts=datetime(2026, 6, 8, 15, i, tzinfo=UTC),
                sleeve_value=0.0,
                day_pnl=float(i * 10),
                round_trips=i,
                n_orders=1,
                halted=(i == 2),
                note="x",
            ),
        )
    s = summarize_day(tmp_path)
    assert s["n_ticks"] == 3
    assert s["last_day_pnl"] == 20.0
    assert s["max_round_trips"] == 2
    assert s["halted_any"] is True


def test_position_mismatch_detects_drift():
    cfg = SleeveConfig(universe=("QQQ", "IWM", "DIA"))
    broker = _Broker([_Pos("QQQ", 7), _Pos("SPY", 50)])  # SPY = daily system, ignored
    ledger_positions = {"QQQ": 5}  # ledger thinks 5, broker says 7 -> mismatch
    bad = position_mismatches(ledger_positions, broker, cfg)
    assert bad == {"QQQ": (5, 7)}
