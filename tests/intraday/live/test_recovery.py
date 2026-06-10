from quant.intraday.live.config import SleeveConfig
from quant.intraday.live.loop import recover_ledger


class _Pos:
    def __init__(self, symbol, qty, avg):
        self.symbol, self.qty, self.avg_entry_price = symbol, qty, avg


class _Broker:
    def __init__(self, positions):
        self._p = positions

    def positions(self):
        return self._p


def test_recovers_only_sleeve_universe_positions():
    cfg = SleeveConfig(universe=("QQQ", "IWM", "DIA"))
    broker = _Broker([_Pos("QQQ", 7, 100.0), _Pos("SPY", 50, 400.0)])  # SPY = daily system
    led = recover_ledger(broker, cfg)
    assert led.position("QQQ") == 7
    assert led.position("SPY") == 0  # ignored: not in sleeve universe


def test_recovers_empty_when_no_sleeve_positions():
    cfg = SleeveConfig()
    led = recover_ledger(_Broker([]), cfg)
    assert led.positions() == {}
