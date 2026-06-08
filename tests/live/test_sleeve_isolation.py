"""Isolation invariant between the daily system and the intraday sleeve (issue I2):
the sleeve trades a disjoint universe and its positions must NOT enter the daily
portfolio-risk gate's view, even though both share one Alpaca account."""

from quant.data.universe import ETF_UNIVERSE, MEGACAP_UNIVERSE, SLEEVE_UNIVERSE
from quant.live.rebalance import exclude_sleeve_positions


def test_sleeve_universe_is_disjoint_from_daily_universes():
    daily = set(ETF_UNIVERSE) | set(MEGACAP_UNIVERSE)
    assert set(SLEEVE_UNIVERSE).isdisjoint(daily)


def test_sleeve_config_defaults_to_shared_universe():
    from quant.intraday.live.config import SleeveConfig

    assert SleeveConfig().universe == SLEEVE_UNIVERSE


def test_exclude_sleeve_positions_drops_only_sleeve_symbols():
    post_trade = {"SPY": 100, "TLT": -50, "QQQ": 20, "IWM": -5, "DIA": 3}
    filtered = exclude_sleeve_positions(post_trade)
    # Daily holdings untouched; sleeve symbols removed.
    assert filtered == {"SPY": 100, "TLT": -50}


def test_exclude_sleeve_positions_noop_when_no_sleeve_symbols():
    post_trade = {"SPY": 100, "GLD": 10}
    assert exclude_sleeve_positions(post_trade) == post_trade
