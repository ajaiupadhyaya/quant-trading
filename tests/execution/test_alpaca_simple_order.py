from unittest.mock import MagicMock

from quant.execution.alpaca import AlpacaClient


def _client_with_fake_trading():
    c = AlpacaClient.__new__(AlpacaClient)  # bypass __init__ (no network)
    c._trading = MagicMock()
    return c


def test_submit_simple_order_market_uses_supplied_coid():
    c = _client_with_fake_trading()
    coid = c.submit_simple_order(symbol="QQQ", side="buy", qty=5,
                                 client_order_id="sleeve:QQQ:123:0")
    assert coid == "sleeve:QQQ:123:0"
    call = c._trading.submit_order.call_args
    req = call.kwargs["order_data"] if "order_data" in call.kwargs else call.args[0]
    assert req.client_order_id == "sleeve:QQQ:123:0"
    assert req.qty == 5


def test_submit_simple_order_dry_run_skips_broker():
    c = _client_with_fake_trading()
    coid = c.submit_simple_order(symbol="IWM", side="sell", qty=3,
                                 client_order_id="sleeve:IWM:9:1", dry_run=True)
    assert coid == "sleeve:IWM:9:1"
    c._trading.submit_order.assert_not_called()
