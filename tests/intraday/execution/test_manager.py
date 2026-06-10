from quant.intraday.execution.config import ExecConfig
from quant.intraday.execution.manager import ExecutionManager
from quant.intraday.strategy import Order, OrderType, Side


def test_start_entry_builds_program_and_blocks_restart():
    mgr = ExecutionManager(ExecConfig(horizon_ticks=3))
    parent = Order("QQQ", Side.BUY, 90, OrderType.MARKET)
    mgr.start_entry(parent, tick_index=0, sigma=0.02, eta=1e-4, gamma=1e-5)
    assert mgr.has_active("QQQ")
    assert mgr.start_entry(parent, tick_index=0, sigma=0.02, eta=1e-4, gamma=1e-5) is False


def test_due_slices_emit_orders_summing_to_parent_over_horizon():
    mgr = ExecutionManager(ExecConfig(horizon_ticks=3))
    mgr.start_entry(
        Order("QQQ", Side.BUY, 90, OrderType.MARKET), tick_index=0, sigma=0.02, eta=1e-4, gamma=1e-5
    )
    total = 0
    for t in range(3):
        for o in mgr.due_slices(t):
            assert o.symbol == "QQQ" and o.side is Side.BUY
            total += o.qty
            mgr.record_fill("QQQ", o.qty)
    assert total == 90
    assert not mgr.has_active("QQQ")


def test_cancel_removes_program():
    mgr = ExecutionManager(ExecConfig(horizon_ticks=3))
    mgr.start_entry(
        Order("IWM", Side.SELL, 30, OrderType.MARKET),
        tick_index=0,
        sigma=0.02,
        eta=1e-4,
        gamma=1e-5,
    )
    mgr.cancel("IWM")
    assert not mgr.has_active("IWM")
    assert mgr.due_slices(0) == []
