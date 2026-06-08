from quant.intraday.execution.baselines import immediate, twap, vwap


def test_twap_equal_slices_sum_to_parent():
    sizes = twap(total_shares=100, n_intervals=4)
    assert sizes == [25, 25, 25, 25]
    assert sum(sizes) == 100


def test_twap_handles_indivisible_remainder_on_last():
    sizes = twap(total_shares=103, n_intervals=4)
    assert sum(sizes) == 103
    assert sizes[:3] == [25, 25, 25] and sizes[-1] == 28


def test_vwap_weights_proportional_to_volume_and_sum_to_parent():
    sizes = vwap(total_shares=100, volume_curve=[1.0, 3.0, 1.0])  # 20%/60%/20%
    assert sum(sizes) == 100
    assert sizes == [20, 60, 20]


def test_immediate_is_single_slice():
    assert immediate(total_shares=42) == [42]
