# tests/intraday/data/test_adjustments.py
from datetime import date

import pandas as pd

from quant.intraday.data.adjustments import Adjustment, adjust_prices


def _prices():
    idx = pd.to_datetime(["2023-05-30T13:30:00Z", "2023-06-02T13:30:00Z"])
    return pd.DataFrame({"open": [400.0, 100.0], "close": [400.0, 100.0]}, index=idx)


def test_split_applied_when_known_as_of():
    # 4:1 split ex-date 2023-06-01; reading as_of 2023-06-05 -> pre-split prices divided by 4
    factors = [Adjustment(ex_date=date(2023, 6, 1), split_ratio=4.0, cash_dividend=0.0)]
    out = adjust_prices(_prices(), factors, as_of=date(2023, 6, 5))
    assert out.iloc[0]["open"] == 100.0  # 400 / 4 (pre-split bar back-adjusted)
    assert out.iloc[1]["open"] == 100.0  # post-split bar unchanged


def test_split_NOT_applied_when_ex_date_after_as_of():  # noqa: N802
    # Same split, but reading as_of 2023-05-31 (before ex-date) -> must NOT adjust (no lookahead)
    factors = [Adjustment(ex_date=date(2023, 6, 1), split_ratio=4.0, cash_dividend=0.0)]
    out = adjust_prices(_prices(), factors, as_of=date(2023, 5, 31))
    assert out.iloc[0]["open"] == 400.0  # untouched — the split wasn't known yet


def test_no_factors_is_identity():
    out = adjust_prices(_prices(), [], as_of=date(2023, 6, 5))
    pd.testing.assert_frame_equal(out, _prices())


def test_split_applied_on_tz_naive_index():
    # A tz-NAIVE index must not crash the ex_date comparison (regression).
    idx = pd.to_datetime(["2023-05-30", "2023-06-02"])  # tz-naive
    df = pd.DataFrame({"open": [400.0, 100.0], "close": [400.0, 100.0]}, index=idx)
    factors = [Adjustment(ex_date=date(2023, 6, 1), split_ratio=4.0, cash_dividend=0.0)]
    out = adjust_prices(df, factors, as_of=date(2023, 6, 5))
    assert out.iloc[0]["open"] == 100.0  # pre-split bar back-adjusted, no crash
