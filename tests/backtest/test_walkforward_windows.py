"""Tests for quant.backtest.walkforward.iter_windows."""

from __future__ import annotations

from datetime import date

import pytest

from quant.backtest.walkforward import WalkforwardWindow, iter_windows


def test_first_window_starts_at_beginning() -> None:
    windows = list(iter_windows(date(2010, 1, 1), date(2020, 12, 31)))
    assert windows[0].train_start == date(2010, 1, 1)


def test_default_5y_train_1y_test() -> None:
    windows = list(iter_windows(date(2010, 1, 1), date(2020, 12, 31)))
    first = windows[0]
    assert first.train_end == date(2014, 12, 31)
    assert first.test_start == date(2015, 1, 1)
    assert first.test_end == date(2015, 12, 31)


def test_default_6m_step() -> None:
    windows = list(iter_windows(date(2010, 1, 1), date(2020, 12, 31)))
    assert windows[1].train_start == date(2010, 7, 1)
    assert windows[1].test_start == date(2015, 7, 1)


def test_windows_stop_when_test_exceeds_end() -> None:
    windows = list(iter_windows(date(2010, 1, 1), date(2017, 12, 31)))
    # 5y train requires train_start <= 2012-12-31 → only a few step positions valid.
    for w in windows:
        assert w.test_end <= date(2017, 12, 31)


def test_custom_train_test_step() -> None:
    windows = list(
        iter_windows(
            start=date(2020, 1, 1),
            end=date(2024, 12, 31),
            train_years=2,
            test_years=1,
            step_months=12,
        )
    )
    assert windows[0].train_start == date(2020, 1, 1)
    assert windows[0].train_end == date(2021, 12, 31)
    assert windows[0].test_start == date(2022, 1, 1)
    assert windows[0].test_end == date(2022, 12, 31)
    assert windows[1].train_start == date(2021, 1, 1)


def test_empty_when_window_doesnt_fit() -> None:
    # 5y train + 1y test = 6y total; only 2y of data.
    windows = list(iter_windows(date(2020, 1, 1), date(2022, 1, 1)))
    assert windows == []


def test_window_is_dataclass() -> None:
    w = WalkforwardWindow(
        train_start=date(2010, 1, 1),
        train_end=date(2014, 12, 31),
        test_start=date(2015, 1, 1),
        test_end=date(2015, 12, 31),
    )
    assert w.train_start == date(2010, 1, 1)


def test_invalid_arguments_raise() -> None:
    with pytest.raises(ValueError):
        list(iter_windows(date(2020, 1, 1), date(2010, 1, 1)))  # end before start
    with pytest.raises(ValueError):
        list(iter_windows(date(2010, 1, 1), date(2020, 1, 1), train_years=0))
