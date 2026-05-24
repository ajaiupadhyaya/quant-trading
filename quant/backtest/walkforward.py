"""Walk-forward harness: rolling train/test windows, grid search per window."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, timedelta

from dateutil.relativedelta import relativedelta


@dataclass(frozen=True)
class WalkforwardWindow:
    train_start: date
    train_end: date
    test_start: date
    test_end: date


def iter_windows(
    start: date,
    end: date,
    train_years: int = 5,
    test_years: int = 1,
    step_months: int = 6,
) -> Iterator[WalkforwardWindow]:
    """Yield rolling train/test windows over [start, end].

    The first window's train_start = ``start``. Each subsequent window steps the
    train_start forward by ``step_months``. A window is yielded only if its
    test_end <= ``end``.
    """
    if end <= start:
        raise ValueError(f"end ({end}) must be > start ({start})")
    if train_years <= 0 or test_years <= 0 or step_months <= 0:
        raise ValueError("train_years, test_years, step_months must all be positive")

    train_start = start
    while True:
        train_end = train_start + relativedelta(years=train_years) - timedelta(days=1)
        test_start = train_end + timedelta(days=1)
        test_end = test_start + relativedelta(years=test_years) - timedelta(days=1)
        if test_end > end:
            return
        yield WalkforwardWindow(
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        )
        train_start = train_start + relativedelta(months=step_months)
