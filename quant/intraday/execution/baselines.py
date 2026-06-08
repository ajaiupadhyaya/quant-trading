"""Execution baselines to compare against the Almgren-Chriss schedule. Each returns
a list of integer child sizes that sum EXACTLY to total_shares."""

from __future__ import annotations


def _fix_residual(sizes: list[int], total: int) -> list[int]:
    sizes[-1] += total - sum(sizes)
    return sizes


def twap(*, total_shares: int, n_intervals: int) -> list[int]:
    """Equal slices across n_intervals; the indivisible remainder lands on the last."""
    base = total_shares // n_intervals
    sizes = [base] * n_intervals
    return _fix_residual(sizes, total_shares)


def vwap(*, total_shares: int, volume_curve: list[float]) -> list[int]:
    """Slices proportional to the expected per-interval volume curve."""
    total_vol = sum(volume_curve)
    if total_vol <= 0:
        return twap(total_shares=total_shares, n_intervals=len(volume_curve))
    sizes = [round(total_shares * v / total_vol) for v in volume_curve]
    return _fix_residual(sizes, total_shares)


def immediate(*, total_shares: int) -> list[int]:
    """One shot."""
    return [total_shares]
