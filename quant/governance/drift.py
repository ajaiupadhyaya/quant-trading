"""Paper P&L drift monitoring against validation expectations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

DriftFlag = Literal["normal", "watch", "halt_candidate"]


@dataclass(frozen=True)
class DriftConfig:
    watch_z: float = -1.0
    halt_z: float = -2.0


@dataclass(frozen=True)
class DriftRow:
    strategy: str
    window: int
    realized_return: float
    expected_return: float
    z_score: float
    flag: DriftFlag


def drift_flag(z_score: float, config: DriftConfig | None = None) -> DriftFlag:
    config = config or DriftConfig()
    if z_score <= config.halt_z:
        return "halt_candidate"
    if z_score <= config.watch_z:
        return "watch"
    return "normal"


def summarize_drift(
    realized_returns: dict[str, pd.Series],
    expected_returns: dict[str, pd.Series],
    *,
    windows: tuple[int, ...] = (5, 20, 60),
    config: DriftConfig | None = None,
) -> list[DriftRow]:
    config = config or DriftConfig()
    rows: list[DriftRow] = []
    for slug, realized in sorted(realized_returns.items()):
        expected = expected_returns.get(slug)
        if expected is None:
            continue
        aligned = pd.concat(
            [realized.rename("realized"), expected.rename("expected")],
            axis=1,
        ).dropna()
        for window in windows:
            if len(aligned) < window:
                continue
            recent = aligned.tail(window)
            diff = recent["realized"] - recent["expected"]
            vol = float(diff.std(ddof=1))
            gap = float(diff.sum())
            z = 0.0 if vol <= 0 else gap / (vol * (window ** 0.5))
            rows.append(
                DriftRow(
                    strategy=slug,
                    window=window,
                    realized_return=float(recent["realized"].sum()),
                    expected_return=float(recent["expected"].sum()),
                    z_score=z,
                    flag=drift_flag(z, config),
                )
            )
    return rows
