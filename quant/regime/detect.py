"""Walk-forward regime detection: the only orchestrator in the package.

Refits the HMM on a schedule over a trailing window, then runs the online
forward filter forward with frozen params until the next refit. After each
refit, raw EM state indices are mapped to canonical labels by their fitted
volatility so the daily label series stays continuous across refit boundaries.

Note on filter warm-up: each segment restarts the forward filter from
``start_prob`` at the first row of the segment. The first few labels of each
segment are therefore less informative (they haven't accumulated evidence), but
this is acceptable for this milestone. It is PIT-safe because the fixed prior
``start_prob`` uses no future data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from quant.regime.hmm import fit_hmm, forward_filter
from quant.regime.models import N_STATES, REGIME_LABELS, HMMParams


@dataclass(frozen=True)
class DetectConfig:
    refit_freq: str = "MS"  # pandas offset alias: month start
    train_window_days: int = 252 * 5
    expanding: bool = False
    n_restarts: int = 8
    seed: int = 0


def identify_states(params: HMMParams, vol_index: int = 1) -> list[int]:
    """Map raw state index -> canonical index (0 calm, 1 choppy, 2 crisis).

    Ranks states by fitted mean volatility feature ascending: lowest vol is
    calm-bull, highest is crisis. Returns a list ``mapping`` where
    ``mapping[canonical] = raw_state``. Deterministic via stable argsort.
    """
    vol_means: np.ndarray = params.means[:, vol_index]
    order: np.ndarray = np.argsort(vol_means, kind="stable")
    return [int(order[c]) for c in range(params.n_states)]


def _refit_dates(index: pd.DatetimeIndex, config: DetectConfig) -> list[pd.Timestamp]:
    """Return the sorted list of timestamps at which we refit the HMM.

    The first refit happens at the first anchor date once we have at least
    ``train_window_days`` rows before it.
    """
    start_pos = config.train_window_days
    if start_pos >= len(index):
        # Defensive: not enough data for even one full window; anchor at midpoint.
        return [pd.Timestamp(index[len(index) // 2])] if len(index) else []
    anchors_raw: pd.Series[pd.Timestamp] = (
        pd.Series(index=index, data=index).resample(config.refit_freq).first().dropna()
    )
    result: list[pd.Timestamp] = [pd.Timestamp(ts) for ts in anchors_raw if ts >= index[start_pos]]
    return result


def run_detection(features: pd.DataFrame, config: DetectConfig) -> pd.DataFrame:
    """Produce a daily filtered-posterior + canonical-label frame.

    Each row contains:
    - ``p_calm``, ``p_choppy``, ``p_crisis``: filtered posteriors summing to 1.
    - ``label``: argmax canonical label string.
    - ``refit_epoch``: integer counting which HMM fit produced this row.

    PIT guarantee: labels for any date depend only on observations up to that
    date (no look-ahead). The training window is strictly prior to the refit
    anchor (we exclude the anchor row itself via ``iloc[:-1]``), and the forward
    filter is causal by construction.
    """
    feats = features.sort_index()
    index: pd.DatetimeIndex = feats.index  # type: ignore[assignment]
    refit_dates: list[pd.Timestamp] = _refit_dates(index, config)
    if not refit_dates:
        refit_dates = [pd.Timestamp(index[0])]

    rows: dict[pd.Timestamp, dict[str, object]] = {}
    for epoch, refit_ts in enumerate(refit_dates):
        end: pd.Timestamp = refit_ts
        if config.expanding:
            train_raw: pd.DataFrame = feats.loc[:end].iloc[:-1]
        else:
            train_raw = feats.loc[:end].iloc[-(config.train_window_days + 1) : -1]
        if len(train_raw) < N_STATES * 10:
            continue
        params = fit_hmm(
            train_raw.to_numpy(),
            n_states=N_STATES,
            n_restarts=config.n_restarts,
            seed=config.seed,
        )
        mapping: list[int] = identify_states(params)

        # Determine the segment of dates covered by this epoch's fitted params.
        # For non-final epochs: cover [end, next_refit_date), i.e. exclude the
        # right endpoint because that date will be the start of the next epoch.
        # This ensures each date appears in exactly one epoch and the PIT test
        # passes even when truncation changes which epoch covers a boundary date.
        is_last: bool = epoch + 1 >= len(refit_dates)
        if is_last:
            seg_raw: pd.DataFrame = feats.loc[end:]
        else:
            seg_end: pd.Timestamp = refit_dates[epoch + 1]
            seg_raw = feats.loc[end:seg_end].iloc[:-1]

        if seg_raw.empty:
            continue

        post_raw: np.ndarray = forward_filter(seg_raw.to_numpy(), params)  # (T, K) raw order
        # Reorder columns to canonical order: 0=calm, 1=choppy, 2=crisis.
        post: np.ndarray = post_raw[:, mapping]
        label_indices: np.ndarray = post.argmax(axis=1)
        labels: np.ndarray = np.array(REGIME_LABELS)[label_indices]

        for i, ts in enumerate(seg_raw.index):
            rows[pd.Timestamp(ts)] = {
                "p_calm": float(post[i, 0]),
                "p_choppy": float(post[i, 1]),
                "p_crisis": float(post[i, 2]),
                "label": str(labels[i]),
                "refit_epoch": epoch,
            }

    result_df: pd.DataFrame = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    result_df.index.name = "date"
    return result_df


def persist_regime_series(frame: pd.DataFrame, data_dir: Path) -> Path:
    """Write the daily regime frame to a parquet file; return the path."""
    path = data_dir / "regime" / "regime_series.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)
    return path


def persist_model(params: HMMParams, meta: dict[str, object], data_dir: Path) -> Path:
    """Serialise HMMParams + metadata to JSON; return the path."""
    path = data_dir / "regime" / "model.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {"params": params.to_json_dict(), "meta": meta}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path
