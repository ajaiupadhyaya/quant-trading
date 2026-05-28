from __future__ import annotations

import numpy as np
import pandas as pd

from quant.regime.detect import DetectConfig, identify_states, run_detection
from quant.regime.models import HMMParams


def test_identify_states_orders_by_volatility():
    # Feature column order is [ret, vol, ...]; vol is index 1.
    # State means: state0 high vol, state1 low vol, state2 mid vol.
    params = HMMParams(
        start_prob=np.full(3, 1 / 3),
        trans_mat=np.full((3, 3), 1 / 3),
        means=np.array([[0.0, 2.0], [0.1, -1.0], [0.0, 0.5]]),
        variances=np.ones((3, 2)),
    )
    mapping = identify_states(params, vol_index=1)
    # canonical 0=calm (lowest vol)=raw1, 1=choppy (mid)=raw2, 2=crisis (high)=raw0
    assert mapping == [1, 2, 0]


def _synthetic_features(n: int) -> pd.DataFrame:
    idx = pd.bdate_range("2010-01-01", periods=n)
    rng = np.random.default_rng(0)
    # Two clearly separated blocks so the HMM finds structure.
    ret = np.where(np.arange(n) % 500 < 400, 0.5, -0.5) + rng.normal(0, 0.2, n)
    vol = np.where(np.arange(n) % 500 < 400, -0.5, 1.5) + rng.normal(0, 0.2, n)
    return pd.DataFrame({"ret": ret, "vol": vol}, index=idx)


def test_fit_final_model_and_persist(tmp_path):
    from quant.regime.detect import fit_final_model, persist_model
    from quant.regime.models import HMMParams

    feats = _synthetic_features(400)
    cfg = DetectConfig(train_window_days=250, n_restarts=1, seed=0)
    params, meta = fit_final_model(feats, cfg)
    assert params.n_states == 3
    assert meta["n_train_obs"] == 250
    assert "loglik" in meta and "state_mapping" in meta

    path = persist_model(params, meta, tmp_path)
    assert path.exists()
    import json

    payload = json.loads(path.read_text())
    restored = HMMParams.from_json_dict(payload["params"])
    assert restored.n_states == 3
    assert payload["meta"]["seed"] == 0


def test_run_detection_outputs_daily_labels_and_is_pit():
    feats = _synthetic_features(600)
    cfg = DetectConfig(train_window_days=250, refit_freq="YS", n_restarts=1, seed=0)
    out = run_detection(feats, cfg)
    assert set(out.columns) >= {"p_calm", "p_choppy", "p_crisis", "label", "refit_epoch"}
    np.testing.assert_allclose(
        out[["p_calm", "p_choppy", "p_crisis"]].sum(axis=1).to_numpy(),
        np.ones(len(out)),
        atol=1e-9,
    )
    assert set(out["label"].unique()).issubset({"calm-bull", "choppy", "crisis"})
    # PIT: re-running on a truncated feature frame must reproduce earlier labels
    # exactly (no future data influences a past label).
    trunc = run_detection(feats.iloc[:420], cfg)
    shared = trunc.index.intersection(out.index)
    assert (out.loc[shared, "label"] == trunc.loc[shared, "label"]).all()
