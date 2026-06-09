"""The persisted dual-track evaluation record: structure, JSON-serializability, and
reproducibility. Trains the LSTM, so it is torch-gated."""

import json

import pytest

torch = pytest.importorskip("torch")

from quant.intraday.dl.config import DLConfig  # noqa: E402
from quant.intraday.dl.evaluate import (  # noqa: E402
    DL_EVAL_SCHEMA_VERSION,
    build_evaluation,
)

# Small + fast, still enough to train.
_CFG = DLConfig(window=12, hidden_size=24, epochs=8, batch_size=64, seed=7, train_frac=0.7)


def test_record_is_structured_and_json_serializable():
    rec = build_evaluation(_CFG, n=600, seed=7, cost_per_turn=0.05)
    assert rec["schema_version"] == DL_EVAL_SCHEMA_VERSION
    assert rec["config"]["window"] == 12
    assert rec["config"]["seed"] == 7
    assert rec["n"] == 600
    assert rec["cost_per_turn"] == 0.05
    assert set(rec["tracks"]) == {"synthetic-signal", "random"}
    for track in rec["tracks"].values():
        assert set(track["models"]) == {"lstm", "linear", "naive"}
        for m in track["models"].values():
            assert {"mse", "directional_accuracy", "r2"} <= set(m)  # statistical metrics
            assert {"sharpe_gross", "sharpe_net", "hit_rate", "avg_turnover"} <= set(m)  # economics
        assert len(track["loss_curve"]) >= 1
        assert all(isinstance(x, float) for x in track["loss_curve"])
    # round-trips through JSON unchanged
    assert json.loads(json.dumps(rec)) == rec


def test_record_is_deterministic():
    a = build_evaluation(_CFG, n=600, seed=7, cost_per_turn=0.05)
    b = build_evaluation(_CFG, n=600, seed=7, cost_per_turn=0.05)
    # Deterministic training (seeded + torch.use_deterministic_algorithms) -> identical record.
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
