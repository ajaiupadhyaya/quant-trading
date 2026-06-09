import pytest

torch = pytest.importorskip("torch")  # skip cleanly where torch is absent

from quant.intraday.dl.config import DLConfig  # noqa: E402
from quant.intraday.dl.model import build_model  # noqa: E402


def test_forward_shape():
    cfg = DLConfig(window=5, hidden_size=8)
    model = build_model(cfg)
    x = torch.zeros((4, cfg.window, 1))  # (batch, window, 1)
    out = model(x)
    assert out.shape == (4,)  # scalar per sample


def test_deterministic_init_same_seed():
    cfg = DLConfig(window=5, hidden_size=8, seed=123)
    m1 = build_model(cfg)
    m2 = build_model(cfg)
    p1 = torch.cat([p.flatten() for p in m1.parameters()])
    p2 = torch.cat([p.flatten() for p in m2.parameters()])
    assert torch.allclose(p1, p2)  # same seed => identical initial weights


def test_different_seed_differs():
    a = build_model(DLConfig(window=5, hidden_size=8, seed=1))
    b = build_model(DLConfig(window=5, hidden_size=8, seed=2))
    pa = torch.cat([p.flatten() for p in a.parameters()])
    pb = torch.cat([p.flatten() for p in b.parameters()])
    assert not torch.allclose(pa, pb)


def test_model_module_has_no_top_level_torch():
    import quant.intraday.dl.model as m

    # torch must be imported lazily inside build_model, not at module top.
    assert not hasattr(m, "torch")
