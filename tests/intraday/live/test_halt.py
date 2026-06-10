from datetime import UTC, datetime

from quant.intraday.live.halt import clear_sleeve_halt, load_sleeve_halt, set_sleeve_halt


def test_default_is_not_halted(tmp_path):
    st = load_sleeve_halt(tmp_path)
    assert st.active is False


def test_set_then_load_is_active(tmp_path):
    set_sleeve_halt(
        tmp_path, reason="daily loss breach", created_at=datetime(2026, 6, 7, tzinfo=UTC)
    )
    st = load_sleeve_halt(tmp_path)
    assert st.active is True
    assert "daily loss" in st.reason


def test_clear_then_load_is_inactive(tmp_path):
    set_sleeve_halt(tmp_path, reason="x")
    clear_sleeve_halt(tmp_path, reason="manual resume")
    assert load_sleeve_halt(tmp_path).active is False


def test_corrupt_artifact_fails_closed(tmp_path):
    path = tmp_path / "intraday" / "live" / "sleeve_halt.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json")
    assert load_sleeve_halt(tmp_path).active is True
