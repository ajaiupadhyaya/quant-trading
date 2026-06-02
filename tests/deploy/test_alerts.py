"""AlertClient with an injected HTTP transport (no network in tests)."""

from __future__ import annotations

from quant.deploy.alerts import AlertClient, AlertConfig


class _Recorder:
    def __init__(self) -> None:
        self.gets: list[tuple[str, float]] = []
        self.posts: list[tuple[str, dict]] = []
        self.fail = False

    def get(self, url: str, timeout: float) -> int:
        if self.fail:
            raise OSError("network down")
        self.gets.append((url, timeout))
        return 200

    def post(self, url: str, data: dict, timeout: float) -> int:
        if self.fail:
            raise OSError("network down")
        self.posts.append((url, data))
        return 200


def _cfg() -> AlertConfig:
    return AlertConfig(
        healthcheck_tick_url="https://hc-ping.com/tick",
        healthcheck_guard_url=None,
        pushover_app_token="apptok",
        pushover_user_key="userkey",
    )


def test_ping_success_hits_url() -> None:
    r = _Recorder()
    AlertClient(_cfg(), get=r.get, post=r.post).ping_success("https://hc-ping.com/tick")
    assert r.gets and r.gets[0][0] == "https://hc-ping.com/tick"


def test_ping_success_none_url_is_noop() -> None:
    r = _Recorder()
    AlertClient(_cfg(), get=r.get, post=r.post).ping_success(None)
    assert r.gets == []


def test_ping_fail_appends_fail_path() -> None:
    r = _Recorder()
    AlertClient(_cfg(), get=r.get, post=r.post).ping_fail("https://hc-ping.com/tick", "boom")
    assert r.gets[0][0].endswith("/fail")


def test_send_emergency_builds_priority2_payload() -> None:
    r = _Recorder()
    ok = AlertClient(_cfg(), get=r.get, post=r.post).send_emergency("HALT", "drift breach")
    assert ok is True
    url, data = r.posts[0]
    assert "pushover" in url
    assert data["priority"] == 2 and data["retry"] == 60 and data["expire"] == 3600
    assert data["title"] == "HALT" and data["message"] == "drift breach"
    assert data["token"] == "apptok" and data["user"] == "userkey"


def test_send_emergency_returns_false_when_unconfigured() -> None:
    cfg = AlertConfig(None, None, None, None)
    assert (
        AlertClient(cfg, get=_Recorder().get, post=_Recorder().post).send_emergency("x", "y")
        is False
    )


def test_send_emergency_returns_false_on_network_error() -> None:
    r = _Recorder()
    r.fail = True
    assert AlertClient(_cfg(), get=r.get, post=r.post).send_emergency("x", "y") is False
