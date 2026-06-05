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


class _JsonRecorder:
    def __init__(self) -> None:
        self.posts: list[tuple[str, dict]] = []

    def post_json(self, url: str, payload: dict, timeout: float) -> int:
        self.posts.append((url, payload))
        return 200


_SLACK = "https://hooks.slack.com/services/T/B/x"


def test_send_slack_posts_json_payload() -> None:
    jr = _JsonRecorder()
    cfg = AlertConfig(None, None, None, None, slack_webhook_url=_SLACK)
    ok = AlertClient(cfg, post_json=jr.post_json).send_slack("hello", blocks=[{"type": "section"}])
    assert ok is True
    url, payload = jr.posts[0]
    assert url == _SLACK
    assert payload["text"] == "hello" and payload["blocks"] == [{"type": "section"}]


def test_send_slack_noop_when_unset() -> None:
    jr = _JsonRecorder()
    cfg = AlertConfig(None, None, None, None)  # slack_webhook_url defaults to None
    assert AlertClient(cfg, post_json=jr.post_json).send_slack("x") is False
    assert jr.posts == []


def test_send_emergency_fans_out_to_pushover_and_slack() -> None:
    r, jr = _Recorder(), _JsonRecorder()
    cfg = AlertConfig(None, None, "apptok", "userkey", slack_webhook_url=_SLACK)
    ok = AlertClient(cfg, post=r.post, post_json=jr.post_json).send_emergency("HALT", "drift")
    assert ok is True
    assert r.posts and r.posts[0][1]["priority"] == 2  # pushover form-post
    assert jr.posts and "HALT" in jr.posts[0][1]["text"]  # slack json-post


def test_send_emergency_slack_only_still_delivers() -> None:
    jr = _JsonRecorder()
    cfg = AlertConfig(None, None, None, None, slack_webhook_url=_SLACK)
    ok = AlertClient(cfg, post_json=jr.post_json).send_emergency("HALT", "drift")
    assert ok is True
    assert jr.posts and "drift" in jr.posts[0][1]["text"]
