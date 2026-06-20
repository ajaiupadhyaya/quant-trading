"""Alert self-test fires each configured channel and reports delivery."""

from __future__ import annotations

from quant.deploy.alerts import AlertClient, AlertConfig


def test_send_test_fires_each_configured_channel() -> None:
    posts: list[str] = []

    def fake_get(url: str, timeout: float) -> int:
        posts.append("get:" + url)
        return 200

    def fake_post(url: str, data: dict[str, object], timeout: float) -> int:
        posts.append("post:" + url)
        return 200

    def fake_post_json(url: str, payload: dict[str, object], timeout: float) -> int:
        posts.append("json:" + url)
        return 200

    cfg = AlertConfig(
        healthcheck_tick_url="https://hc/tick",
        healthcheck_guard_url=None,
        pushover_app_token="t",
        pushover_user_key="u",
        slack_webhook_url="https://slack/hook",
    )
    client = AlertClient(cfg, get=fake_get, post=fake_post, post_json=fake_post_json)
    result = client.send_test()
    assert result == {"healthchecks": True, "pushover": True, "slack": True}
    assert any("hc/tick" in p for p in posts)


def test_send_test_skips_unconfigured() -> None:
    cfg = AlertConfig(None, None, None, None, None)
    assert AlertClient(cfg).send_test() == {}


def test_send_test_reports_channel_failure() -> None:
    def bad_post(url: str, data: dict[str, object], timeout: float) -> int:
        return 500  # pushover rejects

    cfg = AlertConfig(None, None, "t", "u", None)
    client = AlertClient(cfg, post=bad_post)
    assert client.send_test() == {"pushover": False}
