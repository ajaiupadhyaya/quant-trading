"""AlertConfig.from_settings + channel introspection."""

from __future__ import annotations

from dataclasses import dataclass

from quant.deploy.alerts import AlertConfig


@dataclass
class _S:
    healthcheck_tick_url: str | None = None
    healthcheck_guard_url: str | None = None
    pushover_app_token: str | None = None
    pushover_user_key: str | None = None
    slack_webhook_url: str | None = None


def test_from_settings_copies_all_fields() -> None:
    cfg = AlertConfig.from_settings(
        _S(slack_webhook_url="https://hook", pushover_app_token="t", pushover_user_key="u")
    )
    assert cfg.slack_webhook_url == "https://hook"
    assert cfg.pushover_app_token == "t"


def test_configured_channels_and_is_configured() -> None:
    none = AlertConfig.from_settings(_S())
    assert none.configured_channels() == ()
    assert none.is_configured is False
    some = AlertConfig.from_settings(
        _S(healthcheck_tick_url="https://hc", pushover_app_token="t", pushover_user_key="u")
    )
    assert set(some.configured_channels()) == {"healthchecks", "pushover"}
    assert some.is_configured is True


def test_pushover_needs_both_token_and_key() -> None:
    # token without key is not a usable channel
    cfg = AlertConfig.from_settings(_S(pushover_app_token="t"))
    assert "pushover" not in cfg.configured_channels()
