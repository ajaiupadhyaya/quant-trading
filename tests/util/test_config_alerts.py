"""Alert-channel settings are optional so CI's dummy env still constructs Settings."""

from __future__ import annotations

import pytest

from quant.util.config import Settings


def test_alert_settings_default_none(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in (
        "HEALTHCHECKS_TICK_URL",
        "HEALTHCHECKS_GUARD_URL",
        "PUSHOVER_APP_TOKEN",
        "PUSHOVER_USER_KEY",
    ):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    monkeypatch.setenv("FRED_API_KEY", "f")
    # _env_file=None ignores any local .env so this tests the true "unset" default
    # (a local .env with empty `KEY=` lines yields "" via dotenv, not None).
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.healthcheck_tick_url is None
    assert s.pushover_app_token is None


def test_alert_settings_read_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    monkeypatch.setenv("FRED_API_KEY", "f")
    monkeypatch.setenv("PUSHOVER_APP_TOKEN", "tok")
    monkeypatch.setenv("HEALTHCHECKS_TICK_URL", "https://hc-ping.com/abc")
    s = Settings()  # type: ignore[call-arg]
    assert s.pushover_app_token == "tok"
    assert s.healthcheck_tick_url == "https://hc-ping.com/abc"
