"""Alerting: healthchecks.io liveness pings + Pushover emergency push.

HTTP is injected (get/post callables) so tests assert on calls without network.
Secret-bearing URLs (healthchecks ping URLs) are never logged — only outcomes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import requests

from quant.util.logging import logger

GetFn = Callable[[str, float], int]
PostFn = Callable[[str, dict[str, object], float], int]
PostJsonFn = Callable[[str, dict[str, object], float], int]

_PUSHOVER_URL = "https://api.pushover.net/1/messages.json"


def _default_get(url: str, timeout: float) -> int:
    return requests.get(url, timeout=timeout).status_code


def _default_post(url: str, data: dict[str, object], timeout: float) -> int:
    return requests.post(url, data=data, timeout=timeout).status_code


def _default_post_json(url: str, payload: dict[str, object], timeout: float) -> int:
    # Slack Incoming Webhooks expect a JSON body, not form-encoding.
    return requests.post(url, json=payload, timeout=timeout).status_code


class _SettingsLike(Protocol):
    healthcheck_tick_url: str | None
    healthcheck_guard_url: str | None
    pushover_app_token: str | None
    pushover_user_key: str | None
    slack_webhook_url: str | None


@dataclass(frozen=True)
class AlertConfig:
    healthcheck_tick_url: str | None
    healthcheck_guard_url: str | None
    pushover_app_token: str | None
    pushover_user_key: str | None
    slack_webhook_url: str | None = None

    @classmethod
    def from_settings(cls, settings: _SettingsLike) -> "AlertConfig":
        """Build from any object exposing the five alert-setting attributes."""
        return cls(
            healthcheck_tick_url=settings.healthcheck_tick_url,
            healthcheck_guard_url=settings.healthcheck_guard_url,
            pushover_app_token=settings.pushover_app_token,
            pushover_user_key=settings.pushover_user_key,
            slack_webhook_url=settings.slack_webhook_url,
        )

    def configured_channels(self) -> tuple[str, ...]:
        """Names of the channels that have enough config to actually deliver."""
        out: list[str] = []
        if self.healthcheck_tick_url or self.healthcheck_guard_url:
            out.append("healthchecks")
        if self.pushover_app_token and self.pushover_user_key:
            out.append("pushover")
        if self.slack_webhook_url:
            out.append("slack")
        return tuple(out)

    @property
    def is_configured(self) -> bool:
        return bool(self.configured_channels())


class AlertClient:
    def __init__(
        self,
        config: AlertConfig,
        *,
        get: GetFn = _default_get,
        post: PostFn = _default_post,
        post_json: PostJsonFn = _default_post_json,
    ) -> None:
        self._cfg = config
        self._get = get
        self._post = post
        self._post_json = post_json

    def ping_success(self, url: str | None) -> None:
        if not url:
            return
        try:
            self._get(url, 10.0)
        except Exception:  # liveness ping is best-effort; a gap is itself the signal
            logger.warning("healthcheck success ping failed (name suppressed)")

    def ping_fail(self, url: str | None, body: str = "") -> None:
        if not url:
            return
        try:
            self._get(url.rstrip("/") + "/fail", 10.0)
        except Exception:
            logger.warning("healthcheck fail ping failed (name suppressed)")

    def send_slack(self, text: str, blocks: list[dict[str, object]] | None = None) -> bool:
        """Post to the Slack Incoming Webhook. No-op returning False when unset."""
        url = self._cfg.slack_webhook_url
        if not url:
            return False
        payload: dict[str, object] = {"text": text}
        if blocks is not None:
            payload["blocks"] = blocks
        try:
            status = self._post_json(url, payload, 10.0)
        except Exception as exc:
            logger.error("slack post failed to send: {!r}", exc)
            return False
        if status >= 400:
            logger.error("slack post rejected: HTTP {}", status)
            return False
        return True

    def send_test(self) -> dict[str, bool]:
        """Fire each CONFIGURED channel with a benign test payload. Returns per-channel delivery."""
        out: dict[str, bool] = {}
        channels = self._cfg.configured_channels()
        if "healthchecks" in channels:
            url = self._cfg.healthcheck_tick_url or self._cfg.healthcheck_guard_url
            try:
                if url:
                    self._get(url, 10.0)
                out["healthchecks"] = True
            except Exception:
                out["healthchecks"] = False
        if "pushover" in channels:
            out["pushover"] = self._pushover_emergency(
                "quant alert-test", "Phase 2a alert self-test — channel OK."
            )
        if "slack" in channels:
            out["slack"] = self.send_slack(":test_tube: quant alert-test — Slack channel OK.")
        return out

    def _pushover_emergency(self, title: str, message: str) -> bool:
        if not (self._cfg.pushover_app_token and self._cfg.pushover_user_key):
            return False
        payload: dict[str, object] = {
            "token": self._cfg.pushover_app_token,
            "user": self._cfg.pushover_user_key,
            "title": title,
            "message": message,
            "priority": 2,
            "retry": 60,
            "expire": 3600,
        }
        try:
            status = self._post(_PUSHOVER_URL, payload, 10.0)
        except Exception as exc:
            logger.error("emergency push failed to send: {!r}", exc)
            return False
        if status >= 400:
            logger.error("emergency push rejected: HTTP {}", status)
            return False
        return True

    def send_emergency(self, title: str, message: str) -> bool:
        """Emergency alert — fans out to every configured channel (Pushover + Slack).

        Returns True iff at least one channel delivered. The dispatcher calls this
        on a fresh halt and on MISSED_CRITICAL; wiring Slack in here is what puts
        those break-glass events on the phone without touching the dispatcher.
        """
        delivered = self._pushover_emergency(title, message)
        if self._cfg.slack_webhook_url:
            delivered = self.send_slack(f":rotating_light: *{title}*\n{message}") or delivered
        if not delivered:
            logger.error("emergency requested but no channel delivered: {}", title)
        return delivered
