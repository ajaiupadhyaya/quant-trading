"""Alerting: healthchecks.io liveness pings + Pushover emergency push.

HTTP is injected (get/post callables) so tests assert on calls without network.
Secret-bearing URLs (healthchecks ping URLs) are never logged — only outcomes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import requests

from quant.util.logging import logger

GetFn = Callable[[str, float], int]
PostFn = Callable[[str, dict[str, object], float], int]

_PUSHOVER_URL = "https://api.pushover.net/1/messages.json"


def _default_get(url: str, timeout: float) -> int:
    return requests.get(url, timeout=timeout).status_code


def _default_post(url: str, data: dict[str, object], timeout: float) -> int:
    return requests.post(url, data=data, timeout=timeout).status_code


@dataclass(frozen=True)
class AlertConfig:
    healthcheck_tick_url: str | None
    healthcheck_guard_url: str | None
    pushover_app_token: str | None
    pushover_user_key: str | None


class AlertClient:
    def __init__(
        self, config: AlertConfig, *, get: GetFn = _default_get, post: PostFn = _default_post
    ) -> None:
        self._cfg = config
        self._get = get
        self._post = post

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

    def send_emergency(self, title: str, message: str) -> bool:
        """Pushover Emergency (priority 2) push. Returns True iff delivered."""
        if not (self._cfg.pushover_app_token and self._cfg.pushover_user_key):
            logger.error("emergency push requested but Pushover not configured: {}", title)
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
