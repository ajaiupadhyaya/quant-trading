"""Tests for quant.util.logging."""

from __future__ import annotations

import pytest

from quant.util.logging import configure_logging, logger


def test_logger_emits_at_configured_level(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO")
    logger.info("hello-info")
    logger.debug("hidden-debug")
    captured = capsys.readouterr()
    assert "hello-info" in captured.err
    assert "hidden-debug" not in captured.err


def test_logger_respects_lower_level(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("DEBUG")
    logger.debug("now-visible")
    captured = capsys.readouterr()
    assert "now-visible" in captured.err


def test_configure_logging_is_idempotent() -> None:
    configure_logging("INFO")
    configure_logging("INFO")  # second call must not raise
