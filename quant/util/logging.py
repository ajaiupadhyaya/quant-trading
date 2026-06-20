"""Logging setup. Single loguru sink, level controlled by configure_logging."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger as _logger

logger = _logger


def configure_logging(level: str = "INFO", *, json_path: str | Path | None = None) -> None:
    """Reset loguru sinks: a colorized stderr sink, plus an optional ANSI-free JSON file sink.

    The JSON sink (``serialize=True``, ``colorize=False``) gives grep/jq-able logs for the
    long-running M4 agents without ANSI escape codes leaking into the files. Idempotent.
    """
    logger.remove()
    logger.add(
        sys.stderr,
        level=level.upper(),
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> "
            "<level>{level: <8}</level> "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        colorize=True,
        backtrace=False,
        diagnose=False,
    )
    if json_path is not None:
        logger.add(
            str(json_path),
            level=level.upper(),
            serialize=True,
            colorize=False,
            backtrace=False,
            diagnose=False,
        )
