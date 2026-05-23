"""Logging setup. Single loguru sink, level controlled by configure_logging."""

from __future__ import annotations

import sys

from loguru import logger as _logger

logger = _logger


def configure_logging(level: str = "INFO") -> None:
    """Reset loguru sinks and add a single stderr sink at the requested level.

    Idempotent — safe to call repeatedly.
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
