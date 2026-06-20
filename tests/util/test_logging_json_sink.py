"""Optional ANSI-free JSON file sink for structured logs."""

from __future__ import annotations

import json
from pathlib import Path

from quant.util.logging import configure_logging, logger


def test_json_sink_writes_ansi_free_json(tmp_path: Path) -> None:
    p = tmp_path / "out.json"
    configure_logging("INFO", json_path=p)
    logger.info("hello structured")
    configure_logging("INFO")  # detach the file sink so the handle flushes/closes
    text = p.read_text(encoding="utf-8")
    assert "\x1b[" not in text  # no ANSI escape codes leaked into the file
    record = json.loads(text.splitlines()[0])
    assert record["record"]["message"] == "hello structured"


def test_configure_logging_without_json_path_is_stderr_only(tmp_path: Path) -> None:
    # Should not raise and should not create any file.
    configure_logging("DEBUG")
    logger.debug("stderr only")
    assert not list(tmp_path.iterdir())
