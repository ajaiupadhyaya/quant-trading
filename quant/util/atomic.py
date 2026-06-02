"""Atomic file writes: write to a sibling temp file, then os.replace.

os.replace is an atomic rename on POSIX, so a reader never sees a half-written
file and a crash mid-write leaves either the old file or nothing — never a
truncated file. This is the durability primitive the marker/halt/status writers
use. (Mirrors the proven tmp+replace pattern in quant/intraday/data/store.py.)
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_json_atomic(path: Path, payload: object) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
