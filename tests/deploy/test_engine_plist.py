"""The engine launchd agent is wired, read-only, and its run-state is gitignored."""

from __future__ import annotations

import plistlib
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PLIST = REPO / "deploy" / "com.ajaiupadhyaya.quant-engine.plist"


def test_plist_exists_and_parses() -> None:
    with PLIST.open("rb") as fh:
        data = plistlib.load(fh)
    assert data["Label"] == "com.ajaiupadhyaya.quant-engine"
    assert data["ProgramArguments"][-2:] == ["engine", "run"]  # the read-only loop
    assert data["KeepAlive"] is True  # supervised: restarts if it exits
    assert data["RunAtLoad"] is True


def test_install_script_loads_engine() -> None:
    txt = (REPO / "deploy" / "install.sh").read_text(encoding="utf-8")
    assert "com.ajaiupadhyaya.quant-engine" in txt


def test_engine_state_dir_is_gitignored() -> None:
    out = subprocess.run(
        ["git", "check-ignore", "data/engine/state.json"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0
