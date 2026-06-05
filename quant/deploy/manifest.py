"""Version-controlled job manifest: the single source of truth that replaces the
six GitHub Actions cron schedules. Editing jobs.toml is how the schedule changes.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from datetime import time
from enum import StrEnum
from pathlib import Path


class DayRule(StrEnum):
    WEEKDAYS_TRADING = "WEEKDAYS_TRADING"
    SATURDAY = "SATURDAY"
    TRADING_DAY_EVENING = "TRADING_DAY_EVENING"


class CatchUpPolicy(StrEnum):
    SAME_DAY = "SAME_DAY"
    NONE = "NONE"


@dataclass(frozen=True)
class Job:
    name: str
    trigger_et: time | None
    close_offset_min: int | None
    days: DayRule
    catch_up: CatchUpPolicy
    max_lateness: time
    max_lateness_next_day: bool
    max_runtime_s: int
    timing_critical: bool
    commands: tuple[tuple[str, ...], ...]
    commit_paths: tuple[str, ...]


@dataclass(frozen=True)
class Manifest:
    jobs: tuple[Job, ...]


def _parse_time(s: str) -> time:
    hh, mm = s.split(":")
    return time(int(hh), int(mm))


def load_manifest(path: Path) -> Manifest:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    jobs: list[Job] = []
    for entry in raw.get("job", []):
        trigger_et = _parse_time(entry["trigger_et"]) if "trigger_et" in entry else None
        close_offset = entry.get("close_offset_min")
        if (trigger_et is None) == (close_offset is None):
            raise ValueError(
                f"job {entry.get('name')!r}: set exactly one of trigger_et / close_offset_min"
            )
        commands = tuple(tuple(str(a) for a in c) for c in entry["commands"])
        if not commands or not all(isinstance(c, tuple) and c for c in commands):
            raise ValueError(
                f"job {entry.get('name')!r}: commands must be a non-empty list of arg-lists"
            )
        jobs.append(
            Job(
                name=str(entry["name"]),
                trigger_et=trigger_et,
                close_offset_min=int(close_offset) if close_offset is not None else None,
                days=DayRule(entry["days"]),
                catch_up=CatchUpPolicy(entry["catch_up"]),
                max_lateness=_parse_time(entry["max_lateness"]),
                max_lateness_next_day=bool(entry.get("max_lateness_next_day", False)),
                max_runtime_s=int(entry["max_runtime_s"]),
                timing_critical=bool(entry.get("timing_critical", False)),
                commands=commands,
                commit_paths=tuple(str(p) for p in entry.get("commit_paths", [])),
            )
        )
    names = [j.name for j in jobs]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate job names in manifest: {names}")
    return Manifest(jobs=tuple(jobs))
