"""Immutable data snapshot manifests for reproducible validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class SnapshotSymbol:
    symbol: str
    path: str
    sha256: str
    rows: int
    data_start: str | None
    data_end: str | None


@dataclass(frozen=True)
class DataSnapshotManifest:
    snapshot_id: str
    created_at: str
    requested_start: str
    requested_end: str
    symbols: dict[str, SnapshotSymbol]

    def to_json_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["symbols"] = {
            symbol: asdict(symbol_manifest)
            for symbol, symbol_manifest in sorted(self.symbols.items())
        }
        return payload


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _symbol_manifest(data_dir: Path, symbol: str) -> SnapshotSymbol:
    path = data_dir / "raw" / f"{symbol}.parquet"
    if not path.exists():
        return SnapshotSymbol(
            symbol=symbol, path=str(path), sha256="", rows=0, data_start=None, data_end=None
        )
    df = pd.read_parquet(path)
    idx = pd.DatetimeIndex(df.index)
    return SnapshotSymbol(
        symbol=symbol,
        path=str(path),
        sha256=_sha256(path),
        rows=len(df),
        data_start=None if df.empty else idx.min().date().isoformat(),
        data_end=None if df.empty else idx.max().date().isoformat(),
    )


def create_data_snapshot(
    data_dir: Path,
    *,
    symbols: list[str],
    start: date,
    end: date,
    snapshot_id: str | None = None,
) -> DataSnapshotManifest:
    now = datetime.now(UTC).replace(microsecond=0)
    snapshot_id = snapshot_id or f"snapshot-{now:%Y%m%d%H%M%S}"
    manifest = DataSnapshotManifest(
        snapshot_id=snapshot_id,
        created_at=now.isoformat(),
        requested_start=start.isoformat(),
        requested_end=end.isoformat(),
        symbols={symbol: _symbol_manifest(data_dir, symbol) for symbol in sorted(symbols)},
    )
    out = data_dir / "snapshots" / snapshot_id / "manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(manifest.to_json_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest
