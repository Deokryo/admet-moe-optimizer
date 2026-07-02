"""Safe file helpers for live training logs."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    """Return a timezone-free UTC timestamp for lightweight status files."""
    return datetime.utcnow().replace(microsecond=0).isoformat()


def atomic_write_json(path: str | Path, data: dict[str, Any]) -> None:
    """Write JSON through a temporary file and atomically replace the target."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f"{target.name}.tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temp, target)


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    """Append one JSON object to a JSONL file and flush it immediately."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_json_safe(path: str | Path) -> dict[str, Any] | None:
    """Read JSON and return None while another process is updating it."""
    target = Path(path)
    if not target.exists():
        return None
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def read_jsonl_safe(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file and skip partial or malformed lines."""
    target = Path(path)
    if not target.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        lines = target.read_text(encoding="utf-8").splitlines()
    except OSError:
        return rows
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows
