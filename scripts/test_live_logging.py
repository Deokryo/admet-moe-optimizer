"""Smoke tests for live logging file helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.live_logging import atomic_write_json, append_jsonl, read_json_safe, read_jsonl_safe


def test_atomic_json_roundtrip() -> None:
    """atomic_write_json and read_json_safe should round-trip dictionaries."""
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "run_status.json"
        atomic_write_json(path, {"status": "running", "epoch": 1})
        assert read_json_safe(path) == {"status": "running", "epoch": 1}


def test_jsonl_skips_bad_lines() -> None:
    """read_jsonl_safe should ignore partial or invalid JSONL lines."""
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "live_metrics.jsonl"
        append_jsonl(path, {"epoch": 1, "train_loss": 0.9})
        with path.open("a", encoding="utf-8") as handle:
            handle.write("{broken\n")
        append_jsonl(path, {"epoch": 2, "train_loss": 0.8})
        rows = read_jsonl_safe(path)
        assert [row["epoch"] for row in rows] == [1, 2]


if __name__ == "__main__":
    test_atomic_json_roundtrip()
    test_jsonl_skips_bad_lines()
    print("test_live_logging.py: ok")
