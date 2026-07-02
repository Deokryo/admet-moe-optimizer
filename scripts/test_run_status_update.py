"""Smoke tests for 10-fold run status updates."""

from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.run_status import (
    initial_run_status,
    load_run_status,
    mark_fold_completed,
    mark_fold_failed,
    mark_fold_started,
    mark_run_finished,
    write_run_status,
)


def test_run_status_transitions() -> None:
    """Fold and run status helpers should produce monitor-friendly states."""
    status = initial_run_status("AMES", "classification", "cmpnn", num_folds=3, epochs=5)
    assert status["fold_status"]["0"] == "pending"

    status = mark_fold_started(status, 0)
    assert status["status"] == "running"
    assert status["current_fold"] == 0
    assert status["fold_status"]["0"] == "running"
    assert 0 not in status["pending_folds"]

    status = mark_fold_completed(status, 0, epochs=5)
    assert status["completed_folds"] == [0]
    assert status["fold_status"]["0"] == "completed"

    status = mark_fold_failed(status, 1, "boom")
    assert status["failed_folds"] == [1]
    assert status["fold_status"]["1"] == "failed"
    assert status["error"] == "boom"

    status = mark_run_finished(status, "completed", "done")
    assert status["status"] == "completed"
    assert status["finished_at"] is not None


def test_status_file_roundtrip() -> None:
    """run_status.json should be atomically writable and safely readable."""
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "run_status.json"
        status = initial_run_status("BBB_Martins", "classification", "gine", num_folds=2, epochs=1)
        write_run_status(path, status)
        loaded = load_run_status(path)
        assert loaded is not None
        assert loaded["dataset"] == "BBB_Martins"
        assert loaded["num_folds"] == 2


if __name__ == "__main__":
    test_run_status_transitions()
    test_status_file_roundtrip()
    print("test_run_status_update.py: ok")
