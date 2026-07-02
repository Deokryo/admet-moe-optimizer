"""Run status helpers for 10-fold CV monitoring."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.training.live_logging import atomic_write_json, read_json_safe, utc_now_iso


RUN_STATUSES = {"pending", "running", "completed", "failed", "stopped"}


def make_run_id(dataset: str, model_type: str, started_at: str | None = None) -> str:
    """Build a stable run id from dataset/model/timestamp."""
    timestamp = (started_at or utc_now_iso()).replace("-", "").replace(":", "").replace("T", "_")
    return f"{dataset}_{model_type}_{timestamp}"


def initial_run_status(
    dataset: str,
    task: str,
    model_type: str,
    num_folds: int,
    epochs: int,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Create an initial run_status.json payload."""
    now = utc_now_iso()
    return {
        "dataset": dataset,
        "task": task,
        "model_type": model_type,
        "run_id": run_id or make_run_id(dataset, model_type, now),
        "status": "pending",
        "num_folds": num_folds,
        "epochs": epochs,
        "current_fold": None,
        "current_epoch": 0,
        "completed_folds": [],
        "failed_folds": [],
        "pending_folds": list(range(num_folds)),
        "fold_status": {str(fold): "pending" for fold in range(num_folds)},
        "started_at": now,
        "updated_at": now,
        "finished_at": None,
        "last_message": "Run created",
        "error": None,
    }


def load_run_status(path: str | Path) -> dict[str, Any] | None:
    """Load a run status file safely."""
    return read_json_safe(path)


def write_run_status(path: str | Path, status: dict[str, Any]) -> None:
    """Write run status atomically."""
    if status.get("status") not in RUN_STATUSES:
        status["status"] = "running"
    status["updated_at"] = utc_now_iso()
    atomic_write_json(path, status)


def update_run_status(path: str | Path, **updates: Any) -> dict[str, Any]:
    """Patch a run status file and write it atomically."""
    status = load_run_status(path) or {}
    status.update(updates)
    write_run_status(path, status)
    return status


def mark_fold_started(status: dict[str, Any], fold: int) -> dict[str, Any]:
    """Mark one fold as running."""
    pending = [item for item in status.get("pending_folds", []) if item != fold]
    status.update(
        {
            "status": "running",
            "current_fold": fold,
            "current_epoch": 0,
            "pending_folds": pending,
            "last_message": f"Fold {fold} started",
        }
    )
    status.setdefault("fold_status", {})[str(fold)] = "running"
    return status


def mark_fold_completed(status: dict[str, Any], fold: int, epochs: int) -> dict[str, Any]:
    """Mark one fold as completed."""
    completed = list(dict.fromkeys([*status.get("completed_folds", []), fold]))
    pending = [item for item in status.get("pending_folds", []) if item != fold]
    status.update(
        {
            "current_fold": fold,
            "current_epoch": epochs,
            "completed_folds": completed,
            "pending_folds": pending,
            "last_message": f"Fold {fold} completed",
        }
    )
    status.setdefault("fold_status", {})[str(fold)] = "completed"
    return status


def mark_fold_failed(status: dict[str, Any], fold: int, error: str) -> dict[str, Any]:
    """Mark one fold as failed."""
    failed = list(dict.fromkeys([*status.get("failed_folds", []), fold]))
    pending = [item for item in status.get("pending_folds", []) if item != fold]
    status.update(
        {
            "current_fold": fold,
            "failed_folds": failed,
            "pending_folds": pending,
            "last_message": f"Fold {fold} failed: {error}",
            "error": error,
        }
    )
    status.setdefault("fold_status", {})[str(fold)] = "failed"
    return status


def mark_run_finished(status: dict[str, Any], final_status: str, message: str, error: str | None = None) -> dict[str, Any]:
    """Mark a run as completed, failed, or stopped."""
    status.update(
        {
            "status": final_status,
            "finished_at": utc_now_iso(),
            "last_message": message,
            "error": error,
        }
    )
    return status
