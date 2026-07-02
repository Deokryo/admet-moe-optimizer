"""Aggregate cross-validation fold metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.training.live_logging import atomic_write_json


REGRESSION_METRICS = ["mae", "rmse", "r2"]
CLASSIFICATION_METRICS = ["auroc", "auprc", "f1", "accuracy"]


def _read_json(path: Path) -> dict[str, Any] | None:
    """Read JSON or return None."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def summarize_cv_run(
    cv_root: str | Path,
    dataset: str,
    model_type: str,
    task: str,
    num_folds: int = 10,
) -> dict[str, Any]:
    """Collect fold metrics and compute mean/std summary."""
    root = Path(cv_root) / dataset / model_type
    metric_names = REGRESSION_METRICS if task == "regression" else CLASSIFICATION_METRICS
    fold_metrics: list[dict[str, Any]] = []
    failed_folds: list[int] = []

    for fold in range(num_folds):
        metrics_path = root / f"fold_{fold}" / "metrics.json"
        metrics = _read_json(metrics_path)
        test_metrics = metrics.get("test_metrics", {}) if metrics else {}
        if not test_metrics:
            failed_folds.append(fold)
            continue
        row = {"fold": fold}
        for metric in metric_names:
            row[metric] = test_metrics.get(metric)
        fold_metrics.append(row)

    metrics_mean_std: dict[str, dict[str, float | None]] = {}
    for metric in metric_names:
        values = [float(row[metric]) for row in fold_metrics if row.get(metric) is not None]
        if not values:
            metrics_mean_std[metric] = {"mean": None, "std": None}
            continue
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        metrics_mean_std[metric] = {"mean": float(np.mean(values)), "std": std}

    main_metric = "mae" if task == "regression" else "auroc"
    summary = {
        "dataset": dataset,
        "model_type": model_type,
        "task": task,
        "num_folds": num_folds,
        "completed_folds": len(fold_metrics),
        "failed_folds": failed_folds,
        "warning": "Incomplete CV run" if len(fold_metrics) < num_folds else "",
        "main_metric": main_metric,
        "main_metric_direction": "lower" if task == "regression" else "higher",
        "metrics_mean_std": metrics_mean_std,
        "fold_metrics": fold_metrics,
    }
    root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(root / "cv_summary.json", summary)
    return summary


def _summary_to_row(summary: dict[str, Any]) -> dict[str, Any]:
    """Flatten a cv_summary.json dict to a CSV row."""
    metrics = summary.get("metrics_mean_std", {})
    row: dict[str, Any] = {
        "dataset": summary.get("dataset"),
        "task": summary.get("task"),
        "model_type": summary.get("model_type"),
        "main_metric": summary.get("main_metric"),
        "main_metric_mean": metrics.get(summary.get("main_metric"), {}).get("mean"),
        "main_metric_std": metrics.get(summary.get("main_metric"), {}).get("std"),
        "completed_folds": summary.get("completed_folds", 0),
        "is_best_model": False,
    }
    for metric in [*REGRESSION_METRICS, *CLASSIFICATION_METRICS]:
        row[f"{metric}_mean"] = metrics.get(metric, {}).get("mean")
        row[f"{metric}_std"] = metrics.get(metric, {}).get("std")
    return row


def build_comparison_csv(cv_root: str | Path = "checkpoints_cv", output_path: str | Path = "experiments/gnn_10fold_comparison.csv") -> pd.DataFrame:
    """Build a flat comparison CSV from all cv_summary.json files."""
    rows = []
    for summary_path in Path(cv_root).glob("*/*/cv_summary.json"):
        summary = _read_json(summary_path)
        if summary:
            rows.append(_summary_to_row(summary))
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    for dataset, group in frame.groupby("dataset"):
        task = str(group["task"].iloc[0])
        metric = "mae" if task == "regression" else "auroc"
        ascending = task == "regression"
        sorted_group = group.sort_values([f"{metric}_mean", f"{metric}_std"], ascending=[ascending, True], na_position="last")
        if not sorted_group.empty:
            best_idx = sorted_group.index[0]
            frame.loc[best_idx, "is_best_model"] = True

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame
