"""Smoke tests for CV summary aggregation."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.cv_summary import build_comparison_csv, summarize_cv_run


def _write_metrics(path: Path, metrics: dict[str, float]) -> None:
    """Write a minimal fold metrics file."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "metrics.json").write_text(json.dumps({"test_metrics": metrics}, indent=2), encoding="utf-8")


def test_regression_summary_mean_std() -> None:
    """Regression summaries should aggregate fold metrics with sample std."""
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_metrics(root / "Solubility_AqSolDB" / "gine" / "fold_0", {"mae": 1.0, "rmse": 2.0, "r2": 0.1})
        _write_metrics(root / "Solubility_AqSolDB" / "gine" / "fold_1", {"mae": 3.0, "rmse": 4.0, "r2": 0.3})

        summary = summarize_cv_run(root, "Solubility_AqSolDB", "gine", "regression", num_folds=2)
        assert summary["completed_folds"] == 2
        assert summary["metrics_mean_std"]["mae"]["mean"] == 2.0
        assert round(summary["metrics_mean_std"]["mae"]["std"], 6) == 1.414214


def test_comparison_best_model() -> None:
    """Comparison CSV should mark lower MAE as best for regression."""
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        for model, values in {"gine": [1.0, 1.2], "cmpnn": [0.7, 0.9]}.items():
            for fold, mae in enumerate(values):
                _write_metrics(root / "Lipophilicity_AstraZeneca" / model / f"fold_{fold}", {"mae": mae, "rmse": mae + 0.1, "r2": 0.2})
            summarize_cv_run(root, "Lipophilicity_AstraZeneca", model, "regression", num_folds=2)

        frame = build_comparison_csv(root, root / "comparison.csv")
        best = frame[frame["is_best_model"] == True]  # noqa: E712
        assert best.iloc[0]["model_type"] == "cmpnn"


if __name__ == "__main__":
    test_regression_summary_mean_std()
    test_comparison_best_model()
    print("test_cv_summary.py: ok")
