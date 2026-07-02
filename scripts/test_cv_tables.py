"""Smoke tests for CV reporting tables."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reporting.cv_tables import build_detailed_metric_table, build_main_comparison_table, export_markdown_table, format_mean_std


def test_format_mean_std() -> None:
    """Mean/std formatting should be stable for reports."""
    assert format_mean_std(0.12345, 0.06789, decimals=2) == "0.12 ± 0.07"
    assert format_mean_std(None, 0.1) == "-"


def test_main_and_detail_tables() -> None:
    """Main and detailed tables should include model labels and best model."""
    summary = pd.DataFrame(
        [
            {
                "dataset": "BBB_Martins",
                "task": "classification",
                "model_type": "gine",
                "auroc_mean": 0.71,
                "auroc_std": 0.03,
                "auprc_mean": 0.66,
                "auprc_std": 0.02,
                "f1_mean": 0.65,
                "f1_std": 0.01,
                "accuracy_mean": 0.69,
                "accuracy_std": 0.02,
                "completed_folds": 10,
                "is_best_model": False,
            },
            {
                "dataset": "BBB_Martins",
                "task": "classification",
                "model_type": "cmpnn",
                "auroc_mean": 0.75,
                "auroc_std": 0.04,
                "auprc_mean": 0.70,
                "auprc_std": 0.03,
                "f1_mean": 0.68,
                "f1_std": 0.02,
                "accuracy_mean": 0.72,
                "accuracy_std": 0.02,
                "completed_folds": 10,
                "is_best_model": True,
            },
        ]
    )
    main = build_main_comparison_table(summary)
    detail = build_detailed_metric_table(summary, "BBB_Martins")
    markdown = export_markdown_table(main)

    assert main.iloc[0]["Best"] == "CMPNN"
    assert "GINEConv" in main.columns
    assert detail.iloc[0]["AUROC"] == "0.710 ± 0.030"
    assert "| Dataset" in markdown


if __name__ == "__main__":
    test_format_mean_std()
    test_main_and_detail_tables()
    print("test_cv_tables.py: ok")
