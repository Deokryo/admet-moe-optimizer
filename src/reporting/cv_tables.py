"""Publication-style tables for GNN cross-validation results."""

from __future__ import annotations

import pandas as pd


MODEL_LABELS = {
    "gine": "GINEConv",
    "attentivefp": "AttentiveFP",
    "dmpnn": "D-MPNN",
    "cmpnn": "CMPNN",
}


def format_mean_std(mean: float | None, std: float | None, decimals: int = 3) -> str:
    """Format a metric as mean +/- std."""
    if mean is None or pd.isna(mean):
        return "-"
    if std is None or pd.isna(std):
        return f"{float(mean):.{decimals}f}"
    return f"{float(mean):.{decimals}f} ± {float(std):.{decimals}f}"


def _best_model_label(group: pd.DataFrame) -> str:
    """Return the best model label for one dataset group."""
    if "is_best_model" not in group.columns:
        return "-"
    best = group[group["is_best_model"].fillna(False).astype(bool)]
    if best.empty:
        return "-"
    model_type = str(best.iloc[0]["model_type"])
    return MODEL_LABELS.get(model_type, model_type)


def build_main_comparison_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Build a dataset x model main metric comparison table."""
    if summary_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for dataset, group in summary_df.groupby("dataset", sort=False):
        task = str(group["task"].iloc[0])
        metric = "mae" if task == "regression" else "auroc"
        row: dict[str, object] = {"Dataset": dataset, "Task": task, "Metric": metric.upper()}
        for model_type, label in MODEL_LABELS.items():
            match = group[group["model_type"] == model_type]
            if match.empty:
                row[label] = "Not trained"
            else:
                item = match.iloc[0]
                row[label] = format_mean_std(item.get(f"{metric}_mean"), item.get(f"{metric}_std"))
        row["Best"] = _best_model_label(group)
        rows.append(row)
    return pd.DataFrame(rows)


def build_detailed_metric_table(summary_df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Build a model x all metrics table for one dataset."""
    group = summary_df[summary_df["dataset"] == dataset_name]
    rows: list[dict[str, object]] = []
    for _, item in group.iterrows():
        task = str(item["task"])
        row: dict[str, object] = {
            "Model": MODEL_LABELS.get(str(item["model_type"]), str(item["model_type"])),
            "Completed folds": item.get("completed_folds", 0),
        }
        metrics = ["mae", "rmse", "r2"] if task == "regression" else ["auroc", "auprc", "f1", "accuracy"]
        for metric in metrics:
            row[metric.upper()] = format_mean_std(item.get(f"{metric}_mean"), item.get(f"{metric}_std"))
        rows.append(row)
    return pd.DataFrame(rows)


def export_markdown_table(df: pd.DataFrame) -> str:
    """Return a markdown table suitable for reports."""
    if df.empty:
        return "_No results available._"
    try:
        return df.to_markdown(index=False)
    except ImportError:
        columns = [str(column) for column in df.columns]
        lines = [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
        ]
        for _, row in df.iterrows():
            values = [str(row[column]) for column in df.columns]
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)
