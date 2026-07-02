"""Streamlit dashboard for single-run and 10-fold GNN training results."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.reporting.cv_tables import (
    build_detailed_metric_table,
    build_main_comparison_table,
    export_markdown_table,
)
from src.training.cv_summary import build_comparison_csv


DATASETS = [
    {"dataset": "Solubility_AqSolDB", "task": "regression"},
    {"dataset": "Lipophilicity_AstraZeneca", "task": "regression"},
    {"dataset": "BBB_Martins", "task": "classification"},
    {"dataset": "hERG_Karim", "task": "classification"},
    {"dataset": "AMES", "task": "classification"},
]

MODEL_TYPES = ["gine", "attentivefp", "dmpnn", "cmpnn"]


def _read_json(path: Path) -> dict[str, Any] | None:
    """Read a JSON file without breaking the dashboard on malformed data."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        st.warning(f"{path} 파일을 읽지 못했습니다: {exc}")
        return None


def _normalise_history(metrics: dict[str, Any] | None) -> pd.DataFrame:
    """Convert old/new metrics.json history formats into a DataFrame."""
    if not metrics:
        return pd.DataFrame()
    raw_history = metrics.get("history") or []
    if not isinstance(raw_history, list):
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(raw_history, start=1):
        if not isinstance(item, dict):
            continue
        row = {"epoch": item.get("epoch", idx)}
        for key, value in item.items():
            if key == "valid" and isinstance(value, dict):
                for metric_name, metric_value in value.items():
                    row[f"valid_{metric_name}"] = metric_value
            elif key == "best_metric":
                row["best_valid_metric_so_far"] = value
            else:
                row[key] = value
        rows.append(row)
    return pd.DataFrame(rows)


def _get_test_metrics(metrics: dict[str, Any] | None) -> dict[str, Any]:
    """Return test metrics from old or new metrics.json structures."""
    if not metrics:
        return {}
    test_metrics = metrics.get("test_metrics")
    if isinstance(test_metrics, dict):
        return test_metrics
    legacy_test = metrics.get("test")
    if isinstance(legacy_test, dict):
        return legacy_test
    return {}


def _get_best_epoch(metrics: dict[str, Any] | None, history: pd.DataFrame, task: str) -> Any:
    """Infer best epoch when it is missing from metrics.json."""
    if metrics and metrics.get("best_epoch") is not None:
        return metrics.get("best_epoch")
    if history.empty:
        return "-"
    if task == "regression" and "valid_mae" in history and history["valid_mae"].notna().any():
        return int(history.loc[history["valid_mae"].idxmin(), "epoch"])
    if task == "classification" and "valid_auroc" in history and history["valid_auroc"].notna().any():
        return int(history.loc[history["valid_auroc"].idxmax(), "epoch"])
    return "-"


def _get_best_metric(metrics: dict[str, Any] | None, history: pd.DataFrame, task: str) -> Any:
    """Infer best validation metric when it is missing from metrics.json."""
    if metrics and metrics.get("best_valid_metric") is not None:
        return metrics.get("best_valid_metric")
    if history.empty:
        return "-"
    if task == "regression" and "valid_mae" in history and history["valid_mae"].notna().any():
        return float(history["valid_mae"].min())
    if task == "classification" and "valid_auroc" in history and history["valid_auroc"].notna().any():
        return float(history["valid_auroc"].max())
    return "-"


def _format_value(value: Any) -> str:
    """Format optional numeric values for dashboard display."""
    if value is None or value == "-":
        return "-"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _plot_line(history: pd.DataFrame, columns: list[str], title: str) -> None:
    """Render a line chart using Plotly with Streamlit fallback."""
    available = [column for column in columns if column in history.columns and history[column].notna().any()]
    if not available:
        st.info(f"{title} 데이터가 없습니다.")
        return
    chart_data = history[["epoch", *available]].copy()
    try:
        import plotly.express as px

        long_df = chart_data.melt(id_vars="epoch", value_vars=available, var_name="metric", value_name="value")
        fig = px.line(long_df, x="epoch", y="value", color="metric", markers=True, title=title)
        fig.update_layout(legend_title_text="metric", height=340)
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.line_chart(chart_data.set_index("epoch"))


def _config_frame(config: dict[str, Any] | None) -> pd.DataFrame:
    """Build a compact config display table."""
    if not config:
        return pd.DataFrame([{"항목": "config", "값": "config 없음"}])
    keys = [
        "dataset_name",
        "dataset",
        "task_type",
        "task",
        "model_type",
        "hidden_dim",
        "num_layers",
        "dropout",
        "learning_rate",
        "batch_size",
        "atom_feature_dim",
        "bond_feature_dim",
        "target_name",
    ]
    rows = [{"항목": key, "값": config.get(key)} for key in keys if key in config]
    return pd.DataFrame(rows or [{"항목": "config", "값": "표시할 config 항목이 없습니다."}])


def _effective_config(metrics: dict[str, Any] | None, config: dict[str, Any] | None) -> dict[str, Any] | None:
    """Use config.json first, then fall back to legacy metrics['config']."""
    if config:
        return config
    if metrics and isinstance(metrics.get("config"), dict):
        return metrics["config"]
    return None


def _modified_time(path: Path) -> str:
    """Return a readable modified time for a file."""
    if not path.exists():
        return "-"
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")


def _summary_row(root: Path, dataset: str, task: str) -> dict[str, Any]:
    """Build one single-run summary table row."""
    dataset_dir = root / dataset
    metrics_path = dataset_dir / "metrics.json"
    metrics = _read_json(metrics_path)
    config = _read_json(dataset_dir / "config.json")
    effective_config = _effective_config(metrics, config)
    history = _normalise_history(metrics)
    test_metrics = _get_test_metrics(metrics)
    best_epoch = _get_best_epoch(metrics, history, task)
    best_metric = _get_best_metric(metrics, history, task)
    test_key = "mae" if task == "regression" else "auroc"
    checkpoint_exists = (dataset_dir / "best.pt").exists()
    return {
        "Dataset": dataset,
        "Task": effective_config.get("task_type", task) if effective_config else task,
        "Status": metrics.get("status", "-") if metrics else "-",
        "Checkpoint": "존재" if checkpoint_exists else "없음",
        "Epochs": len(history),
        "Best epoch": best_epoch,
        "Best valid metric": _format_value(best_metric),
        "Test MAE or AUROC": _format_value(test_metrics.get(test_key)),
        "Last update": _modified_time(metrics_path),
        "Predictor source available": "GNN 사용 가능" if checkpoint_exists and effective_config else "Dummy fallback",
    }


def _render_dataset_card(root: Path, dataset: str, task: str) -> None:
    """Render status, config, metrics, and charts for one single-run dataset."""
    dataset_dir = root / dataset
    checkpoint_path = dataset_dir / "best.pt"
    metrics_path = dataset_dir / "metrics.json"
    config_path = dataset_dir / "config.json"
    metrics = _read_json(metrics_path)
    config = _read_json(config_path)
    effective_config = _effective_config(metrics, config)
    history = _normalise_history(metrics)
    test_metrics = _get_test_metrics(metrics)
    best_epoch = _get_best_epoch(metrics, history, task)
    best_metric = _get_best_metric(metrics, history, task)

    with st.expander(f"{dataset} ({task})", expanded=False):
        if metrics is None:
            st.info("아직 학습 결과가 없습니다. 해당 endpoint는 현재 Dummy Predictor로 fallback됩니다.")

        status_cols = st.columns(5)
        status_cols[0].metric("Status", metrics.get("status", "-") if metrics else "-")
        status_cols[1].metric("Checkpoint", "존재" if checkpoint_path.exists() else "없음")
        status_cols[2].metric("Metrics file", "존재" if metrics_path.exists() else "없음")
        status_cols[3].metric("Epochs", str(len(history)))
        status_cols[4].metric("Best epoch", str(best_epoch))

        st.markdown("**체크포인트 상태**")
        status_table = pd.DataFrame(
            [
                {"항목": "Dataset name", "값": dataset},
                {"항목": "Task type", "값": effective_config.get("task_type", task) if effective_config else task},
                {"항목": "Best valid metric", "값": _format_value(best_metric)},
                {"항목": "Last metrics update", "값": _modified_time(metrics_path)},
                {"항목": "Test metric 요약", "값": ", ".join(f"{k}={_format_value(v)}" for k, v in test_metrics.items()) or "-"},
                {
                    "항목": "Predictor 상태",
                    "값": "GNN 사용 가능" if checkpoint_path.exists() and effective_config else "Dummy Predictor로 fallback",
                },
            ]
        )
        st.dataframe(status_table, use_container_width=True, hide_index=True)

        left, right = st.columns([1, 1])
        with left:
            st.markdown("**Config**")
            st.dataframe(_config_frame(effective_config), use_container_width=True, hide_index=True)
        with right:
            st.markdown("**Test metrics**")
            if test_metrics:
                st.dataframe(pd.DataFrame([test_metrics]), use_container_width=True, hide_index=True)
            else:
                st.info("학습 중이거나 test metric이 아직 없습니다.")

        st.markdown("**학습 곡선**")
        if history.empty:
            st.info("history가 없어 그래프를 표시할 수 없습니다.")
            return
        _plot_line(history, ["train_loss", "valid_loss"], "Train loss / Valid loss")
        st.markdown("**검증 성능**")
        if task == "regression":
            _plot_line(history, ["valid_mae"], "Valid MAE")
            _plot_line(history, ["valid_rmse"], "Valid RMSE")
            _plot_line(history, ["valid_r2"], "Valid R2")
        else:
            _plot_line(history, ["valid_auroc"], "Valid AUROC")
            _plot_line(history, ["valid_auprc"], "Valid AUPRC")
            _plot_line(history, ["valid_f1"], "Valid F1")
            _plot_line(history, ["valid_accuracy"], "Valid Accuracy")


def _load_cv_summary(cv_root: Path, dataset: str, model_type: str) -> dict[str, Any] | None:
    """Load one CV summary file."""
    return _read_json(cv_root / dataset / model_type / "cv_summary.json")


def _cv_summary_frame(cv_root: Path) -> pd.DataFrame:
    """Load or rebuild the 10-fold comparison CSV."""
    comparison_path = Path("experiments") / "gnn_10fold_comparison.csv"
    if comparison_path.exists():
        try:
            return pd.read_csv(comparison_path)
        except Exception as exc:
            st.warning(f"CV comparison CSV를 읽지 못했습니다: {exc}")
    try:
        return build_comparison_csv(cv_root=str(cv_root), output_path=str(comparison_path))
    except Exception as exc:
        st.warning(f"CV summary를 생성하지 못했습니다: {exc}")
        return pd.DataFrame()


def _render_cv_dashboard(cv_root: Path) -> None:
    """Render 10-fold cross-validation comparison results."""
    st.subheader("10-Fold CV 비교")
    st.caption("5개 endpoint와 4개 GNN 계열(GINEConv, AttentiveFP, D-MPNN, CMPNN)의 fold별 성능을 비교합니다.")

    summary_df = _cv_summary_frame(cv_root)
    if summary_df.empty:
        st.info("아직 10-Fold CV 학습 결과가 없습니다.")
        st.code(
            "python scripts/run_10fold_gnn_cv.py --all --epochs 5 --device cpu --skip-existing",
            language="bash",
        )
        return

    total_expected = len(DATASETS) * len(MODEL_TYPES)
    completed_runs = len(summary_df)
    total_completed_folds = int(pd.to_numeric(summary_df.get("completed_folds", 0), errors="coerce").fillna(0).sum())
    progress = min(completed_runs / total_expected, 1.0) if total_expected else 0.0
    st.progress(progress, text=f"CV run 완료: {completed_runs}/{total_expected}, fold 완료 합계: {total_completed_folds}")

    main_table = build_main_comparison_table(summary_df)
    st.markdown("**모델 비교 요약**")
    st.dataframe(main_table, use_container_width=True, hide_index=True)
    st.download_button(
        "요약 CSV 다운로드",
        data=summary_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="gnn_10fold_comparison.csv",
        mime="text/csv",
    )
    st.download_button(
        "요약 Markdown 다운로드",
        data=export_markdown_table(main_table).encode("utf-8"),
        file_name="gnn_10fold_comparison.md",
        mime="text/markdown",
    )

    dataset_names = [item["dataset"] for item in DATASETS]
    selected_dataset = st.selectbox("상세 확인 dataset", dataset_names)
    detail_table = build_detailed_metric_table(summary_df, selected_dataset)
    st.markdown("**Dataset별 상세 metric**")
    st.dataframe(detail_table, use_container_width=True, hide_index=True)

    for model_type in MODEL_TYPES:
        summary = _load_cv_summary(cv_root, selected_dataset, model_type)
        label = model_type.upper()
        with st.expander(f"{selected_dataset} / {label}", expanded=False):
            if not summary:
                st.info("아직 학습 결과가 없습니다.")
                continue
            fold_metrics = summary.get("fold_metrics") or []
            if not fold_metrics:
                st.warning("fold metric이 없습니다.")
                continue
            fold_df = pd.DataFrame(fold_metrics)
            st.dataframe(fold_df, use_container_width=True, hide_index=True)

            task = str(summary.get("task", "regression"))
            chart_metrics = ["mae", "rmse", "r2"] if task == "regression" else ["auroc", "auprc", "f1", "accuracy"]
            chart_cols = [metric for metric in chart_metrics if metric in fold_df.columns]
            if chart_cols and "fold" in fold_df.columns:
                chart_df = fold_df[["fold", *chart_cols]].copy()
                try:
                    import plotly.express as px

                    long_df = chart_df.melt(id_vars="fold", value_vars=chart_cols, var_name="metric", value_name="value")
                    fig = px.line(long_df, x="fold", y="value", color="metric", markers=True, title=f"{label} fold별 metric")
                    fig.update_layout(height=340)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    st.line_chart(chart_df.set_index("fold"))

            failed_folds = summary.get("failed_folds") or []
            if failed_folds:
                st.warning(f"실패 fold: {failed_folds}")


def _server_side_auto_refresh(interval_seconds: int) -> None:
    """Force a Streamlit rerun after an interval."""
    with st.spinner(f"{interval_seconds}초 뒤 학습 현황을 새로고침합니다..."):
        time.sleep(interval_seconds)
    st.rerun()


def render_training_dashboard(checkpoint_root: str = "checkpoints") -> None:
    """Render the GNN training dashboard in Streamlit."""
    root = Path(checkpoint_root)
    cv_root = Path("checkpoints_cv")
    st.title("GNN 학습 결과 대시보드")
    st.caption("endpoint-specific GNN expert의 checkpoint, config, 학습 곡선, 검증 성능, test metric을 확인합니다.")

    with st.sidebar:
        st.divider()
        st.subheader("Dashboard")
        auto_refresh = st.checkbox("학습 현황 자동 새로고침", value=False)
        refresh_seconds = st.slider("새로고침 간격(초)", min_value=2, max_value=60, value=10, step=1)
        if auto_refresh:
            st.caption(f"{refresh_seconds}초마다 dashboard를 새로고침합니다.")

    single_tab, cv_tab = st.tabs(["Single Run Results", "10-Fold CV Comparison"])
    with single_tab:
        summary = pd.DataFrame([_summary_row(root, item["dataset"], item["task"]) for item in DATASETS])
        st.subheader("5개 모델 비교 요약")
        st.dataframe(summary, use_container_width=True, hide_index=True)

        st.subheader("Endpoint별 상세 결과")
        for item in DATASETS:
            _render_dataset_card(root, item["dataset"], item["task"])

    with cv_tab:
        _render_cv_dashboard(cv_root)

    if auto_refresh:
        _server_side_auto_refresh(refresh_seconds)
