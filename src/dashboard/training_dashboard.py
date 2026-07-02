"""GNN training result dashboard."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


DATASETS = [
    {"dataset": "Solubility_AqSolDB", "task": "regression"},
    {"dataset": "Lipophilicity_AstraZeneca", "task": "regression"},
    {"dataset": "BBB_Martins", "task": "classification"},
    {"dataset": "hERG_Karim", "task": "classification"},
    {"dataset": "AMES", "task": "classification"},
]


def _read_json(path: Path) -> dict[str, Any] | None:
    """Read JSON with graceful failure."""
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
    """Render a line chart using plotly with Streamlit fallback."""
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
    """Build one summary table row."""
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
    """Render status, config, metrics, and charts for one dataset."""
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
                {"항목": "Predictor 상태", "값": "GNN 사용 가능" if checkpoint_path.exists() and effective_config else "Dummy Predictor로 fallback"},
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


def _install_auto_refresh(interval_seconds: int) -> None:
    """Install a tiny browser-side auto refresh script."""
    interval_ms = max(interval_seconds, 2) * 1000
    components.html(
        f"""
        <script>
        setTimeout(function() {{
            window.parent.location.reload();
        }}, {interval_ms});
        </script>
        """,
        height=0,
    )


def render_training_dashboard(checkpoint_root: str = "checkpoints") -> None:
    """Render the GNN training dashboard in Streamlit."""
    root = Path(checkpoint_root)
    st.title("GNN 학습 결과 대시보드")
    st.caption("endpoint-specific GNN expert의 checkpoint, config, 학습 곡선, 검증 성능, test metric을 확인합니다.")

    with st.sidebar:
        st.divider()
        st.subheader("Dashboard")
        auto_refresh = st.checkbox("학습 현황 자동 새로고침", value=False)
        refresh_seconds = st.slider("새로고침 간격(초)", min_value=2, max_value=60, value=10, step=1)
        if auto_refresh:
            _install_auto_refresh(refresh_seconds)
            st.caption(f"{refresh_seconds}초마다 dashboard를 새로고침합니다.")

    summary = pd.DataFrame([_summary_row(root, item["dataset"], item["task"]) for item in DATASETS])
    st.subheader("5개 모델 비교 요약")
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.subheader("Endpoint별 상세 결과")
    for item in DATASETS:
        _render_dataset_card(root, item["dataset"], item["task"])
