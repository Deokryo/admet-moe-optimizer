"""Streamlit dashboard for single-run, 10-fold CV, and live GNN training results."""

from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from src.reporting.cv_tables import (
    build_detailed_metric_table,
    build_main_comparison_table,
    export_markdown_table,
)
from src.training.cv_summary import CLASSIFICATION_METRICS, REGRESSION_METRICS, build_comparison_csv
from src.training.live_logging import atomic_write_json, read_json_safe, read_jsonl_safe, utc_now_iso


DATASETS = [
    {"dataset": "Solubility_AqSolDB", "task": "regression"},
    {"dataset": "Lipophilicity_AstraZeneca", "task": "regression"},
    {"dataset": "BBB_Martins", "task": "classification"},
    {"dataset": "hERG_Karim", "task": "classification"},
    {"dataset": "AMES", "task": "classification"},
]
DATASET_TASKS = {item["dataset"]: item["task"] for item in DATASETS}
MODEL_TYPES = ["gine", "attentivefp", "dmpnn", "cmpnn"]
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _read_json(path: Path) -> dict[str, Any] | None:
    """Read a JSON file without breaking the dashboard on malformed data."""
    data = read_json_safe(path)
    if data is None and path.exists():
        st.info(f"{path.name} 파일을 갱신 중입니다. 잠시 후 다시 읽습니다.")
    return data


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


def _format_mean_std(values: list[float]) -> str:
    """Format interim mean/std for completed folds."""
    if not values:
        return "-"
    mean = float(np.mean(values))
    if len(values) < 2:
        return f"{mean:.4f} ± N/A"
    return f"{mean:.4f} ± {float(np.std(values, ddof=1)):.4f}"


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


def _run_status_path(cv_root: Path, dataset: str, model_type: str) -> Path:
    """Return the run_status.json path for a dataset/model pair."""
    return cv_root / dataset / model_type / "run_status.json"


def _is_running(status: dict[str, Any] | None) -> bool:
    """Return whether a run status indicates active training."""
    return bool(status and status.get("status") == "running")


def _is_stale(status: dict[str, Any] | None, stale_seconds: int = 600) -> bool:
    """Heuristically detect stale running logs."""
    if not _is_running(status):
        return False
    updated_at = status.get("updated_at")
    if not updated_at:
        return False
    try:
        updated = datetime.fromisoformat(str(updated_at))
    except ValueError:
        return False
    return (datetime.utcnow() - updated).total_seconds() > stale_seconds


def _start_cv_process(command: list[str], log_path: Path) -> None:
    """Start a CV training subprocess and redirect output to a log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("a", encoding="utf-8")
    subprocess.Popen(command, cwd=PROJECT_ROOT, stdout=handle, stderr=subprocess.STDOUT)
    handle.close()


def _render_cv_launcher(cv_root: Path) -> None:
    """Render controls that start CV runs in a separate Python process."""
    st.markdown("### 10-fold CV 학습 실행")
    st.caption(
        "10-fold CV 학습은 별도 Python process에서 실행되며, app은 checkpoint/log 파일을 읽어 진행 상황을 모니터링합니다. "
        "브라우저를 닫아도 학습 process가 계속 실행될 수 있습니다."
    )

    left, right = st.columns(2)
    with left:
        dataset = st.selectbox("Dataset", [item["dataset"] for item in DATASETS], key="cv_launch_dataset")
        model_type = st.selectbox("Model", MODEL_TYPES, key="cv_launch_model")
        epochs = st.number_input("Epochs", min_value=1, max_value=1000, value=100, step=1, key="cv_launch_epochs")
        batch_size = st.number_input("Batch size", min_value=1, max_value=1024, value=64, step=1, key="cv_launch_batch")
    with right:
        device = st.selectbox("Device", ["auto", "cuda", "cpu"], key="cv_launch_device")
        split_type = st.selectbox("Split type", ["scaffold", "random", "stratified"], key="cv_launch_split")
        num_folds = st.number_input("Num folds", min_value=2, max_value=20, value=10, step=1, key="cv_launch_num_folds")
        limit_value = st.number_input("Limit for debug (0 = 전체)", min_value=0, max_value=1_000_000, value=0, step=100, key="cv_launch_limit")

    skip_existing = st.checkbox("Skip existing folds", value=True, key="cv_launch_skip")
    continue_on_error = st.checkbox("Continue on fold error", value=True, key="cv_launch_continue")

    status_path = _run_status_path(cv_root, dataset, model_type)
    status = read_json_safe(status_path)
    if _is_running(status):
        st.warning(f"{dataset}/{model_type} 학습이 이미 running 상태입니다. 새 run을 시작하지 않습니다.")

    launch_cols = st.columns([1, 1])
    with launch_cols[0]:
        if st.button("Start 10-fold CV Run", disabled=_is_running(status), type="primary"):
            command = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "run_10fold_gnn_cv.py"),
                "--dataset",
                dataset,
                "--model",
                model_type,
                "--epochs",
                str(int(epochs)),
                "--device",
                device,
                "--batch-size",
                str(int(batch_size)),
                "--split-type",
                split_type,
                "--num-folds",
                str(int(num_folds)),
            ]
            if limit_value:
                command.extend(["--limit", str(int(limit_value))])
            if skip_existing:
                command.append("--skip-existing")
            if continue_on_error:
                command.append("--continue-on-error")
            _start_cv_process(command, cv_root / dataset / model_type / "run.log")
            st.success(f"{dataset}/{model_type} 10-fold CV process를 시작했습니다.")

    with launch_cols[1]:
        confirm_full = st.checkbox("전체 benchmark 실행 확인", value=False)
        full_running = any(_is_running(read_json_safe(_run_status_path(cv_root, item["dataset"], model))) for item in DATASETS for model in MODEL_TYPES)
        if st.button("Start full benchmark", disabled=(not confirm_full or full_running)):
            command = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "run_10fold_gnn_cv.py"),
                "--all",
                "--epochs",
                str(int(epochs)),
                "--device",
                device,
                "--batch-size",
                str(int(batch_size)),
                "--split-type",
                split_type,
                "--num-folds",
                str(int(num_folds)),
            ]
            if limit_value:
                command.extend(["--limit", str(int(limit_value))])
            if skip_existing:
                command.append("--skip-existing")
            if continue_on_error:
                command.append("--continue-on-error")
            log_path = PROJECT_ROOT / "logs" / f"full_benchmark_{utc_now_iso().replace(':', '')}.log"
            _start_cv_process(command, log_path)
            st.success("전체 10-fold benchmark process를 시작했습니다.")


def _fold_status_frame(cv_root: Path, dataset: str, model_type: str, status: dict[str, Any] | None) -> pd.DataFrame:
    """Build a fold-level live status table."""
    num_folds = int((status or {}).get("num_folds", 10))
    task = str((status or {}).get("task", DATASET_TASKS.get(dataset, "regression")))
    main_metric = "mae" if task == "regression" else "auroc"
    fold_status = (status or {}).get("fold_status", {})
    rows = []
    for fold in range(num_folds):
        fold_dir = cv_root / dataset / model_type / f"fold_{fold}"
        metrics = read_json_safe(fold_dir / "metrics.json")
        live_rows = read_jsonl_safe(fold_dir / "live_metrics.jsonl")
        best_metric = metrics.get("best_valid_metric") if metrics else None
        test_metrics = metrics.get("test_metrics", {}) if metrics else {}
        rows.append(
            {
                "Fold": fold,
                "Status": fold_status.get(str(fold), "pending"),
                "Best epoch": metrics.get("best_epoch") if metrics else "-",
                "Best valid metric": _format_value(best_metric),
                f"Test {main_metric.upper()}": _format_value(test_metrics.get(main_metric)),
                "Last updated": live_rows[-1].get("timestamp") if live_rows else _modified_time(fold_dir / "metrics.json"),
            }
        )
    return pd.DataFrame(rows)


def _completed_fold_metrics(cv_root: Path, dataset: str, model_type: str, task: str, num_folds: int) -> pd.DataFrame:
    """Return test metrics for completed folds."""
    metric_names = REGRESSION_METRICS if task == "regression" else CLASSIFICATION_METRICS
    rows: list[dict[str, Any]] = []
    for fold in range(num_folds):
        metrics = read_json_safe(cv_root / dataset / model_type / f"fold_{fold}" / "metrics.json")
        test_metrics = metrics.get("test_metrics", {}) if metrics else {}
        if not test_metrics:
            continue
        row = {"fold": fold}
        for metric in metric_names:
            row[metric] = test_metrics.get(metric)
        rows.append(row)
    return pd.DataFrame(rows)


def _render_interim_summary(completed_df: pd.DataFrame, task: str) -> None:
    """Render interim mean/std from completed folds."""
    if completed_df.empty:
        st.info("완료된 fold test metric이 아직 없습니다.")
        return
    metrics = REGRESSION_METRICS if task == "regression" else CLASSIFICATION_METRICS
    rows = []
    for metric in metrics:
        values = [float(value) for value in completed_df.get(metric, pd.Series(dtype=float)).dropna().tolist()]
        rows.append({"Metric": metric.upper(), "Interim mean ± std": _format_mean_std(values), "Completed folds": len(values)})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _plot_completed_fold_metrics(completed_df: pd.DataFrame, task: str) -> None:
    """Render completed fold test metrics as fold-level curves."""
    if completed_df.empty or "fold" not in completed_df.columns:
        st.info("그래프로 표시할 완료 fold test metric이 아직 없습니다.")
        return
    chart_df = completed_df.rename(columns={"fold": "epoch"})
    metrics = REGRESSION_METRICS if task == "regression" else CLASSIFICATION_METRICS
    _plot_line(chart_df, [metric for metric in metrics if metric in completed_df.columns], "Completed fold test metrics")


def _live_metrics_frame(
    cv_root: Path,
    dataset: str,
    model_type: str,
    folds: list[int],
    metric_names: list[str],
) -> pd.DataFrame:
    """Build a long DataFrame for overlaying live epoch curves across folds."""
    rows: list[dict[str, Any]] = []
    for fold in folds:
        live_rows = read_jsonl_safe(cv_root / dataset / model_type / f"fold_{fold}" / "live_metrics.jsonl")
        for row in live_rows:
            epoch = row.get("epoch")
            if epoch is None:
                continue
            for metric in metric_names:
                value = row.get(metric)
                if value is None:
                    continue
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError):
                    continue
                rows.append(
                    {
                        "epoch": int(epoch),
                        "fold": int(fold),
                        "metric": metric,
                        "value": numeric_value,
                        "series": f"fold {fold} / {metric}",
                    }
                )
    return pd.DataFrame(rows)


def _plot_live_epoch_metrics(live_df: pd.DataFrame) -> None:
    """Render live epoch metrics with a separate color per fold/metric series."""
    if live_df.empty:
        st.info("선택한 fold/metric에 대한 live epoch 로그가 아직 없습니다.")
        return
    try:
        import plotly.express as px

        fig = px.line(
            live_df,
            x="epoch",
            y="value",
            color="series",
            markers=True,
            title="Live epoch metric curves",
            labels={"epoch": "Epoch", "value": "Metric value", "series": "Fold / Metric"},
        )
        fig.update_layout(height=520, legend_title_text="Fold / Metric")
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        wide = live_df.pivot_table(index="epoch", columns="series", values="value", aggfunc="last").sort_index()
        st.line_chart(wide)


def _render_live_monitor(cv_root: Path) -> None:
    """Render live status, fold table, curves, and interim results."""
    st.markdown("### Live 10-Fold Monitor")
    left, right, third = st.columns([1, 1, 1])
    with left:
        dataset = st.selectbox("Monitor dataset", [item["dataset"] for item in DATASETS], key="cv_monitor_dataset")
    with right:
        model_type = st.selectbox("Monitor model", MODEL_TYPES, key="cv_monitor_model")
    with third:
        refresh_interval = st.selectbox("Refresh interval", ["manual", "5 seconds", "10 seconds", "30 seconds"], key="cv_monitor_refresh")
    auto_refresh = st.checkbox("Auto refresh", value=False, key="cv_monitor_auto")

    status_path = _run_status_path(cv_root, dataset, model_type)
    status = read_json_safe(status_path)
    task = str((status or {}).get("task", DATASET_TASKS.get(dataset, "regression")))
    num_folds = int((status or {}).get("num_folds", 10))
    epochs = int((status or {}).get("epochs", 1) or 1)
    current_epoch = int((status or {}).get("current_epoch", 0) or 0)
    completed_folds = list((status or {}).get("completed_folds", []))
    failed_folds = list((status or {}).get("failed_folds", []))

    if status is None:
        st.info("아직 run_status.json이 없습니다. 학습을 시작하면 이 영역에 진행 상황이 표시됩니다.")
    else:
        shown_status = str(status.get("status", "-"))
        if _is_stale(status):
            shown_status = f"{shown_status} (stale run 가능성)"
        cols = st.columns(5)
        cols[0].metric("Status", shown_status)
        cols[1].metric("Dataset", dataset)
        cols[2].metric("Model", model_type)
        cols[3].metric("Current fold", str(status.get("current_fold", "-")))
        cols[4].metric("Epoch", f"{current_epoch}/{epochs}")

        cols = st.columns(5)
        cols[0].metric("Completed folds", str(len(completed_folds)))
        cols[1].metric("Failed folds", str(len(failed_folds)))
        cols[2].metric("Started at", str(status.get("started_at", "-")))
        cols[3].metric("Updated at", str(status.get("updated_at", "-")))
        cols[4].metric("Run id", str(status.get("run_id", "-"))[:24])
        st.caption(f"Last message: {status.get('last_message', '-')}")
        if status.get("error"):
            st.error(str(status["error"]))

        progress = min(((len(completed_folds) * epochs) + current_epoch) / max(num_folds * epochs, 1), 1.0)
        st.progress(progress, text=f"Overall progress: {progress:.1%}")

        if _is_running(status) and st.button("Mark as stopped"):
            status["status"] = "stopped"
            status["finished_at"] = utc_now_iso()
            status["last_message"] = "사용자가 dashboard에서 stopped로 표시했습니다. OS process kill은 수행하지 않았습니다."
            atomic_write_json(status_path, status)
            st.warning("run_status.json을 stopped로 표시했습니다. 실제 OS process는 종료하지 않았습니다.")

    current_fold = (status or {}).get("current_fold")
    if current_fold is None:
        current_fold = 0

    st.markdown("**실시간 epoch 학습 그래프**")
    fold_options = list(range(num_folds))
    fold_mode = st.radio("Fold 비교 범위", ["Current fold", "All folds", "Select folds"], horizontal=True, key="cv_monitor_fold_mode")
    if fold_mode == "Current fold":
        selected_folds = [int(current_fold)]
    elif fold_mode == "All folds":
        selected_folds = fold_options
    else:
        selected_folds = st.multiselect("비교할 fold 선택", fold_options, default=[int(current_fold)], key="cv_monitor_selected_folds")
    if not selected_folds:
        st.warning("그래프에 표시할 fold를 하나 이상 선택해 주세요.")

    valid_metrics = ["valid_mae", "valid_rmse", "valid_r2"] if task == "regression" else ["valid_auroc", "valid_auprc", "valid_f1", "valid_accuracy"]
    main_valid_metric = "valid_mae" if task == "regression" else "valid_auroc"
    metric_options = [*valid_metrics, "valid_loss", "train_loss"]
    selected_metrics = st.multiselect(
        "그래프에 겹쳐서 볼 metric",
        metric_options,
        default=[main_valid_metric],
        key="cv_monitor_selected_metrics",
    )
    if not selected_metrics:
        st.warning("그래프에 표시할 metric을 하나 이상 선택해 주세요.")

    live_df = _live_metrics_frame(cv_root, dataset, model_type, selected_folds, selected_metrics)
    _plot_live_epoch_metrics(live_df)

    st.caption(
        "각 선은 `fold / metric` 조합입니다. fold를 여러 개 선택하면 같은 valid metric을 fold별 색상으로 겹쳐 비교할 수 있습니다."
    )

    if st.button("Manual refresh"):
        st.rerun()
    if auto_refresh and refresh_interval != "manual":
        seconds = int(refresh_interval.split()[0])
        try:
            from streamlit_autorefresh import st_autorefresh

            st_autorefresh(interval=seconds * 1000, key="cv_live_autorefresh")
        except Exception:
            st.caption("streamlit-autorefresh가 없어 fallback rerun을 사용합니다.")
            time.sleep(seconds)
            st.rerun()


def _render_cv_comparison(cv_root: Path) -> None:
    """Render finished 10-fold cross-validation comparison results."""
    st.markdown("### 10-Fold CV 비교")
    st.caption("5개 endpoint와 4개 GNN 계열(GINEConv, AttentiveFP, D-MPNN, CMPNN)의 fold별 성능을 비교합니다.")

    summary_df = _cv_summary_frame(cv_root)
    if summary_df.empty:
        st.info("아직 10-Fold CV 학습 결과가 없습니다.")
        st.code("python scripts/run_10fold_gnn_cv.py --all --epochs 5 --device cpu --skip-existing", language="bash")
        return

    total_expected = len(DATASETS) * len(MODEL_TYPES)
    completed_runs = len(summary_df)
    total_completed_folds = int(pd.to_numeric(summary_df.get("completed_folds", 0), errors="coerce").fillna(0).sum())
    progress = min(completed_runs / total_expected, 1.0) if total_expected else 0.0
    st.progress(progress, text=f"CV run 완료: {completed_runs}/{total_expected}, fold 완료 합계: {total_completed_folds}")

    main_table = build_main_comparison_table(summary_df)
    st.markdown("**모델 비교 요약**")
    st.dataframe(main_table, use_container_width=True, hide_index=True)
    st.download_button("요약 CSV 다운로드", data=summary_df.to_csv(index=False).encode("utf-8-sig"), file_name="gnn_10fold_comparison.csv", mime="text/csv")
    st.download_button("요약 Markdown 다운로드", data=export_markdown_table(main_table).encode("utf-8"), file_name="gnn_10fold_comparison.md", mime="text/markdown")

    selected_dataset = st.selectbox("상세 확인 dataset", [item["dataset"] for item in DATASETS], key="cv_compare_dataset")
    detail_table = build_detailed_metric_table(summary_df, selected_dataset)
    st.markdown("**Dataset별 상세 metric**")
    st.dataframe(detail_table, use_container_width=True, hide_index=True)

    for model_type in MODEL_TYPES:
        summary = _load_cv_summary(cv_root, selected_dataset, model_type)
        with st.expander(f"{selected_dataset} / {model_type.upper()}", expanded=False):
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
            chart_metrics = REGRESSION_METRICS if task == "regression" else CLASSIFICATION_METRICS
            chart_cols = [metric for metric in chart_metrics if metric in fold_df.columns]
            if chart_cols and "fold" in fold_df.columns:
                _plot_line(fold_df.rename(columns={"fold": "epoch"}), chart_cols, f"{model_type.upper()} fold별 metric")
            failed_folds = summary.get("failed_folds") or []
            if failed_folds:
                st.warning(f"실패 fold: {failed_folds}")


def _render_cv_dashboard(cv_root: Path) -> None:
    """Render 10-fold CV launcher, live monitor, and comparison UI."""
    _render_cv_launcher(cv_root)
    st.divider()
    _render_live_monitor(cv_root)
    st.divider()
    _render_cv_comparison(cv_root)


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
        refresh_seconds = st.slider("새로고침 간격(초)", min_value=5, max_value=60, value=10, step=5)
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
