"""Run 10-fold CV benchmarks for ADMET-MoE GNN models."""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path
import sys

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.cv_split import make_cv_folds
from src.training.cv_summary import build_comparison_csv, summarize_cv_run
from src.training.dataset_loader import load_tdc_dataset
from src.training.live_logging import utc_now_iso
from src.training.run_status import (
    initial_run_status,
    load_run_status,
    mark_fold_completed,
    mark_fold_failed,
    mark_fold_started,
    mark_run_finished,
    write_run_status,
)
from src.training.train import DATASET_TASKS, MODEL_TYPES, train_one_run


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run GNN 10-fold cross-validation.")
    parser.add_argument("--dataset", choices=sorted(DATASET_TASKS))
    parser.add_argument("--task", choices=["regression", "classification"])
    parser.add_argument("--model", choices=MODEL_TYPES)
    parser.add_argument("--all", action="store_true", help="Run all 5 datasets x 4 models.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="cpu")
    parser.add_argument("--checkpoint-dir", default="checkpoints_cv")
    parser.add_argument("--tdc-data-dir", default="./data")
    parser.add_argument("--split-type", choices=["scaffold", "random", "stratified"], default="scaffold")
    parser.add_argument("--num-folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--force-redownload", action="store_true")
    return parser.parse_args()


def _resolve_device(device: str) -> str:
    """Resolve the requested device string."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _prepare_pending_status(args: argparse.Namespace, dataset: str, model_type: str) -> None:
    """Create a pending run status so --all jobs are visible before training starts."""
    task = args.task or DATASET_TASKS[dataset]
    root = Path(args.checkpoint_dir) / dataset / model_type
    status_path = root / "run_status.json"
    existing = load_run_status(status_path)
    if existing and existing.get("status") == "running":
        return
    status = initial_run_status(dataset=dataset, task=task, model_type=model_type, num_folds=args.num_folds, epochs=args.epochs)
    status.update({"status": "pending", "last_message": "Queued in --all benchmark"})
    write_run_status(status_path, status)


def _run_dataset_model(args: argparse.Namespace, dataset: str, model_type: str) -> None:
    """Run CV for one dataset/model pair."""
    task = args.task or DATASET_TASKS[dataset]
    device = _resolve_device(args.device)
    split = load_tdc_dataset(dataset, data_dir=args.tdc_data_dir, force_redownload=args.force_redownload)
    frame = pd.concat([split.train, split.valid, split.test], ignore_index=True)
    if args.limit:
        frame = frame.sample(n=min(args.limit, len(frame)), random_state=args.seed).reset_index(drop=True)
    split_type = args.split_type
    if split_type == "stratified" and task != "classification":
        split_type = "random"
    folds = make_cv_folds(frame, num_folds=args.num_folds, split_type=split_type, seed=args.seed)

    root = Path(args.checkpoint_dir) / dataset / model_type
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / "run_status.json"
    status = load_run_status(status_path) or initial_run_status(dataset=dataset, task=task, model_type=model_type, num_folds=args.num_folds, epochs=args.epochs)
    status.update(
        {
            "dataset": dataset,
            "task": task,
            "model_type": model_type,
            "num_folds": args.num_folds,
            "epochs": args.epochs,
        }
    )
    status.update({"status": "running", "last_message": "10-fold CV run started"})
    write_run_status(status_path, status)

    for fold_idx, fold in enumerate(folds):
        output_dir = root / f"fold_{fold_idx}"
        if args.skip_existing and (output_dir / "metrics.json").exists():
            print(f"Skipping existing {dataset}/{model_type}/fold_{fold_idx}")
            status = mark_fold_completed(status, fold_idx, args.epochs)
            status["last_message"] = f"Fold {fold_idx} skipped because metrics.json already exists"
            write_run_status(status_path, status)
            continue
        print(f"Training {dataset} {model_type} fold {fold_idx}")
        status = mark_fold_started(status, fold_idx)
        write_run_status(status_path, status)

        def _status_callback(record: dict) -> None:
            status["current_fold"] = fold_idx
            status["current_epoch"] = int(record.get("epoch", 0) or 0)
            status["updated_at"] = utc_now_iso()
            status["last_message"] = f"Fold {fold_idx} epoch {status['current_epoch']}/{args.epochs} completed"
            write_run_status(status_path, status)

        try:
            train_one_run(
                train_df=fold["train"],
                valid_df=fold["valid"],
                test_df=fold["test"],
                dataset_name=dataset,
                task_type=task,
                model_type=model_type,
                output_dir=output_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
                device=device,
                seed=args.seed + fold_idx,
                live_metrics_path=output_dir / "live_metrics.jsonl",
                status_callback=_status_callback,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            print(error)
            traceback.print_exc()
            status = mark_fold_failed(status, fold_idx, error)
            write_run_status(status_path, status)
            if not args.continue_on_error:
                status = mark_run_finished(status, "failed", f"Run failed at fold {fold_idx}", error)
                write_run_status(status_path, status)
                raise
            continue

        status = mark_fold_completed(status, fold_idx, args.epochs)
        write_run_status(status_path, status)

    summary = summarize_cv_run(args.checkpoint_dir, dataset, model_type, task, args.num_folds)
    build_comparison_csv(args.checkpoint_dir)
    final_status = "failed" if summary.get("completed_folds", 0) == 0 else "completed"
    final_message = "10-fold CV run completed" if final_status == "completed" else "10-fold CV run failed"
    status = mark_run_finished(status, final_status, final_message, status.get("error"))
    write_run_status(status_path, status)


def main() -> None:
    """Run requested CV jobs."""
    args = parse_args()
    if args.all:
        jobs = [(dataset, model_type) for dataset in DATASET_TASKS for model_type in MODEL_TYPES]
        for dataset, model_type in jobs:
            _prepare_pending_status(args, dataset, model_type)
    else:
        if not args.dataset or not args.model:
            raise SystemExit("--dataset and --model are required unless --all is used.")
        jobs = [(args.dataset, args.model)]

    for dataset, model_type in jobs:
        _run_dataset_model(args, dataset, model_type)


if __name__ == "__main__":
    main()
