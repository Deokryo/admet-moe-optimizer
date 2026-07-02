"""Run 10-fold CV benchmarks for ADMET-MoE GNN models."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.cv_split import make_cv_folds
from src.training.cv_summary import build_comparison_csv, summarize_cv_run
from src.training.dataset_loader import load_tdc_dataset
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
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint-dir", default="checkpoints_cv")
    parser.add_argument("--tdc-data-dir", default="./data")
    parser.add_argument("--split-type", choices=["scaffold", "random", "stratified"], default="scaffold")
    parser.add_argument("--num-folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--force-redownload", action="store_true")
    return parser.parse_args()


def _run_dataset_model(args: argparse.Namespace, dataset: str, model_type: str) -> None:
    """Run CV for one dataset/model pair."""
    task = args.task or DATASET_TASKS[dataset]
    split = load_tdc_dataset(dataset, data_dir=args.tdc_data_dir, force_redownload=args.force_redownload)
    frame = pd.concat([split.train, split.valid, split.test], ignore_index=True)
    if args.limit:
        frame = frame.sample(n=min(args.limit, len(frame)), random_state=args.seed).reset_index(drop=True)
    split_type = args.split_type
    if split_type == "stratified" and task != "classification":
        split_type = "random"
    folds = make_cv_folds(frame, num_folds=args.num_folds, split_type=split_type, seed=args.seed)

    root = Path(args.checkpoint_dir) / dataset / model_type
    for fold_idx, fold in enumerate(folds):
        output_dir = root / f"fold_{fold_idx}"
        if args.skip_existing and (output_dir / "metrics.json").exists():
            print(f"Skipping existing {dataset}/{model_type}/fold_{fold_idx}")
            continue
        print(f"Training {dataset} {model_type} fold {fold_idx}")
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
            device=args.device,
            seed=args.seed + fold_idx,
        )
    summarize_cv_run(args.checkpoint_dir, dataset, model_type, task, args.num_folds)
    build_comparison_csv(args.checkpoint_dir)


def main() -> None:
    """Run requested CV jobs."""
    args = parse_args()
    if args.all:
        jobs = [(dataset, model_type) for dataset in DATASET_TASKS for model_type in MODEL_TYPES]
    else:
        if not args.dataset or not args.model:
            raise SystemExit("--dataset and --model are required unless --all is used.")
        jobs = [(args.dataset, args.model)]

    for dataset, model_type in jobs:
        _run_dataset_model(args, dataset, model_type)


if __name__ == "__main__":
    main()
