"""Train endpoint-specific GNN experts on TDC datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch_geometric.loader import DataLoader

from src.training.dataset_loader import load_tdc_dataset
from src.training.evaluate import evaluate_model
from src.training.featurizer import ATOM_FEATURE_DIM, BOND_FEATURE_DIM, dataframe_to_graphs
from src.training.model import MolecularGNN


DATASET_TASKS = {
    "Solubility_AqSolDB": "regression",
    "Lipophilicity_AstraZeneca": "regression",
    "BBB_Martins": "classification",
    "hERG_Karim": "classification",
    "AMES": "classification",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train an ADMET-MoE GNN expert.")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_TASKS))
    parser.add_argument("--task", choices=["regression", "classification"], default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    return parser.parse_args()


def _loss_for_task(task_type: str) -> nn.Module:
    """Return a training loss for task type."""
    if task_type == "regression":
        return nn.SmoothL1Loss()
    return nn.BCEWithLogitsLoss()


def _is_better(metric: dict[str, float | None], best_value: float | None, task_type: str) -> tuple[bool, float]:
    """Return whether validation metric improved and the comparable value."""
    if task_type == "regression":
        value = float(metric["mae"])
        return best_value is None or value < best_value, value
    auroc = metric.get("auroc")
    value = float(auroc) if auroc is not None else float(metric.get("accuracy", 0.0) or 0.0)
    return best_value is None or value > best_value, value


def _flatten_valid_metrics(metrics: dict[str, float | None], task_type: str) -> dict[str, float | None]:
    """Flatten validation metrics into stable metrics.json history keys."""
    if task_type == "regression":
        return {
            "valid_mae": metrics.get("mae"),
            "valid_rmse": metrics.get("rmse"),
            "valid_r2": metrics.get("r2"),
        }
    return {
        "valid_auroc": metrics.get("auroc"),
        "valid_auprc": metrics.get("auprc"),
        "valid_f1": metrics.get("f1"),
        "valid_accuracy": metrics.get("accuracy"),
    }


def _write_metrics(
    output_dir: Path,
    dataset: str,
    task_type: str,
    history: list[dict[str, object]],
    best_epoch: int | None,
    best_value: float | None,
    test_metrics: dict[str, float | None] | None = None,
    status: str = "running",
) -> None:
    """Write metrics.json during and after training."""
    output = {
        "dataset": dataset,
        "task": task_type,
        "status": status,
        "history": history,
        "best_epoch": best_epoch,
        "best_valid_metric": best_value,
        "test_metrics": test_metrics or {},
    }
    metrics_path = output_dir / "metrics.json"
    temp_path = output_dir / "metrics.tmp.json"
    temp_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    temp_path.replace(metrics_path)


def _evaluate_loss(model, loader, loss_fn, device: str) -> float:
    """Return average validation loss."""
    model.eval()
    total_loss = 0.0
    total_graphs = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch)
            loss = loss_fn(output.view(-1), batch.y.view(-1))
            graph_count = int(batch.num_graphs)
            total_loss += float(loss.item()) * graph_count
            total_graphs += graph_count
    return total_loss / max(total_graphs, 1)


def train_one_epoch(model, loader, optimizer, loss_fn, device: str) -> float:
    """Train for one epoch and return average loss."""
    model.train()
    total_loss = 0.0
    total_graphs = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(batch)
        loss = loss_fn(output.view(-1), batch.y.view(-1))
        loss.backward()
        optimizer.step()
        graph_count = int(batch.num_graphs)
        total_loss += float(loss.item()) * graph_count
        total_graphs += graph_count
    return total_loss / max(total_graphs, 1)


def main() -> None:
    """Run full training, validation checkpointing, and final test evaluation."""
    args = parse_args()
    task_type = args.task or DATASET_TASKS[args.dataset]
    split = load_tdc_dataset(args.dataset)

    train_graphs = dataframe_to_graphs(split.train)
    valid_graphs = dataframe_to_graphs(split.valid)
    test_graphs = dataframe_to_graphs(split.test)
    if not train_graphs or not valid_graphs or not test_graphs:
        raise RuntimeError("At least one split has no valid molecular graphs after featurization.")

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_graphs, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size, shuffle=False)

    model = MolecularGNN(
        atom_feature_dim=ATOM_FEATURE_DIM,
        bond_feature_dim=BOND_FEATURE_DIM,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = _loss_for_task(task_type)

    output_dir = Path(args.checkpoint_dir) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "dataset_name": args.dataset,
        "task_type": task_type,
        "target_name": "y",
        "atom_feature_dim": ATOM_FEATURE_DIM,
        "bond_feature_dim": BOND_FEATURE_DIM,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "output_dim": 1,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    history: list[dict[str, object]] = []
    best_value: float | None = None
    best_epoch: int | None = None
    best_path = output_dir / "best.pt"
    _write_metrics(output_dir, args.dataset, task_type, history, best_epoch, best_value, status="initializing")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, args.device)
        valid_loss = _evaluate_loss(model, valid_loader, loss_fn, args.device)
        valid_metrics = evaluate_model(model, valid_loader, args.device, task_type)
        improved, comparable = _is_better(valid_metrics, best_value, task_type)
        if improved:
            best_value = comparable
            best_epoch = epoch
            torch.save({"model_state_dict": model.state_dict(), "config": config}, best_path)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            **_flatten_valid_metrics(valid_metrics, task_type),
        }
        history.append(record)
        _write_metrics(output_dir, args.dataset, task_type, history, best_epoch, best_value, status="running")
        print(json.dumps({**record, "best_valid_metric": best_value}, ensure_ascii=False))

    checkpoint = torch.load(best_path, map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate_model(model, test_loader, args.device, task_type)
    _write_metrics(
        output_dir,
        args.dataset,
        task_type,
        history,
        best_epoch,
        best_value,
        test_metrics=test_metrics,
        status="finished",
    )
    print("Final test metrics:")
    print(json.dumps(test_metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
