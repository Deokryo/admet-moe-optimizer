"""Model evaluation loops."""

from __future__ import annotations

from collections.abc import Iterable

from src.training.metrics import classification_metrics, regression_metrics


def predict_loader(model, loader, device: str) -> tuple[list[float], list[float]]:
    """Collect y_true and raw model outputs for a PyG DataLoader."""
    import torch

    model.eval()
    y_true: list[float] = []
    y_pred: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch)
            y_true.extend(batch.y.view(-1).detach().cpu().tolist())
            y_pred.extend(output.view(-1).detach().cpu().tolist())
    return y_true, y_pred


def evaluate_model(model, loader: Iterable, device: str, task_type: str) -> dict[str, float | None]:
    """Evaluate a model for the given task type."""
    y_true, y_pred = predict_loader(model, loader, device)
    if task_type == "regression":
        return regression_metrics(y_true, y_pred)
    return classification_metrics(y_true, y_pred)

